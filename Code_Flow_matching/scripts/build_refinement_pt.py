#!/usr/bin/env python3
"""Build schema-v2 RNA refinement samples from native/Protenix/RNA-FM data.

The predicted structure is the atom-index authority.  Native atoms are matched
by sequence alignment plus normalized atom name, rigidly aligned into the
predicted frame, and supervised only where ``target_mask`` is true.
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import io
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import gemmi
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load the small constants module directly.  Importing ``etflow`` itself also
# imports Lightning/model dependencies, which a data-preparation environment
# should not need.
_constants_spec = importlib.util.spec_from_file_location(
    "refinement_data_constants", PROJECT_ROOT / "etflow/data/constants.py"
)
if _constants_spec is None or _constants_spec.loader is None:
    raise RuntimeError("cannot load etflow/data/constants.py")
_constants = importlib.util.module_from_spec(_constants_spec)
_constants_spec.loader.exec_module(_constants)
ATOM_NAME_TO_ID = _constants.ATOM_NAME_TO_ID
RNA_RESIDUE_TO_ID = _constants.RNA_RESIDUE_TO_ID
UNKNOWN_ATOM_NAME_ID = _constants.UNKNOWN_ATOM_NAME_ID
UNKNOWN_RESIDUE_TYPE_ID = _constants.UNKNOWN_RESIDUE_TYPE_ID
normalize_atom_name = _constants.normalize_atom_name


SCHEMA_VERSION = 2
GENERATOR_VERSION = "2.2-strict-complete-sequence-mapping"
MAPPING_POLICY = "complete-canonical-sequence-exact-unique-chain-v1"
SAMPLE_CIF_RE = re.compile(r"^(?P<prefix>.+)_sample_(?P<sample>\d+)\.cif$", re.I)
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$", re.I)


@dataclass(frozen=True)
class Atom:
    row: int
    element: str
    atom_name: str
    comp_id: str
    label_chain: str
    auth_chain: str
    label_seq_id: str
    auth_seq_id: str
    ins_code: str
    alt_id: str
    occupancy: float
    xyz: tuple[float, float, float]


@dataclass
class NativeChain:
    label_chain: str
    auth_chain: str
    sequence: str
    atoms_by_residue: dict[int, dict[str, Atom]]
    sequence_source: str = "unknown"
    label_seq_ids: tuple[str, ...] = ()


def _column_values(block: gemmi.cif.Block, tag: str) -> list[str]:
    # Gemmi's CIF DOM stores lexical values, including surrounding quotes.
    # Preserve null markers because downstream selection distinguishes them.
    return [value if value in {".", "?"} else gemmi.cif.as_string(value)
            for value in block.find_values(tag)]


def _values(block: gemmi.cif.Block, tag: str, n: int, default: str = "") -> list[str]:
    values = _column_values(block, tag)
    if not values:
        return [default] * n
    if len(values) != n:
        raise ValueError(f"mmCIF column {tag} has {len(values)} rows, expected {n}")
    return values


def read_atoms(path: Path) -> tuple[gemmi.cif.Block, list[Atom]]:
    block = gemmi.cif.read_file(str(path)).sole_block()
    xs = _column_values(block, "_atom_site.Cartn_x")
    if not xs:
        raise ValueError(f"{path}: no _atom_site rows")
    n = len(xs)
    columns = {
        "ys": _values(block, "_atom_site.Cartn_y", n),
        "zs": _values(block, "_atom_site.Cartn_z", n),
        "element": _values(block, "_atom_site.type_symbol", n),
        "atom": _values(block, "_atom_site.label_atom_id", n),
        "comp": _values(block, "_atom_site.label_comp_id", n),
        "lchain": _values(block, "_atom_site.label_asym_id", n),
        "achain": _values(block, "_atom_site.auth_asym_id", n),
        "lseq": _values(block, "_atom_site.label_seq_id", n),
        "aseq": _values(block, "_atom_site.auth_seq_id", n),
        "ins": _values(block, "_atom_site.pdbx_PDB_ins_code", n, "?"),
        "alt": _values(block, "_atom_site.label_alt_id", n, "."),
        "occ": _values(block, "_atom_site.occupancy", n, "1"),
        "model": _values(block, "_atom_site.pdbx_PDB_model_num", n, "1"),
    }
    first_model = next((v for v in columns["model"] if v not in {"", ".", "?"}), "1")
    atoms: list[Atom] = []
    for i in range(n):
        if columns["model"][i] not in {first_model, "", ".", "?"}:
            continue
        atom_name = normalize_atom_name(columns["atom"][i])
        element = columns["element"][i].strip().upper()
        if element in {"H", "D"}:
            continue
        atoms.append(
            Atom(
                row=i,
                element=element,
                atom_name=atom_name,
                comp_id=columns["comp"][i].strip().upper(),
                label_chain=columns["lchain"][i],
                auth_chain=columns["achain"][i],
                label_seq_id=columns["lseq"][i],
                auth_seq_id=columns["aseq"][i],
                ins_code=columns["ins"][i],
                alt_id=columns["alt"][i],
                occupancy=float(columns["occ"][i]),
                xyz=(float(xs[i]), float(columns["ys"][i]), float(columns["zs"][i])),
            )
        )
    return block, atoms


def _comp_parent_map(block: gemmi.cif.Block) -> dict[str, str]:
    ids = _column_values(block, "_chem_comp.id")
    parents = _column_values(block, "_chem_comp.mon_nstd_parent_comp_id")
    if len(ids) != len(parents):
        return {}
    return {key.upper(): value.upper() for key, value in zip(ids, parents)}


def comp_symbol(comp_id: str, parents: dict[str, str]) -> str:
    comp_id = comp_id.upper()
    seen: set[str] = set()
    while comp_id not in seen:
        seen.add(comp_id)
        if comp_id in {"A", "C", "G", "U", "N", "X", "I", "T"}:
            return comp_id
        parent = parents.get(comp_id, "")
        if parent in {"", ".", "?"}:
            break
        # Some files list multiple comma-separated parents; those are ambiguous.
        if "," in parent:
            break
        comp_id = parent
    return "X"


def _entity_sequences(block: gemmi.cif.Block, parents: dict[str, str]) -> dict[str, tuple[list[str], list[str]]]:
    entities = _column_values(block, "_entity_poly_seq.entity_id")
    nums = _column_values(block, "_entity_poly_seq.num")
    monomers = _column_values(block, "_entity_poly_seq.mon_id")
    result: dict[str, tuple[list[str], list[str]]] = {}
    if not len(entities) == len(nums) == len(monomers):
        raise ValueError("incomplete _entity_poly_seq columns; cannot establish residue identity")
    grouped: dict[str, dict[int, str]] = defaultdict(dict)
    for entity, num, monomer in zip(entities, nums, monomers):
        position = int(num)
        old = grouped[entity].get(position)
        if old is not None and old != monomer:
            raise ValueError(f"ambiguous _entity_poly_seq entity={entity} position={position}: {old}/{monomer}")
        grouped[entity][position] = monomer
    for entity, rows in grouped.items():
        positions = sorted(rows)
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(f"incomplete _entity_poly_seq numbering for entity={entity}")
        result[entity] = ([comp_symbol(rows[p], parents) for p in positions], [str(p) for p in positions])
    return result


@functools.lru_cache(maxsize=8)
def native_chains(path: Path) -> list[NativeChain]:
    block, atoms = read_atoms(path)
    parents = _comp_parent_map(block)
    entity_sequences = _entity_sequences(block, parents)
    asym_ids = _column_values(block, "_struct_asym.id")
    entity_ids = _column_values(block, "_struct_asym.entity_id")
    asym_to_entity = dict(zip(asym_ids, entity_ids))
    polymer_entities = _column_values(block, "_entity_poly.entity_id")
    polymer_types = _column_values(block, "_entity_poly.type")
    entity_to_polymer_type = {
        entity: polymer_type.lower()
        for entity, polymer_type in zip(polymer_entities, polymer_types)
    }
    atoms_by_chain: dict[str, list[Atom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_chain[atom.label_chain].append(atom)

    result: list[NativeChain] = []
    for label_chain, chain_atoms in atoms_by_chain.items():
        entity = asym_to_entity.get(label_chain)
        polymer_type = entity_to_polymer_type.get(entity or "", "")
        if polymer_type and "ribonucleotide" not in polymer_type:
            continue
        seq_data = entity_sequences.get(entity or "")
        if seq_data:
            symbols, seq_ids = seq_data
        else:
            observed = sorted(
                {a.label_seq_id for a in chain_atoms if a.label_seq_id not in {"", ".", "?"}},
                key=lambda value: int(value),
            )
            seq_ids = observed
            first_by_residue = {
                atom.label_seq_id: atom for atom in chain_atoms if atom.label_seq_id in observed
            }
            symbols = [comp_symbol(first_by_residue[key].comp_id, parents) for key in observed]
        observed_rna_fraction = sum(
            comp_symbol(atom.comp_id, parents) != "X" for atom in chain_atoms
        ) / len(chain_atoms)
        if not symbols or (not polymer_type and observed_rna_fraction < 0.5):
            continue
        seq_to_index = {seq_id: index for index, seq_id in enumerate(seq_ids)}
        selected: dict[tuple[int, str], Atom] = {}
        for atom in chain_atoms:
            residue_index = seq_to_index.get(atom.label_seq_id)
            if residue_index is None:
                if seq_data:
                    raise ValueError(f"native chain {label_chain} atom row {atom.row} has label_seq_id={atom.label_seq_id!r} outside entity sequence")
                continue
            if seq_data and comp_symbol(atom.comp_id, parents) != symbols[residue_index]:
                raise ValueError(f"native chain {label_chain} label_seq_id={atom.label_seq_id}: atom component {atom.comp_id} disagrees with entity sequence")
            key = (residue_index, atom.atom_name)
            old = selected.get(key)
            preferred_alt = atom.alt_id in {"", ".", "?", "A"}
            old_preferred = old is not None and old.alt_id in {"", ".", "?", "A"}
            if old is None or (preferred_alt and not old_preferred) or (
                preferred_alt == old_preferred and atom.occupancy > old.occupancy
            ):
                selected[key] = atom
        by_residue: dict[int, dict[str, Atom]] = defaultdict(dict)
        for (residue_index, atom_name), atom in selected.items():
            by_residue[residue_index][atom_name] = atom
        result.append(
            NativeChain(
                label_chain=label_chain,
                auth_chain=chain_atoms[0].auth_chain,
                sequence="".join(symbols),
                atoms_by_residue=dict(by_residue),
                sequence_source="entity_poly_seq" if seq_data else "observed_atoms_only",
                label_seq_ids=tuple(seq_ids),
            )
        )
    return result


def global_align(query: str, target: str) -> tuple[dict[int, int], float, float]:
    """Diagnostic alignment only; NOT used to authorize training atom matches."""
    n, m = len(query), len(target)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diagonal, 1 up, 2 left
    for i in range(1, n + 1):
        scores[i][0], trace[i][0] = -2 * i, 1
    for j in range(1, m + 1):
        scores[0][j], trace[0][j] = -2 * j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (scores[i - 1][j - 1] + (2 if query[i - 1] == target[j - 1] else -1),
                       scores[i - 1][j] - 2, scores[i][j - 1] - 2)
            best = max(range(3), key=lambda k: choices[k])
            scores[i][j], trace[i][j] = choices[best], best
    i, j = n, m
    mapping: dict[int, int] = {}
    matches = aligned = 0
    while i or j:
        step = trace[i][j]
        if i and j and step == 0:
            mapping[i - 1] = j - 1
            aligned += 1
            matches += query[i - 1] == target[j - 1]
            i -= 1
            j -= 1
        elif i and (not j or step == 1):
            i -= 1
        else:
            j -= 1
    identity = matches / aligned if aligned else 0.0
    coverage = len(mapping) / len(query) if query else 0.0
    return mapping, identity, coverage


def _sequence_mapping_error(
    message: str,
    prediction_sequence: str,
    chains: list[NativeChain],
) -> ValueError:
    native_sequences = [
        {
            "label_chain": chain.label_chain,
            "auth_chain": chain.auth_chain,
            "sequence_source": chain.sequence_source,
            "sequence": chain.sequence,
        }
        for chain in chains
    ]
    return ValueError(
        f"{message}; prediction_sequence={prediction_sequence!r}; "
        f"native_sequences={native_sequences!r}"
    )


def choose_native_chain(chains: list[NativeChain], sequence: str, preferred_auth_chain: str = "") -> tuple[NativeChain, dict[int, int], float, float]:
    """Require declared complete sequence and unambiguous chain provenance.

    This deliberately rejects partial/approximate sequence alignments: a
    unique best alignment is not evidence that a mutated construct is the
    same experimental RNA. Missing ATOM coordinates are allowed, missing
    sequence identity information is not.
    """
    all_chains = chains
    if not sequence or any(symbol not in "ACGU" for symbol in sequence):
        raise _sequence_mapping_error(
            "strict mapping requires a nonempty canonical A/C/G/U prediction sequence",
            sequence,
            all_chains,
        )
    if preferred_auth_chain:
        chains = [chain for chain in chains if chain.auth_chain == preferred_auth_chain]
        if not chains:
            raise _sequence_mapping_error(
                f"RNA-FM original_chain_id={preferred_auth_chain!r} not found in native RNA chains",
                sequence,
                all_chains,
            )
    complete = [chain for chain in chains if chain.sequence_source == "entity_poly_seq"]
    if not complete:
        raise _sequence_mapping_error(
            "strict mapping requires native complete _entity_poly_seq; observed-only sequence cannot prove missing-residue positions",
            sequence,
            chains,
        )
    exact = [chain for chain in complete if chain.sequence == sequence]
    if not exact:
        raise _sequence_mapping_error(
            "native complete sequence differs from prediction; strict mapping rejects mismatches, insertions/deletions and unknown bases (no 80% fallback)",
            sequence,
            complete,
        )
    if len(exact) != 1:
        raise _sequence_mapping_error(
            "ambiguous native chain: multiple complete exact sequence matches; supply verified RNA-FM original_chain_ids",
            sequence,
            exact,
        )
    chain = exact[0]
    return chain, dict(enumerate(range(len(sequence)))), 1.0, 1.0


def _tensor(data: dict, key: str, dtype: torch.dtype) -> torch.Tensor:
    if key not in data:
        raise ValueError(f"full-data JSON is missing {key!r}; rerun Protenix with --need_atom_confidence true")
    return torch.as_tensor(data[key], dtype=dtype)


@functools.lru_cache(maxsize=8)
def load_rnafm(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"residue_embedding", "sequences", "chain_offsets"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing RNA-FM keys {sorted(missing)}")
    return payload


def _predicted_chain_candidates(atoms: list[Atom], atom_to_token: torch.Tensor, parents: dict[str, str]) -> list[tuple[str, list[int], str]]:
    by_chain_token: dict[str, dict[int, list[Atom]]] = defaultdict(lambda: defaultdict(list))
    for atom in atoms:
        by_chain_token[atom.label_chain][int(atom_to_token[atom.row])].append(atom)
    candidates = []
    for chain, token_atoms in by_chain_token.items():
        tokens = sorted(token_atoms)
        symbols = []
        for token in tokens:
            comp = Counter(atom.comp_id for atom in token_atoms[token]).most_common(1)[0][0]
            symbols.append(comp_symbol(comp, parents))
        candidates.append((chain, tokens, "".join(symbols)))
    return candidates


def choose_predicted_and_embedding_chain(
    block: gemmi.cif.Block,
    atoms: list[Atom],
    atom_to_token: torch.Tensor,
    rnafm: dict,
) -> tuple[str, list[int], str, torch.Tensor, str]:
    parents = _comp_parent_map(block)
    pred_candidates = _predicted_chain_candidates(atoms, atom_to_token, parents)
    sequences = [str(value).upper() for value in rnafm["sequences"]]
    offsets = torch.as_tensor(rnafm["chain_offsets"], dtype=torch.long).tolist()
    all_embeddings = torch.as_tensor(rnafm["residue_embedding"])
    if len(offsets) != len(sequences) + 1:
        raise ValueError(
            "RNA-FM chain_offsets must have one more entry than sequences: "
            f"got {len(offsets)} offsets for {len(sequences)} sequences"
        )
    if not offsets or offsets[0] != 0 or any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("RNA-FM chain_offsets must start at zero and be nondecreasing")
    if all_embeddings.ndim != 2 or offsets[-1] != len(all_embeddings):
        raise ValueError(
            "RNA-FM chain_offsets do not span residue_embedding: "
            f"last offset={offsets[-1] if offsets else None}, embedding rows={len(all_embeddings)}"
        )
    for chain_index, sequence in enumerate(sequences):
        if offsets[chain_index + 1] - offsets[chain_index] != len(sequence):
            raise ValueError(
                f"RNA-FM chain {chain_index} sequence length {len(sequence)} disagrees "
                f"with its embedding span {offsets[chain_index + 1] - offsets[chain_index]}"
            )
    expected_ids = [str(value) for value in rnafm.get("expected_protenix_chain_ids", [])]
    original_ids = [str(value) for value in rnafm.get("original_chain_ids", [])]
    matches = []
    for pred_chain, tokens, pred_sequence in pred_candidates:
        for chain_index, sequence in enumerate(sequences):
            if pred_sequence == sequence:
                id_match = int(chain_index < len(expected_ids) and pred_chain == expected_ids[chain_index])
                matches.append((id_match, pred_chain, tokens, sequence, chain_index))
    if not matches:
        details = [(chain, sequence) for chain, _, sequence in pred_candidates]
        raise ValueError(f"no exact predicted/RNA-FM sequence match; predicted={details}, RNA-FM={sequences}")
    matches.sort(reverse=True, key=lambda row: row[0])
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError("predicted/RNA-FM chain match is ambiguous")
    _, pred_chain, tokens, sequence, chain_index = matches[0]
    start, stop = offsets[chain_index], offsets[chain_index + 1]
    embedding = all_embeddings[start:stop]
    original_chain = original_ids[chain_index] if chain_index < len(original_ids) else ""
    return pred_chain, tokens, sequence, embedding, original_chain


def kabsch_align(mobile: torch.Tensor, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mobile.shape != fixed.shape or mobile.ndim != 2 or mobile.size(1) != 3 or mobile.size(0) < 3:
        raise ValueError("Kabsch alignment needs at least three paired 3D atoms")
    mobile64, fixed64 = mobile.double(), fixed.double()
    mobile_center, fixed_center = mobile64.mean(0), fixed64.mean(0)
    centered_mobile, centered_fixed = mobile64 - mobile_center, fixed64 - fixed_center
    if torch.linalg.matrix_rank(centered_mobile) < 2 or torch.linalg.matrix_rank(centered_fixed) < 2:
        raise ValueError("Kabsch atom set is collinear")
    u, _, vh = torch.linalg.svd(centered_mobile.T @ centered_fixed)
    correction = torch.eye(3, dtype=torch.float64)
    correction[-1, -1] = torch.sign(torch.det(u @ vh))
    rotation = u @ correction @ vh
    aligned = centered_mobile @ rotation + fixed_center
    rmsd = float(torch.sqrt(((aligned - fixed64) ** 2).sum(-1).mean()))
    translation = fixed_center - mobile_center @ rotation
    return aligned.float(), rotation.float(), translation.float(), rmsd


# Standard ideal RNA heavy-atom bond lengths (angstrom).  Values are template
# targets, never measurements from a native structure.
COMMON_BONDS = (
    ("P", "OP1", 1.485), ("P", "OP2", 1.485), ("P", "OP3", 1.485),
    ("P", "O5'", 1.607), ("O5'", "C5'", 1.430), ("C5'", "C4'", 1.526),
    ("C4'", "O4'", 1.453), ("C4'", "C3'", 1.526), ("C3'", "O3'", 1.420),
    ("C3'", "C2'", 1.525), ("C2'", "O2'", 1.412), ("C2'", "C1'", 1.525),
    ("C1'", "O4'", 1.415),
)
BASE_BONDS = {
    "A": (("C1'", "N9", 1.470), ("N9", "C8", 1.370), ("N9", "C4", 1.370), ("C8", "N7", 1.310), ("N7", "C5", 1.390), ("C5", "C6", 1.400), ("C6", "N1", 1.340), ("N1", "C2", 1.340), ("C2", "N3", 1.330), ("N3", "C4", 1.350), ("C4", "C5", 1.380), ("C6", "N6", 1.340)),
    "G": (("C1'", "N9", 1.470), ("N9", "C8", 1.370), ("N9", "C4", 1.370), ("C8", "N7", 1.310), ("N7", "C5", 1.390), ("C5", "C6", 1.400), ("C6", "N1", 1.390), ("N1", "C2", 1.370), ("C2", "N3", 1.330), ("N3", "C4", 1.350), ("C4", "C5", 1.380), ("C6", "O6", 1.230), ("C2", "N2", 1.340)),
    "C": (("C1'", "N1", 1.470), ("N1", "C2", 1.380), ("C2", "N3", 1.340), ("N3", "C4", 1.350), ("C4", "C5", 1.430), ("C5", "C6", 1.340), ("C6", "N1", 1.370), ("C2", "O2", 1.230), ("C4", "N4", 1.340)),
    "U": (("C1'", "N1", 1.470), ("N1", "C2", 1.380), ("C2", "N3", 1.370), ("N3", "C4", 1.380), ("C4", "C5", 1.450), ("C5", "C6", 1.340), ("C6", "N1", 1.370), ("C2", "O2", 1.220), ("C4", "O4", 1.230)),
}


def build_graph(sequence: str, atom_lookup: dict[tuple[int, str], int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bonds: dict[tuple[int, int], tuple[float, int]] = {}
    for residue, symbol in enumerate(sequence):
        for left, right, length in COMMON_BONDS + BASE_BONDS.get(symbol, ()):
            if (residue, left) in atom_lookup and (residue, right) in atom_lookup:
                pair = tuple(sorted((atom_lookup[(residue, left)], atom_lookup[(residue, right)])))
                bonds[pair] = (length, 0)
        if residue + 1 < len(sequence):
            left = atom_lookup.get((residue, "O3'"))
            right = atom_lookup.get((residue + 1, "P"))
            if left is not None and right is not None:
                bonds[tuple(sorted((left, right)))] = (1.607, 1)
    static_edges: list[tuple[int, int, int]] = []
    for (left, right), (_, edge_type) in sorted(bonds.items()):
        static_edges.extend(((left, right, edge_type), (right, left, edge_type)))
    for residue in range(len(sequence) - 1):
        left = atom_lookup.get((residue, "C4'"))
        right = atom_lookup.get((residue + 1, "C4'"))
        if left is not None and right is not None:
            static_edges.extend(((left, right, 2), (right, left, 2)))
    edge_index = torch.tensor([[e[0] for e in static_edges], [e[1] for e in static_edges]], dtype=torch.long) if static_edges else torch.empty((2, 0), dtype=torch.long)
    edge_types = torch.tensor([e[2] for e in static_edges], dtype=torch.long)
    edge_attr = F.one_hot(edge_types, num_classes=7).float()
    bond_pairs = sorted(bonds)
    geometry_bond_index = torch.tensor(bond_pairs, dtype=torch.long).T.contiguous() if bond_pairs else torch.empty((2, 0), dtype=torch.long)
    ideal_bond_length = torch.tensor([bonds[pair][0] for pair in bond_pairs], dtype=torch.float32)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in bond_pairs:
        adjacency[left].add(right); adjacency[right].add(left)
    exclusions = set(bond_pairs)
    for center, neighbors in adjacency.items():
        ordered = sorted(neighbors)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                exclusions.add(tuple(sorted((left, right))))
    clash_exclusion_index = torch.tensor(sorted(exclusions), dtype=torch.long).T.contiguous() if exclusions else torch.empty((2, 0), dtype=torch.long)
    return edge_index, edge_attr, geometry_bond_index, ideal_bond_length, clash_exclusion_index


def _atomic_number(element: str, atom_name: str) -> int:
    symbol = element or re.sub(r"[^A-Za-z]", "", atom_name)[:1]
    number = gemmi.Element(symbol.title()).atomic_number
    if number <= 0:
        raise ValueError(f"cannot infer element for atom {atom_name!r}")
    return number


def build_sample(pred_cif: Path, confidence_json: Path, native_cif: Path, rnafm_path: Path, pdb_id: str, seed: int, sample_number: int) -> dict:
    block, all_atoms = read_atoms(pred_cif)
    with confidence_json.open("r", encoding="utf-8") as handle:
        confidence = json.load(handle)
    atom_to_token_all = _tensor(confidence, "atom_to_token_idx", torch.long).view(-1)
    atom_plddt_all = _tensor(confidence, "atom_plddt", torch.float32).view(-1)
    cif_atom_count = len(list(block.find_values("_atom_site.Cartn_x")))
    if len(atom_to_token_all) != cif_atom_count:
        raise ValueError(
            "predicted CIF atom rows and full-data JSON atom_to_token_idx have "
            f"different lengths: {cif_atom_count} versus {len(atom_to_token_all)}"
        )
    if len(atom_plddt_all) != cif_atom_count:
        raise ValueError(
            "predicted CIF atom rows and full-data JSON atom_plddt have "
            f"different lengths: {cif_atom_count} versus {len(atom_plddt_all)}"
        )
    if bool((atom_to_token_all < 0).any()):
        raise ValueError("atom_to_token_idx contains a negative token index")
    rnafm = load_rnafm(rnafm_path)
    pred_chain, original_tokens, sequence, embedding, original_chain = choose_predicted_and_embedding_chain(block, all_atoms, atom_to_token_all, rnafm)
    if embedding.shape != (len(sequence), 640):
        raise ValueError(f"RNA-FM embedding shape is {tuple(embedding.shape)}, expected {(len(sequence), 640)}")
    token_to_local = {token: i for i, token in enumerate(original_tokens)}
    selected_atoms = [atom for atom in all_atoms if atom.label_chain == pred_chain and int(atom_to_token_all[atom.row]) in token_to_local]
    if not selected_atoms:
        raise ValueError("no atoms selected from predicted RNA chain")
    atom_lookup: dict[tuple[int, str], int] = {}
    residue_index_list: list[int] = []
    for index, atom in enumerate(selected_atoms):
        local_residue = token_to_local[int(atom_to_token_all[atom.row])]
        key = (local_residue, atom.atom_name)
        if key in atom_lookup:
            raise ValueError(f"duplicate predicted atom identity {key}")
        atom_lookup[key] = index
        residue_index_list.append(local_residue)

    native_chain, residue_mapping, identity, coverage = choose_native_chain(native_chains(native_cif), sequence, original_chain)
    pred_pos = torch.tensor([atom.xyz for atom in selected_atoms], dtype=torch.float32)
    native_atom_rows = torch.full((len(selected_atoms),), -1, dtype=torch.long)
    native_pairs, pred_pairs, matched_indices = [], [], []
    for pred_residue, native_residue in residue_mapping.items():
        native_atoms = native_chain.atoms_by_residue.get(native_residue, {})
        for atom_name, native_atom in native_atoms.items():
            pred_index = atom_lookup.get((pred_residue, atom_name))
            if pred_index is not None:
                if selected_atoms[pred_index].element != native_atom.element:
                    raise ValueError(f"atom element mismatch at residue={pred_residue} atom={atom_name}")
                native_pairs.append(native_atom.xyz)
                pred_pairs.append(pred_pos[pred_index])
                matched_indices.append(pred_index)
                native_atom_rows[pred_index] = native_atom.row
    if len(matched_indices) < 3:
        raise ValueError(f"only {len(matched_indices)} native atoms could be mapped")
    aligned_native, rotation, translation, aligned_rmsd = kabsch_align(torch.tensor(native_pairs), torch.stack(pred_pairs))
    target = pred_pos.clone()
    target_mask = torch.zeros(len(selected_atoms), dtype=torch.bool)
    for row, pred_index in enumerate(matched_indices):
        target[pred_index] = aligned_native[row]
        target_mask[pred_index] = True

    selected_rows = torch.tensor([atom.row for atom in selected_atoms], dtype=torch.long)
    atom_plddt = atom_plddt_all[selected_rows]
    if not bool(torch.isfinite(atom_plddt).all()):
        raise ValueError("atom_plddt contains NaN or Inf")
    if atom_plddt.max() > 1.0 and atom_plddt.max() <= 100.0:
        atom_plddt = atom_plddt / 100.0
    if atom_plddt.min() < 0 or atom_plddt.max() > 1:
        raise ValueError("atom_plddt is outside [0, 1]")
    token_index = torch.tensor(original_tokens, dtype=torch.long)
    pair_features = {}
    for source_key, target_key in (("token_pair_pae", "token_pair_pae"), ("token_pair_pde", "token_pair_pde"), ("contact_probs", "contact_probs")):
        matrix = _tensor(confidence, source_key, torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{source_key} must be a square matrix")
        if token_index.numel() and int(token_index.max()) >= matrix.shape[0]:
            raise ValueError(
                f"{source_key} has size {matrix.shape[0]}, but selected token index "
                f"{int(token_index.max())} is out of range"
            )
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError(f"{source_key} contains NaN or Inf")
        if source_key == "contact_probs" and (float(matrix.min()) < 0.0 or float(matrix.max()) > 1.0):
            raise ValueError("contact_probs is outside [0, 1]")
        pair_features[target_key] = matrix[token_index][:, token_index].half().contiguous()

    residue_index = torch.tensor(residue_index_list, dtype=torch.long)
    edge_index, edge_attr, geometry_bond_index, ideal_bond_length, clash_exclusion_index = build_graph(sequence, atom_lookup)
    residue_type_id = torch.tensor([RNA_RESIDUE_TO_ID.get(symbol, UNKNOWN_RESIDUE_TYPE_ID) for symbol in sequence], dtype=torch.long)
    atom_name_id = torch.tensor([ATOM_NAME_TO_ID.get(atom.atom_name, UNKNOWN_ATOM_NAME_ID) for atom in selected_atoms], dtype=torch.long)
    observed_residue_mask = torch.zeros(len(sequence), dtype=torch.bool)
    observed_residue_mask[residue_index[target_mask].unique()] = True
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{pdb_id.lower()}_seed_{seed}_sample_{sample_number}",
        "pos": target,
        "pos_pred": pred_pos,
        "target_mask": target_mask,
        "atomic_numbers": torch.tensor([_atomic_number(atom.element, atom.atom_name) for atom in selected_atoms], dtype=torch.long),
        "atom_name_id": atom_name_id,
        "residue_index": residue_index,
        "sequence": sequence,
        "residue_type_id": residue_type_id,
        "rnafm_embedding": embedding.half().contiguous(),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "geometry_bond_index": geometry_bond_index,
        "ideal_bond_length": ideal_bond_length,
        "clash_exclusion_index": clash_exclusion_index,
        "atom_plddt": atom_plddt,
        "atom_to_token_idx": residue_index.clone(),
        **pair_features,
        "native_structure_id": pdb_id.upper(),
        "native_chain_id": native_chain.auth_chain,
        "predicted_chain_id": pred_chain,
        "protenix_seed": seed,
        "protenix_sample": sample_number,
        "protenix_model_name": "protenix_base_default_v1.0.0",
        "protenix_original_token_index": token_index,
        "atom_mapping_version": 3,
        "mapping_policy": MAPPING_POLICY,
        "native_sequence_source": native_chain.sequence_source,
        "native_label_chain_id": native_chain.label_chain,
        "native_label_seq_ids": list(native_chain.label_seq_ids),
        "prediction_to_native_residue_index": torch.tensor([residue_mapping[i] for i in range(len(sequence))]),
        "predicted_atom_site_row": selected_rows,
        "native_atom_site_row": native_atom_rows,
        "edge_type_version": 1,
        "geometry_template_version": 2,
        "rnafm_model_name": str(rnafm.get("model_name", "RNA-FM t12")),
        "native_sequence": native_chain.sequence,
        "native_sequence_identity": identity,
        "native_sequence_coverage": coverage,
        "observed_atom_count": int(target_mask.sum()),
        "observed_atom_fraction": float(target_mask.float().mean()),
        "observed_residue_mask": observed_residue_mask,
        "native_to_prediction_rotation": rotation,
        "native_to_prediction_translation": translation,
        "pre_refinement_aligned_rmsd": aligned_rmsd,
        "generator_version": GENERATOR_VERSION,
        "source_predicted_cif": str(pred_cif),
        "source_confidence_json": str(confidence_json),
        "source_native_cif": str(native_cif),
        "source_rnafm_pt": str(rnafm_path),
    }
    return payload


def find_predictions(pdb_dir: Path) -> Iterable[tuple[Path, Path, int, int]]:
    for cif_path in sorted(pdb_dir.rglob("*_sample_*.cif")):
        if cif_path.name.lower().endswith("_wounresol.cif") or cif_path.parent.name != "predictions":
            continue
        match = SAMPLE_CIF_RE.match(cif_path.name)
        if not match:
            continue
        seed = next((int(m.group("seed")) for parent in cif_path.parents if (m := SEED_RE.match(parent.name))), None)
        if seed is None:
            raise ValueError(f"cannot find seed_<N> ancestor for {cif_path}")
        sample_number = int(match.group("sample"))
        json_path = cif_path.with_name(f"{match.group('prefix')}_full_data_sample_{sample_number}.json")
        yield cif_path, json_path, seed, sample_number


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def serialized_size(payload: dict) -> int:
    """Run torch serialization without creating a .pt file and return its size."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_resumable_sample(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("generator_version") != GENERATOR_VERSION
            or payload.get("mapping_policy") != MAPPING_POLICY):
        raise ValueError(f"existing PT was not built with current strict mapping: {path}; use a new output directory or explicitly rebuild with --overwrite")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=home / "Data_V1")
    parser.add_argument("--native-root", type=Path, default=home / "pdb_data")
    parser.add_argument("--rnafm-root", type=Path, default=home / "Data_FM/RNA_FM_embeddings")
    parser.add_argument("--output-root", type=Path, default=home / "Data_PT_V1")
    parser.add_argument("--split", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--pdb-id", nargs="*", help="Optional case-insensitive PDB ID subset")
    parser.add_argument(
        "--min-observed-atom-fraction",
        type=float,
        default=0.5,
        help="Reject samples with less native atom supervision (default: 0.5)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate every sample and serialize it in memory, but do not "
            "create or modify any .pt file. Write dry-run reports under output-root."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.min_observed_atom_fraction <= 1.0:
        raise SystemExit("--min-observed-atom-fraction must be in [0, 1]")
    run_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for output_split in ("train", "val", "test"):
        (args.output_root / output_split).mkdir(parents=True, exist_ok=True)
    selected_ids = {value.lower() for value in args.pdb_id} if args.pdb_id else None
    manifest: list[dict] = []
    issues: list[dict] = []
    discovered_samples = 0
    total_estimated_pt_bytes = 0
    for split in args.split:
        split_dir = args.prediction_root / split
        if not split_dir.is_dir():
            issues.append({"split": split, "pdb_id": "", "sample": "", "error": f"missing split directory: {split_dir}"})
            continue
        for pdb_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            pdb_id = pdb_dir.name.lower()
            if selected_ids is not None and pdb_id not in selected_ids:
                continue
            native_cif = args.native_root / f"{pdb_id}.cif"
            rnafm_path = args.rnafm_root / pdb_id.upper() / "rnafm_t12_residue_embeddings.pt"
            try:
                predictions = list(find_predictions(pdb_dir))
                if not predictions:
                    raise FileNotFoundError(f"no primary sample CIF under {pdb_dir}")
                discovered_samples += len(predictions)
                destinations: dict[tuple[int, int], list[Path]] = defaultdict(list)
                for pred_cif, _, seed, sample_number in predictions:
                    destinations[(seed, sample_number)].append(pred_cif)
                collisions = {
                    key: paths for key, paths in destinations.items() if len(paths) > 1
                }
                if collisions:
                    details = "; ".join(
                        f"seed_{seed}/sample_{sample}: {[str(path) for path in paths]}"
                        for (seed, sample), paths in sorted(collisions.items())
                    )
                    raise ValueError(
                        "multiple prediction CIFs map to the same output path: " + details
                    )
                if not native_cif.is_file():
                    raise FileNotFoundError(native_cif)
                if not rnafm_path.is_file():
                    raise FileNotFoundError(rnafm_path)
                for pred_cif, confidence_json, seed, sample_number in predictions:
                    sample_started = time.perf_counter()
                    try:
                        output_path = args.output_root / split / pdb_id / f"seed_{seed}" / f"sample_{sample_number}.pt"
                        estimated_pt_bytes = ""
                        if args.dry_run:
                            payload = build_sample(pred_cif, confidence_json, native_cif, rnafm_path, pdb_id, seed, sample_number)
                            if payload["observed_atom_fraction"] < args.min_observed_atom_fraction:
                                raise ValueError(
                                    "observed native atom fraction "
                                    f"{payload['observed_atom_fraction']:.3f} is below "
                                    f"{args.min_observed_atom_fraction:.3f}"
                                )
                            estimated_pt_bytes = serialized_size(payload)
                            total_estimated_pt_bytes += estimated_pt_bytes
                            status = "DRY_RUN_OK"
                        elif output_path.exists() and not args.overwrite:
                            payload = load_resumable_sample(output_path)
                            status = "SKIPPED"
                        else:
                            payload = build_sample(pred_cif, confidence_json, native_cif, rnafm_path, pdb_id, seed, sample_number)
                            if payload["observed_atom_fraction"] < args.min_observed_atom_fraction:
                                raise ValueError(
                                    "observed native atom fraction "
                                    f"{payload['observed_atom_fraction']:.3f} is below "
                                    f"{args.min_observed_atom_fraction:.3f}"
                                )
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = output_path.with_suffix(".pt.tmp")
                            torch.save(payload, temporary)
                            temporary.replace(output_path)
                            status = "CREATED"
                        duration_seconds = time.perf_counter() - sample_started
                        manifest.append({
                            "split": split, "pdb_id": pdb_id.upper(), "seed": seed,
                            "sample": sample_number, "status": status,
                            "atom_count": len(payload["pos"]) if payload else "",
                            "observed_atom_count": int(payload["target_mask"].sum()) if payload else "",
                            "observed_atom_fraction": f"{float(payload['target_mask'].float().mean()):.6f}" if payload else "",
                            "pre_refinement_aligned_rmsd": f"{float(payload['pre_refinement_aligned_rmsd']):.6f}" if payload else "",
                            "duration_seconds": f"{duration_seconds:.6f}",
                            "estimated_pt_bytes": estimated_pt_bytes,
                            "output": str(output_path),
                        })
                    except Exception as exc:
                        issues.append({
                            "split": split, "pdb_id": pdb_id.upper(),
                            "sample": f"seed_{seed}/sample_{sample_number}",
                            "duration_seconds": f"{time.perf_counter() - sample_started:.6f}",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        if args.fail_fast:
                            raise
            except Exception as exc:  # continue bulk generation but make every failure auditable
                issues.append({"split": split, "pdb_id": pdb_id.upper(), "sample": "", "duration_seconds": "", "error": f"{type(exc).__name__}: {exc}"})
                if args.fail_fast:
                    raise
    report_prefix = "dry_run_" if args.dry_run else "generation_"
    manifest_path = args.output_root / f"{report_prefix}manifest.tsv"
    issues_path = args.output_root / f"{report_prefix}issues.tsv"
    summary_path = args.output_root / f"{report_prefix}summary.json"
    write_tsv(
        manifest_path,
        manifest,
        ["split", "pdb_id", "seed", "sample", "status", "atom_count",
         "observed_atom_count", "observed_atom_fraction",
         "pre_refinement_aligned_rmsd", "duration_seconds", "estimated_pt_bytes", "output"],
    )
    write_tsv(
        issues_path,
        issues,
        ["split", "pdb_id", "sample", "duration_seconds", "error"],
    )
    elapsed_seconds = time.perf_counter() - run_started
    problem_pdb_ids = sorted({row["pdb_id"] for row in issues if row["pdb_id"]})
    failed_samples = max(0, discovered_samples - len(manifest))
    successful_sample_seconds = sum(float(row["duration_seconds"]) for row in manifest)
    failed_attempt_seconds = sum(
        float(row["duration_seconds"])
        for row in issues
        if row.get("duration_seconds")
    )
    non_sample_seconds = max(
        0.0, elapsed_seconds - successful_sample_seconds - failed_attempt_seconds
    )
    average_successful_sample_seconds = (
        successful_sample_seconds / len(manifest) if manifest else None
    )
    estimated_full_generation_seconds = None
    estimated_total_pt_bytes = None
    estimate_basis = None
    if args.dry_run and manifest:
        # A failed sample often exits much earlier than a valid one. Replace its
        # failed-attempt duration with the mean validated-sample duration so the
        # full-generation estimate is not artificially optimistic.
        estimated_full_generation_seconds = (
            non_sample_seconds
            + successful_sample_seconds
            + failed_samples * average_successful_sample_seconds
        )
        estimated_total_pt_bytes = round(
            total_estimated_pt_bytes / len(manifest) * discovered_samples
        )
        estimate_basis = (
            "full construction and in-memory torch serialization for successful "
            "samples; failed samples use the successful-sample mean; physical disk "
            "write latency is excluded"
        )
    elif args.dry_run:
        estimate_basis = (
            "unavailable because no sample completed successfully; see elapsed_seconds "
            "for the validation-run duration"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "mapping_policy": MAPPING_POLICY,
        "mode": "dry_run" if args.dry_run else "write",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "elapsed_duration": format_duration(elapsed_seconds),
        "discovered_samples": discovered_samples,
        "successful_samples": len(manifest),
        "failed_samples": failed_samples,
        "created_samples": sum(row["status"] == "CREATED" for row in manifest),
        "skipped_samples": sum(row["status"] == "SKIPPED" for row in manifest),
        "dry_run_samples": sum(row["status"] == "DRY_RUN_OK" for row in manifest),
        "issues": len(issues),
        "problem_pdb_count": len(problem_pdb_ids),
        "problem_pdb_ids": problem_pdb_ids,
        "validated_pt_bytes": total_estimated_pt_bytes if args.dry_run else None,
        "estimated_total_pt_bytes": estimated_total_pt_bytes,
        "average_successful_sample_seconds": (
            round(average_successful_sample_seconds, 6)
            if args.dry_run and average_successful_sample_seconds is not None else None
        ),
        "estimated_full_generation_seconds": (
            round(estimated_full_generation_seconds, 6)
            if estimated_full_generation_seconds is not None else None
        ),
        "estimated_full_generation_duration": (
            format_duration(estimated_full_generation_seconds)
            if estimated_full_generation_seconds is not None else None
        ),
        "estimate_basis": estimate_basis,
        "output_root": str(args.output_root),
        "manifest_path": str(manifest_path),
        "issues_path": str(issues_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
