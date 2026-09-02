#!/usr/bin/env python3
"""Resolve repeated-sequence RNA chain mappings with complex-level coordinates.

This is the second stage after ``build_chain_maps.py``.  It preserves the
sequence-audit ``<PDB>/chain_map.tsv`` and writes coordinate-resolved variants
under::

    <PDB>/chain_map_variants/mapping_01/chain_map.tsv
    <PDB>/chain_map_variants/mapping_02/chain_map.tsv
    <PDB>/chain_map.txt

Each available prediction candidate is assigned to exactly one mapping variant.
Equivalent coordinate solutions are broken by cross-candidate majority vote,
then by a deterministic lexical rule.  A single global rigid transform is used
for the whole RNA complex; chain pairs are never independently superposed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise SystemExit(
        "Missing scipy. Install dependencies with: "
        "conda install -c conda-forge numpy scipy gemmi openpyxl"
    ) from exc

from build_chain_maps import (
    DEFAULT_SEEDS,
    OUTPUT_COLUMNS,
    STATUS_PASS,
    STATUS_REVIEW,
    CandidatePath,
    ChainRecord,
    clean_cif_value,
    extract_coordinate_rna_chains,
    global_sequence_identity,
    index_files_by_stem,
    index_prediction_seed_dirs,
    inventory_candidates,
    normalize_sequence,
    parse_seed_list,
    read_single_block,
    read_targets_xlsx,
    sequence_length,
    sequence_tokens,
)


REPRESENTATIVE_ATOM = "C4'"


@dataclass(frozen=True)
class CoordinateChain:
    record: ChainRecord
    coordinates: dict[int, np.ndarray]

    @property
    def theoretical_length(self) -> int:
        return sequence_length(self.record.sequence)

    @property
    def coverage(self) -> float:
        length = self.theoretical_length
        return len(self.coordinates) / length if length else 0.0


MappingKey = tuple[tuple[str, str], ...]


@dataclass
class CandidateMapping:
    candidate: CandidatePath
    pred_chains: list[CoordinateChain]
    option_scores: dict[MappingKey, float]
    chosen_mapping: MappingKey | None = None
    chosen_rmsd: float | None = None
    consensus_tie_break: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class ResolvedTarget:
    pdb_id: str
    gt_chains: list[CoordinateChain]
    inventory_count: int
    candidates: list[CandidateMapping]
    inventory_issues: list[str]
    fatal_issues: list[str]
    review_reasons: list[str]
    variants: dict[MappingKey, list[CandidateMapping]]
    status: str


def read_base_chain_map(path: Path) -> tuple[dict[str, str], list[str], bool]:
    """Read the stage-1 chain map as the authoritative JSON-input sequence map."""
    if not path.is_file():
        return {}, [f"base chain_map.tsv is missing: {path}"], True
    issues: list[str] = []
    input_by_pred: dict[str, str] = {}
    has_review = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"pred_chain_id", "input_sequence", "status"} - set(
            reader.fieldnames or []
        )
        if missing:
            return {}, [f"base chain_map.tsv missing columns={sorted(missing)}"], True
        for row_number, row in enumerate(reader, start=2):
            pred_id = str(row.get("pred_chain_id", "")).strip()
            sequence = normalize_sequence(row.get("input_sequence", ""))
            if not pred_id or not sequence:
                issues.append(
                    f"base chain_map.tsv row {row_number} has empty pred_chain_id/input_sequence"
                )
                continue
            if pred_id in input_by_pred and input_by_pred[pred_id] != sequence:
                issues.append(
                    f"base chain_map.tsv has conflicting input sequences for {pred_id}"
                )
                continue
            input_by_pred[pred_id] = sequence
            has_review = has_review or row.get("status", "").strip() != STATUS_PASS
    if not input_by_pred:
        issues.append("base chain_map.tsv has no usable chain rows")
    return input_by_pred, issues, has_review


def apply_input_sequences(
    chains: list[CoordinateChain], input_by_pred: dict[str, str]
) -> tuple[list[CoordinateChain], list[str]]:
    """Replace prediction-derived sequences with stage-1 JSON input sequences."""
    result: list[CoordinateChain] = []
    issues: list[str] = []
    seen: set[str] = set()
    for chain in chains:
        pred_id = chain.record.label_asym_id
        seen.add(pred_id)
        input_sequence = input_by_pred.get(pred_id)
        if not input_sequence:
            issues.append(f"predicted chain {pred_id} is absent from base chain_map.tsv")
            input_sequence = chain.record.sequence
        elif input_sequence != chain.record.sequence:
            if sequence_length(input_sequence) != sequence_length(chain.record.sequence):
                issues.append(
                    f"predicted CIF theoretical-sequence length differs from JSON input "
                    f"for chain {pred_id}: CIF={sequence_length(chain.record.sequence)}, "
                    f"JSON={sequence_length(input_sequence)}"
                )
            else:
                input_tokens = sequence_tokens(input_sequence)
                cif_tokens = sequence_tokens(chain.record.sequence)
                concrete_conflict = any(
                    input_token != cif_token and cif_token in {"A", "C", "G", "U"}
                    for input_token, cif_token in zip(input_tokens, cif_tokens)
                )
                if concrete_conflict:
                    issues.append(
                        f"predicted CIF theoretical sequence conflicts with JSON input "
                        f"for chain {pred_id}"
                    )
                else:
                    issues.append(
                        f"predicted CIF theoretical sequence differs from JSON input for "
                        f"chain {pred_id} (only unknown/modified CIF residue tokens differ)"
                    )
        result.append(
            CoordinateChain(
                record=ChainRecord(
                    label_asym_id=chain.record.label_asym_id,
                    auth_asym_id=chain.record.auth_asym_id,
                    sequence=input_sequence,
                ),
                coordinates=chain.coordinates,
            )
        )
    missing_predictions = sorted(set(input_by_pred) - seen)
    if missing_predictions:
        issues.append(
            f"base chain_map.tsv chains absent from predicted CIF={missing_predictions}"
        )
    return result, issues


def is_blocking_prediction_input_issue(issue: str) -> bool:
    """Only chain-ID/set disagreement blocks mapping.

    The prediction-CIF theoretical sequence is a redundant integrity signal.
    JSON input sequence remains authoritative and is what ``exact_match`` is
    compared with, so a CIF sequence-decoding difference is logged but does
    not by itself force manual review.
    """
    return (
        "is absent from base chain_map.tsv" in issue
        or "base chain_map.tsv chains absent from predicted CIF" in issue
        or "theoretical-sequence length differs from JSON input" in issue
        or "theoretical sequence conflicts with JSON input" in issue
    )


def _category_column(
    category: dict[str, list[Any]], name: str, row_count: int
) -> list[Any]:
    values = category.get(name, [])
    if not values:
        return [""] * row_count
    if len(values) != row_count:
        raise ValueError(
            f"mmCIF category column length mismatch: {name}={len(values)}, "
            f"expected={row_count}"
        )
    return values


def _normalize_atom_name(value: Any) -> str:
    return clean_cif_value(value).upper().replace("*", "'")


def _first_model(values: Iterable[Any]) -> str:
    cleaned = [clean_cif_value(value) for value in values]
    nonempty = [value for value in cleaned if value]
    if not nonempty:
        return ""
    try:
        return min(nonempty, key=lambda item: (float(item), item))
    except ValueError:
        return nonempty[0]


def extract_coordinate_chains(
    path: Path,
    *,
    atom_name: str = REPRESENTATIVE_ATOM,
) -> tuple[list[CoordinateChain], list[str]]:
    """Extract one representative coordinate per theoretical RNA residue.

    ``label_seq_id`` is deliberately used as the residue key.  For alternate
    locations, the highest-occupancy atom is selected with a deterministic
    alt-ID tie break.  Only the first coordinate model is used.
    """
    records, issues = extract_coordinate_rna_chains(path)
    block = read_single_block(path)
    atom_site = block.get_mmcif_category("_atom_site.")
    labels = atom_site.get("label_asym_id", [])
    row_count = len(labels)
    if not row_count:
        raise ValueError(f"Missing _atom_site rows: {path}")

    atom_ids = _category_column(atom_site, "label_atom_id", row_count)
    seq_ids = _category_column(atom_site, "label_seq_id", row_count)
    xs = _category_column(atom_site, "Cartn_x", row_count)
    ys = _category_column(atom_site, "Cartn_y", row_count)
    zs = _category_column(atom_site, "Cartn_z", row_count)
    occupancies = _category_column(atom_site, "occupancy", row_count)
    alt_ids = _category_column(atom_site, "label_alt_id", row_count)
    model_values = _category_column(atom_site, "pdbx_PDB_model_num", row_count)
    selected_model = _first_model(model_values)
    wanted_atom = _normalize_atom_name(atom_name)
    record_by_label = {record.label_asym_id: record for record in records}

    # (chain, label_seq_id) -> (occupancy, alt-id ordering, coordinate)
    selected: dict[tuple[str, int], tuple[float, str, np.ndarray]] = {}
    for raw_label, raw_atom, raw_seq, raw_x, raw_y, raw_z, raw_occ, raw_alt, raw_model in zip(
        labels,
        atom_ids,
        seq_ids,
        xs,
        ys,
        zs,
        occupancies,
        alt_ids,
        model_values,
    ):
        label = clean_cif_value(raw_label)
        if label not in record_by_label or _normalize_atom_name(raw_atom) != wanted_atom:
            continue
        model = clean_cif_value(raw_model)
        if selected_model and model and model != selected_model:
            continue
        try:
            seq_id = int(float(clean_cif_value(raw_seq)))
            coordinate = np.asarray(
                [float(raw_x), float(raw_y), float(raw_z)], dtype=np.float64
            )
            occupancy = float(clean_cif_value(raw_occ) or 0.0)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(coordinate).all():
            continue
        alt_id = clean_cif_value(raw_alt)
        key = (label, seq_id)
        previous = selected.get(key)
        candidate = (occupancy, alt_id, coordinate)
        if previous is None or occupancy > previous[0] or (
            occupancy == previous[0] and alt_id < previous[1]
        ):
            selected[key] = candidate

    result: list[CoordinateChain] = []
    for record in records:
        coordinates = {
            seq_id: value[2]
            for (label, seq_id), value in selected.items()
            if label == record.label_asym_id
        }
        result.append(CoordinateChain(record=record, coordinates=coordinates))
        if not coordinates:
            issues.append(
                f"RNA chain {record.label_asym_id} has no {atom_name} coordinates"
            )
    return result, issues


def _mapping_key(mapping: dict[str, str], pred_order: list[str]) -> MappingKey:
    return tuple((pred_id, mapping[pred_id]) for pred_id in pred_order)


def _mapping_dict(mapping: MappingKey) -> dict[str, str]:
    return dict(mapping)


def _common_points(
    pred: CoordinateChain, gt: CoordinateChain
) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(pred.coordinates) & set(gt.coordinates))
    if not common:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.stack([pred.coordinates[index] for index in common]),
        np.stack([gt.coordinates[index] for index in common]),
    )


def _collect_mapping_points(
    mapping: dict[str, str],
    pred_by_id: dict[str, CoordinateChain],
    gt_by_id: dict[str, CoordinateChain],
) -> tuple[np.ndarray, np.ndarray]:
    pred_points: list[np.ndarray] = []
    gt_points: list[np.ndarray] = []
    for pred_id, gt_id in mapping.items():
        pred, gt = _common_points(pred_by_id[pred_id], gt_by_id[gt_id])
        if len(pred):
            pred_points.append(pred)
            gt_points.append(gt)
    if not pred_points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.concatenate(pred_points), np.concatenate(gt_points)


def kabsch_transform(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Kabsch inputs must be matching (N, 3) arrays")
    if len(mobile) < 3:
        raise ValueError("At least three representative atoms are required for Kabsch")
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    centered_mobile = mobile - mobile_center
    centered_reference = reference - reference_center
    covariance = centered_mobile.T @ centered_reference
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def _rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    if not len(mobile):
        return math.inf
    return float(np.sqrt(np.mean(np.sum((mobile - reference) ** 2, axis=1))))


def _global_mapping_rmsd(
    mapping: dict[str, str],
    pred_by_id: dict[str, CoordinateChain],
    gt_by_id: dict[str, CoordinateChain],
) -> float:
    pred_points, gt_points = _collect_mapping_points(mapping, pred_by_id, gt_by_id)
    if len(pred_points) < 3:
        return math.inf
    rotation, translation = kabsch_transform(pred_points, gt_points)
    return _rmsd(pred_points @ rotation + translation, gt_points)


def _sequence_groups(
    pred_chains: list[CoordinateChain], gt_chains: list[CoordinateChain]
) -> tuple[dict[str, str], list[tuple[list[str], list[str]]], bool, list[str]]:
    pred_by_sequence: dict[str, list[str]] = defaultdict(list)
    gt_by_sequence: dict[str, list[str]] = defaultdict(list)
    for chain in pred_chains:
        pred_by_sequence[chain.record.sequence].append(chain.record.label_asym_id)
    for chain in gt_chains:
        gt_by_sequence[chain.record.sequence].append(chain.record.label_asym_id)
    for ids in pred_by_sequence.values():
        ids.sort()
    for ids in gt_by_sequence.values():
        ids.sort()

    exact_multiset = Counter(
        chain.record.sequence for chain in pred_chains
    ) == Counter(chain.record.sequence for chain in gt_chains)
    issues: list[str] = []
    if not exact_multiset:
        issues.append("prediction and GT theoretical-sequence multisets differ")
        # A deterministic sequence-identity assignment is still written for QC,
        # but it is never eligible for PASS.
        pred_ids = [chain.record.label_asym_id for chain in pred_chains]
        gt_ids = [chain.record.label_asym_id for chain in gt_chains]
        if len(pred_ids) != len(gt_ids):
            return {}, [], False, issues
        pred_lookup = {chain.record.label_asym_id: chain for chain in pred_chains}
        gt_lookup = {chain.record.label_asym_id: chain for chain in gt_chains}
        costs = np.asarray(
            [
                [
                    1.0
                    - global_sequence_identity(
                        pred_lookup[pred_id].record.sequence,
                        gt_lookup[gt_id].record.sequence,
                    )
                    for gt_id in gt_ids
                ]
                for pred_id in pred_ids
            ]
        )
        rows, cols = linear_sum_assignment(costs)
        return (
            {pred_ids[row]: gt_ids[col] for row, col in zip(rows, cols)},
            [],
            False,
            issues,
        )

    fixed: dict[str, str] = {}
    ambiguous: list[tuple[list[str], list[str]]] = []
    for sequence in sorted(pred_by_sequence):
        pred_ids = pred_by_sequence[sequence]
        gt_ids = gt_by_sequence[sequence]
        if len(pred_ids) == 1:
            fixed[pred_ids[0]] = gt_ids[0]
        else:
            ambiguous.append((pred_ids, gt_ids))
    return fixed, ambiguous, True, issues


def _assign_after_global_transform(
    fixed: dict[str, str],
    groups: list[tuple[list[str], list[str]]],
    pred_by_id: dict[str, CoordinateChain],
    gt_by_id: dict[str, CoordinateChain],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, str]:
    mapping = dict(fixed)
    for pred_ids, gt_ids in groups:
        costs = np.full((len(pred_ids), len(gt_ids)), 1.0e12, dtype=np.float64)
        for row, pred_id in enumerate(pred_ids):
            for col, gt_id in enumerate(gt_ids):
                pred_points, gt_points = _common_points(
                    pred_by_id[pred_id], gt_by_id[gt_id]
                )
                if len(pred_points):
                    transformed = pred_points @ rotation + translation
                    costs[row, col] = float(
                        np.mean(np.sum((transformed - gt_points) ** 2, axis=1))
                    )
        rows, cols = linear_sum_assignment(costs)
        if any(costs[row, col] >= 1.0e11 for row, col in zip(rows, cols)):
            raise ValueError("no representative-coordinate overlap for a chain assignment")
        mapping.update(
            {pred_ids[row]: gt_ids[col] for row, col in zip(rows, cols)}
        )
    return mapping


def solve_candidate_mapping(
    pred_chains: list[CoordinateChain],
    gt_chains: list[CoordinateChain],
    *,
    max_iterations: int,
    tie_tolerance: float,
) -> tuple[dict[MappingKey, float], bool, list[str]]:
    """Return all near-equal best complex-level mappings for one candidate."""
    issues: list[str] = []
    if len(pred_chains) != len(gt_chains):
        return {}, False, [
            f"prediction RNA chain count={len(pred_chains)} but GT count={len(gt_chains)}"
        ]
    pred_by_id = {chain.record.label_asym_id: chain for chain in pred_chains}
    gt_by_id = {chain.record.label_asym_id: chain for chain in gt_chains}
    pred_order = sorted(pred_by_id)
    fixed, ambiguous, exact_multiset, sequence_issues = _sequence_groups(
        pred_chains, gt_chains
    )
    issues.extend(sequence_issues)
    if not exact_multiset:
        if len(fixed) != len(pred_chains):
            return {}, False, issues
        score = _global_mapping_rmsd(fixed, pred_by_id, gt_by_id)
        if not math.isfinite(score):
            issues.append("fewer than three common representative atoms")
            return {}, False, issues
        return {_mapping_key(fixed, pred_order): score}, False, issues

    if not ambiguous:
        score = _global_mapping_rmsd(fixed, pred_by_id, gt_by_id)
        if not math.isfinite(score):
            issues.append("fewer than three common representative atoms")
            return {}, True, issues
        return {_mapping_key(fixed, pred_order): score}, True, issues

    initial_mappings: list[dict[str, str]] = []
    fixed_pred, fixed_gt = _collect_mapping_points(fixed, pred_by_id, gt_by_id)
    if len(fixed_pred) >= 3:
        initial_mappings.append(dict(fixed))
    else:
        # Longest sequence is the most geometrically informative anchor.  One
        # fixed predicted chain is tried against every symmetry-equivalent GT
        # chain, so no GT chain label is privileged.
        anchor_pred_ids, anchor_gt_ids = max(
            ambiguous,
            key=lambda group: (
                max(
                    len(pred_by_id[pred_id].coordinates)
                    for pred_id in group[0]
                ),
                -len(group[0]),
            ),
        )
        anchor_pred = anchor_pred_ids[0]
        for anchor_gt in anchor_gt_ids:
            initial = dict(fixed)
            initial[anchor_pred] = anchor_gt
            pred_points, _ = _collect_mapping_points(initial, pred_by_id, gt_by_id)
            if len(pred_points) >= 3:
                initial_mappings.append(initial)
    if not initial_mappings:
        issues.append("no chain supplies three common representative atoms for anchoring")
        return {}, True, issues

    scores: dict[MappingKey, float] = {}
    for initial in initial_mappings:
        mapping = initial
        for _ in range(max_iterations):
            pred_points, gt_points = _collect_mapping_points(
                mapping, pred_by_id, gt_by_id
            )
            if len(pred_points) < 3:
                break
            rotation, translation = kabsch_transform(pred_points, gt_points)
            updated = _assign_after_global_transform(
                fixed,
                ambiguous,
                pred_by_id,
                gt_by_id,
                rotation,
                translation,
            )
            if updated == mapping:
                mapping = updated
                break
            mapping = updated
        if len(mapping) != len(pred_chains):
            continue
        score = _global_mapping_rmsd(mapping, pred_by_id, gt_by_id)
        if not math.isfinite(score):
            continue
        key = _mapping_key(mapping, pred_order)
        scores[key] = min(score, scores.get(key, math.inf))

    if not scores:
        issues.append("coordinate permutation optimization produced no valid mapping")
        return {}, True, issues
    best = min(scores.values())
    near_best = {
        key: score for key, score in scores.items() if score <= best + tie_tolerance
    }
    return near_best, True, issues


def choose_consensus_mappings(candidates: list[CandidateMapping]) -> None:
    """Resolve coordinate ties using mappings supported by other candidates."""
    support: Counter[MappingKey] = Counter()
    unresolved: list[CandidateMapping] = []
    for candidate in candidates:
        if not candidate.option_scores:
            continue
        if len(candidate.option_scores) == 1:
            candidate.chosen_mapping = next(iter(candidate.option_scores))
            support[candidate.chosen_mapping] += 1
        else:
            unresolved.append(candidate)

    # Use only independent/unique evidence for the first vote.  When no such
    # evidence exists, the lexical rule seeds a reproducible consensus.
    for candidate in unresolved:
        ranked = sorted(
            candidate.option_scores,
            key=lambda key: (
                -support[key],
                candidate.option_scores[key],
                key,
            ),
        )
        candidate.chosen_mapping = ranked[0]
        candidate.consensus_tie_break = True
    for candidate in unresolved:
        assert candidate.chosen_mapping is not None
        support[candidate.chosen_mapping] += 1

    for candidate in candidates:
        if candidate.chosen_mapping is not None:
            candidate.chosen_rmsd = candidate.option_scores[candidate.chosen_mapping]


def _all_exact(
    mapping: MappingKey,
    pred_by_id: dict[str, CoordinateChain],
    gt_by_id: dict[str, CoordinateChain],
) -> bool:
    return all(
        pred_by_id[pred_id].record.sequence == gt_by_id[gt_id].record.sequence
        for pred_id, gt_id in mapping
    )


def resolve_target(
    pdb_id: str,
    *,
    gt_path: Path | None,
    base_chain_map_path: Path,
    seed_dirs: dict[int, Path],
    expected_seeds: tuple[int, ...],
    expected_samples: int,
    representative_atom: str,
    min_gt_coverage: float,
    min_dominant_fraction: float,
    max_iterations: int,
    tie_tolerance: float,
) -> ResolvedTarget:
    inventory = inventory_candidates(
        pdb_id,
        seed_dirs,
        expected_seeds=expected_seeds,
        expected_samples=expected_samples,
    )
    fatal: list[str] = []
    review: list[str] = []
    input_by_pred, base_issues, _ = read_base_chain_map(base_chain_map_path)
    if base_issues:
        review.extend(base_issues)
    gt_chains: list[CoordinateChain] = []
    if gt_path is None:
        fatal.append("GT CIF file is missing")
    else:
        try:
            gt_chains, gt_issues = extract_coordinate_chains(
                gt_path, atom_name=representative_atom
            )
            # Parser warnings are retained in the manifest.  Only coordinate
            # coverage and invalid auth IDs below directly control PASS.
            fatal.extend(f"GT: {issue}" for issue in gt_issues if "conflicting" in issue)
        except Exception as exc:
            fatal.append(f"GT CIF parse failed: {type(exc).__name__}: {exc}")

    low_coverage = [
        (
            chain.record.label_asym_id,
            len(chain.coordinates),
            chain.theoretical_length,
            chain.coverage,
        )
        for chain in gt_chains
        if chain.coverage < min_gt_coverage
    ]
    if low_coverage:
        review.append(
            "GT representative-atom coverage below threshold: "
            + ", ".join(
                f"{label}={observed}/{length} ({coverage:.3f})"
                for label, observed, length, coverage in low_coverage
            )
        )
    invalid_auth = [
        chain.record.label_asym_id
        for chain in gt_chains
        if not chain.record.auth_asym_id or "|" in chain.record.auth_asym_id
    ]
    if invalid_auth:
        review.append(f"GT auth_asym_id is missing/non-unique for chains={invalid_auth}")

    candidate_results: list[CandidateMapping] = []
    if not inventory.candidates:
        review.append("no available prediction candidate was found")
    for candidate in inventory.candidates:
        result = CandidateMapping(
            candidate=candidate,
            pred_chains=[],
            option_scores={},
        )
        try:
            pred_chains, pred_issues = extract_coordinate_chains(
                candidate.path, atom_name=representative_atom
            )
            pred_chains, input_issues = apply_input_sequences(
                pred_chains, input_by_pred
            )
            result.pred_chains = pred_chains
            result.issues.extend(pred_issues)
            result.issues.extend(input_issues)
            blocking_input_issues = [
                issue
                for issue in input_issues
                if is_blocking_prediction_input_issue(issue)
            ]
            if blocking_input_issues:
                review.append(
                    f"seed_{candidate.seed} sample_{candidate.sample}: "
                    "prediction chain-ID set disagrees with stage-1 JSON-input mapping"
                )
            options, exact_multiset, solve_issues = solve_candidate_mapping(
                pred_chains,
                gt_chains,
                max_iterations=max_iterations,
                tie_tolerance=tie_tolerance,
            )
            result.option_scores = options
            result.issues.extend(solve_issues)
            if not exact_multiset:
                review.append(
                    f"seed_{candidate.seed} sample_{candidate.sample}: "
                    "at least one mapped chain has exact_match=False"
                )
            if not options:
                review.append(
                    f"seed_{candidate.seed} sample_{candidate.sample}: mapping failed"
                )
        except Exception as exc:
            result.issues.append(
                f"candidate parse/solve failed: {type(exc).__name__}: {exc}"
            )
            review.append(
                f"seed_{candidate.seed} sample_{candidate.sample}: candidate parse failed"
            )
        candidate_results.append(result)

    choose_consensus_mappings(candidate_results)
    variants: dict[MappingKey, list[CandidateMapping]] = defaultdict(list)
    for candidate in candidate_results:
        if candidate.chosen_mapping is not None:
            variants[candidate.chosen_mapping].append(candidate)

    solved_count = sum(bool(candidate.chosen_mapping) for candidate in candidate_results)
    if solved_count != len(inventory.candidates):
        review.append(
            f"only {solved_count}/{len(inventory.candidates)} available candidates were mapped"
        )
    if variants and solved_count:
        dominant = max(len(items) for items in variants.values()) / solved_count
        if solved_count >= 4 and dominant < min_dominant_fraction:
            review.append(
                f"mapping is extremely unstable: dominant fraction={dominant:.3f} "
                f"< {min_dominant_fraction:.3f}"
            )

    # Defensive exact-match check after the coordinate consensus decision.
    gt_by_id = {chain.record.label_asym_id: chain for chain in gt_chains}
    for mapping, items in variants.items():
        pred_by_id = {
            chain.record.label_asym_id: chain for chain in items[0].pred_chains
        }
        if not _all_exact(mapping, pred_by_id, gt_by_id):
            review.append("at least one output row has exact_match=False")
            break

    status = STATUS_PASS if not fatal and not review else STATUS_REVIEW
    return ResolvedTarget(
        pdb_id=pdb_id,
        gt_chains=gt_chains,
        inventory_count=len(inventory.candidates),
        candidates=candidate_results,
        inventory_issues=inventory.issues,
        fatal_issues=list(dict.fromkeys(fatal)),
        review_reasons=list(dict.fromkeys(review)),
        variants=dict(variants),
        status=status,
    )


def rows_for_variant(
    mapping: MappingKey,
    representative: CandidateMapping,
    gt_chains: list[CoordinateChain],
    status: str,
) -> list[dict[str, str]]:
    pred_by_id = {
        chain.record.label_asym_id: chain for chain in representative.pred_chains
    }
    gt_by_id = {chain.record.label_asym_id: chain for chain in gt_chains}
    rows: list[dict[str, str]] = []
    for pred_id, gt_id in mapping:
        pred = pred_by_id[pred_id].record
        gt = gt_by_id[gt_id].record
        identity = global_sequence_identity(pred.sequence, gt.sequence)
        exact = pred.sequence == gt.sequence
        rows.append(
            {
                "pred_chain_id": pred_id,
                "input_sequence": pred.sequence,
                "gt_label_asym_id": gt.label_asym_id,
                "gt_auth_asym_id": gt.auth_asym_id,
                "gt_sequence": gt.sequence,
                "identity": f"{identity:.6f}",
                "exact_match": str(exact),
                "status": STATUS_PASS if status == STATUS_PASS and exact else STATUS_REVIEW,
            }
        )
    return rows


def _atomic_write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _candidate_name(candidate: CandidateMapping) -> str:
    return f"seed_{candidate.candidate.seed}_sample_{candidate.candidate.sample}"


def write_resolved_target(
    output_root: Path,
    result: ResolvedTarget,
    *,
    representative_atom: str,
    min_gt_coverage: float,
    min_dominant_fraction: float,
    overwrite: bool,
) -> int:
    target_dir = output_root / result.pdb_id
    variants_dir = target_dir / "chain_map_variants"
    manifest_path = target_dir / "chain_map.txt"
    if (variants_dir.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"resolved chain-map output exists for {result.pdb_id}; use --overwrite"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir / ".chain_map_variants.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()

    ordered_variants = sorted(
        result.variants.items(), key=lambda item: (-len(item[1]), item[0])
    )
    variant_names: dict[MappingKey, str] = {}
    for index, (mapping, candidates) in enumerate(ordered_variants, start=1):
        variant_name = f"mapping_{index:02d}"
        variant_names[mapping] = variant_name
        rows = rows_for_variant(
            mapping, candidates[0], result.gt_chains, result.status
        )
        _atomic_write_tsv(
            temporary_dir / variant_name / "chain_map.tsv",
            rows,
        )
    if variants_dir.exists():
        shutil.rmtree(variants_dir)
    os.replace(temporary_dir, variants_dir)

    solved = [item for item in result.candidates if item.chosen_mapping is not None]
    dominant_fraction = (
        max((len(items) for items in result.variants.values()), default=0) / len(solved)
        if solved
        else 0.0
    )
    lines = [
        f"target_id: {result.pdb_id}",
        f"status: {result.status}",
        f"representative_atom: {representative_atom}",
        f"available_candidates: {result.inventory_count}",
        f"mapped_candidates: {len(solved)}",
        f"mapping_variants: {len(ordered_variants)}",
        f"dominant_mapping_fraction: {dominant_fraction:.6f}",
        f"min_gt_coverage_threshold: {min_gt_coverage:.6f}",
        f"min_dominant_fraction_threshold: {min_dominant_fraction:.6f}",
        "",
        "GT coordinate coverage:",
    ]
    if result.gt_chains:
        for chain in result.gt_chains:
            lines.append(
                f"  {chain.record.label_asym_id} (auth={chain.record.auth_asym_id}): "
                f"{len(chain.coordinates)}/{chain.theoretical_length} "
                f"({chain.coverage:.6f})"
            )
    else:
        lines.append("  None")

    lines.extend(["", "Mapping variants (most-supported first):"])
    if ordered_variants:
        for mapping, candidates in ordered_variants:
            name = variant_names[mapping]
            lines.append(
                f"  {name}/chain_map.tsv: {len(candidates)} candidate(s)"
            )
            lines.append(
                "    mapping: "
                + ", ".join(f"{pred}->{gt}" for pred, gt in mapping)
            )
            lines.append(
                "    applies_to: "
                + ", ".join(_candidate_name(candidate) for candidate in candidates)
            )
    else:
        lines.append("  None")

    tie_candidates = [item for item in solved if item.consensus_tie_break]
    lines.extend(["", "Consensus tie-breaks:"])
    if tie_candidates:
        for candidate in tie_candidates:
            lines.append(
                f"  {_candidate_name(candidate)}: selected "
                f"{variant_names[candidate.chosen_mapping]} from "
                f"{len(candidate.option_scores)} RMSD-equivalent solutions"
            )
    else:
        lines.append("  None")

    lines.extend(["", "Review reasons:"])
    all_review = result.fatal_issues + result.review_reasons
    if all_review:
        lines.extend(f"  - {reason}" for reason in all_review)
    else:
        lines.append("  None")

    lines.extend(["", "Non-blocking inventory notes:"])
    if result.inventory_issues:
        lines.extend(f"  - {issue}" for issue in result.inventory_issues)
    else:
        lines.append("  None")

    lines.extend(["", "Per-candidate details:"])
    for candidate in result.candidates:
        chosen_variant = (
            variant_names.get(candidate.chosen_mapping, "UNMAPPED")
            if candidate.chosen_mapping is not None
            else "UNMAPPED"
        )
        rmsd = (
            f"{candidate.chosen_rmsd:.6f}"
            if candidate.chosen_rmsd is not None
            else "NA"
        )
        lines.append(
            f"  {_candidate_name(candidate)}: {chosen_variant}, "
            f"global_C4prime_RMSD={rmsd}, "
            f"equivalent_options={len(candidate.option_scores)}"
        )
        lines.extend(f"    - {issue}" for issue in candidate.issues)

    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return len(ordered_variants)


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description=(
            "Resolve repeated-sequence RNA chain permutations for every available "
            "prediction candidate using one global complex alignment."
        )
    )
    parser.add_argument(
        "--targets-xlsx",
        type=Path,
        default=home / "Json_data" / "mapping" / "xlsx" / "targets.xlsx",
    )
    parser.add_argument("--gt-cif-dir", type=Path, default=home / "pdb_data")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=home / "Json_data" / "Foldbench_predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=home / "Json_data" / "mapping" / "tsv",
    )
    parser.add_argument("--seeds", type=parse_seed_list, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=5)
    parser.add_argument("--representative-atom", default=REPRESENTATIVE_ATOM)
    parser.add_argument(
        "--min-gt-coverage",
        type=float,
        default=0.50,
        help="Minimum per-GT-chain representative-atom coverage for PASS (default 0.50).",
    )
    parser.add_argument(
        "--min-dominant-fraction",
        type=float,
        default=0.20,
        help=(
            "Targets with >=4 candidates are unstable when the most common mapping "
            "covers less than this fraction (default 0.20)."
        ),
    )
    parser.add_argument("--tie-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only-current-review",
        action="store_true",
        help=(
            "Process only targets whose existing stage-1 chain_map.tsv contains a "
            "non-PASS row."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_seed < 1 or args.workers < 1 or args.max_iterations < 1:
        raise ValueError("sample/worker/iteration counts must be positive")
    if not 0.0 <= args.min_gt_coverage <= 1.0:
        raise ValueError("--min-gt-coverage must be in [0, 1]")
    if not 0.0 <= args.min_dominant_fraction <= 1.0:
        raise ValueError("--min-dominant-fraction must be in [0, 1]")
    if args.tie_tolerance < 0:
        raise ValueError("--tie-tolerance must be non-negative")

    targets = read_targets_xlsx(args.targets_xlsx.expanduser())
    gt_index = index_files_by_stem(args.gt_cif_dir.expanduser(), ".cif")
    prediction_index = index_prediction_seed_dirs(args.pred_dir.expanduser())
    output_root = args.output_dir.expanduser()
    if args.only_current_review:
        selected: list[str] = []
        for pdb_id in targets:
            _, _, has_review = read_base_chain_map(
                output_root / pdb_id / "chain_map.tsv"
            )
            if has_review:
                selected.append(pdb_id)
        targets = selected
    print(f"Targets loaded: {len(targets)}")
    print(f"Workers: {args.workers}")
    print(f"Representative atom: {args.representative_atom}")

    results: dict[str, ResolvedTarget] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdb = {
            executor.submit(
                resolve_target,
                pdb_id,
                gt_path=gt_index.get(pdb_id),
                base_chain_map_path=output_root / pdb_id / "chain_map.tsv",
                seed_dirs=prediction_index.get(pdb_id, {}),
                expected_seeds=args.seeds,
                expected_samples=args.samples_per_seed,
                representative_atom=args.representative_atom,
                min_gt_coverage=args.min_gt_coverage,
                min_dominant_fraction=args.min_dominant_fraction,
                max_iterations=args.max_iterations,
                tie_tolerance=args.tie_tolerance,
            ): pdb_id
            for pdb_id in targets
        }
        completed = 0
        for future in as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                results[pdb_id] = future.result()
            except Exception as exc:
                results[pdb_id] = ResolvedTarget(
                    pdb_id=pdb_id,
                    gt_chains=[],
                    inventory_count=0,
                    candidates=[],
                    inventory_issues=[],
                    fatal_issues=[
                        f"Unhandled target error: {type(exc).__name__}: {exc}"
                    ],
                    review_reasons=[],
                    variants={},
                    status=STATUS_REVIEW,
                )
            completed += 1
            if completed % 25 == 0 or completed == len(targets):
                print(f"Resolved targets: {completed}/{len(targets)}", flush=True)

    variant_count = 0
    for pdb_id in targets:
        variant_count += write_resolved_target(
            output_root,
            results[pdb_id],
            representative_atom=args.representative_atom,
            min_gt_coverage=args.min_gt_coverage,
            min_dominant_fraction=args.min_dominant_fraction,
            overwrite=args.overwrite,
        )

    pass_targets = sum(result.status == STATUS_PASS for result in results.values())
    review_targets = len(results) - pass_targets
    mapped_candidates = sum(
        bool(candidate.chosen_mapping)
        for result in results.values()
        for candidate in result.candidates
    )
    available_candidates = sum(result.inventory_count for result in results.values())
    variant_count_distribution = Counter(
        len(result.variants) for result in results.values()
    )
    zero_variant_targets = variant_count_distribution.get(0, 0)
    single_variant_targets = variant_count_distribution.get(1, 0)
    multi_variant_targets = sum(
        count
        for mapping_count, count in variant_count_distribution.items()
        if mapping_count > 1
    )
    tie_break_candidates = sum(
        candidate.consensus_tie_break
        for result in results.values()
        for candidate in result.candidates
    )
    theoretical_sequence_note_candidates = sum(
        any(
            "predicted CIF theoretical sequence differs from JSON input" in issue
            for issue in candidate.issues
        )
        for result in results.values()
        for candidate in result.candidates
    )
    chain_id_mismatch_candidates = sum(
        any(
            "is absent from base chain_map.tsv" in issue
            or "base chain_map.tsv chains absent from predicted CIF" in issue
            for issue in candidate.issues
        )
        for result in results.values()
        for candidate in result.candidates
    )
    sequence_length_mismatch_candidates = sum(
        any(
            "theoretical-sequence length differs from JSON input" in issue
            for issue in candidate.issues
        )
        for result in results.values()
        for candidate in result.candidates
    )
    concrete_sequence_conflict_candidates = sum(
        any(
            "theoretical sequence conflicts with JSON input" in issue
            for issue in candidate.issues
        )
        for result in results.values()
        for candidate in result.candidates
    )
    print("\nTargets requiring review:")
    for pdb_id in targets:
        result = results[pdb_id]
        if result.status != STATUS_REVIEW:
            continue
        print(f"  {pdb_id}:")
        for issue in result.fatal_issues + result.review_reasons:
            print(f"    - {issue}")
    print("\nFinal summary:")
    print(f"  targets written: {len(results)}")
    print(f"  PASS targets: {pass_targets}")
    print(f"  需要人工审核 targets: {review_targets}")
    print(f"  available candidates: {available_candidates}")
    print(f"  mapped candidates: {mapped_candidates}")
    print(f"  distinct chain_map.tsv variants written: {variant_count}")
    print(f"  targets with 0 mapping variants: {zero_variant_targets}")
    print(f"  targets with exactly 1 mapping variant: {single_variant_targets}")
    print(f"  targets with >1 mapping variants: {multi_variant_targets}")
    print("  mapping-variant count distribution:")
    for mapping_count in sorted(variant_count_distribution):
        print(
            f"    {mapping_count} mapping variant(s): "
            f"{variant_count_distribution[mapping_count]} target(s)"
        )
    print(f"  candidates resolved by consensus tie-break: {tie_break_candidates}")
    print(
        "  candidates with non-blocking prediction-CIF theoretical-sequence notes: "
        f"{theoretical_sequence_note_candidates}"
    )
    print(
        "  candidates with blocking prediction chain-ID-set mismatch: "
        f"{chain_id_mismatch_candidates}"
    )
    print(
        "  candidates with blocking prediction/JSON sequence-length mismatch: "
        f"{sequence_length_mismatch_candidates}"
    )
    print(
        "  candidates with blocking concrete prediction/JSON sequence conflict: "
        f"{concrete_sequence_conflict_candidates}"
    )
    print(f"  output root: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
