#!/usr/bin/env python3
"""Read schema-v2 PTs and compare training exposure with frozen rank-1 splits.

No model, CUDA, PyG or Lightning imports. Extraction is resumable; reports can
be regenerated from the TSV files without access to the original PT dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
import torch

VERSION = "dataset-shift-1"
SPEC = importlib.util.spec_from_file_location(
    "shift_constants", Path(__file__).resolve().parents[1] / "etflow/data/constants.py")
CONSTANTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSTANTS)
ATOM_IDS = CONSTANTS.ATOM_NAME_TO_ID
RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 15: 1.80, 16: 1.80}
TORSIONS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "eta", "theta")
HIST_BINS = 12
SCALARS = ["length", "atom_count", "observed_fraction", "rmsd", "stored_rmsd",
           "residue_error_q90", "good_residue_fraction", "plddt_mean",
           "plddt_residue_mean", "plddt_q10", "plddt_low_fraction", "ranking_score",
           "clash_pairs_per_1000_atoms", "clash_atom_fraction", "clash_overlap_mse_a2",
           "bond_rmse_a", "bond_bad_fraction", "rg_a", "gc_fraction",
           "modified_fraction", "pae_mean", "pde_mean", "contact_mean"]
SCALARS += [f"{t}_error_mean_deg" for t in TORSIONS]
INPUT_FEATURES = ["length", "plddt_residue_mean", "plddt_q10", "plddt_low_fraction",
                  "clash_pairs_per_1000_atoms", "clash_atom_fraction",
                  "clash_overlap_mse_a2", "bond_rmse_a", "bond_bad_fraction", "rg_a",
                  "gc_fraction", "modified_fraction", "pae_mean", "pde_mean", "contact_mean"]
INPUT_FEATURES += [f"pred_{t}_{kind}{harmonic}" for t in TORSIONS for harmonic in (1, 2) for kind in ("sin", "cos")]
ORACLE_FEATURES = ["rmsd", "observed_fraction", "residue_error_q90", "good_residue_fraction"]
ORACLE_FEATURES += [f"{t}_error_mean_deg" for t in TORSIONS]


def array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def fitted_errors(pred, native, mask):
    """Proper row-vector Kabsch, using only native-observed atoms."""
    moving, fixed = pred[mask], native[mask]
    if len(moving) < 3:
        raise ValueError("fewer than three observed atoms")
    mc, fc = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    aligned = (pred - mc) @ (u @ correction @ vt) + fc
    errors = np.full(len(pred), np.nan)
    errors[mask] = np.linalg.norm(aligned[mask] - native[mask], axis=1)
    return float(np.sqrt(np.mean(errors[mask] ** 2))), errors


def dihedral(points):
    """Signed angle in radians, NaN for undefined/degenerate geometry."""
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.empty(0)
    b0, b1, b2 = p[:, 0] - p[:, 1], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    norm = np.linalg.norm(b1, axis=1)
    axis = b1 / np.maximum(norm[:, None], 1e-12)
    v = b0 - (b0 * axis).sum(1)[:, None] * axis
    w = b2 - (b2 * axis).sum(1)[:, None] * axis
    angle = np.arctan2((np.cross(axis, v) * w).sum(1), (v * w).sum(1))
    valid = (norm > 1e-8) & (np.linalg.norm(v, axis=1) > 1e-8) & (np.linalg.norm(w, axis=1) > 1e-8)
    angle[~valid] = np.nan
    return angle


def circular_error(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def angle_hist(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return np.full(HIST_BINS, np.nan)
    # Wrap +pi to -pi, so both endpoints have the same circular meaning.
    wrapped = (values + np.pi) % (2 * np.pi) - np.pi
    counts = np.histogram(wrapped, bins=HIST_BINS, range=(-np.pi, np.pi))[0]
    return counts / counts.sum()


def torsion_indices(payload):
    sequence = payload["sequence"]
    residues = array(payload["residue_index"], int)
    names = array(payload["atom_name_id"], int)
    lookup = {(int(r), int(n)): i for i, (r, n) in enumerate(zip(residues, names))}
    bonds = {tuple(sorted(pair)) for pair in array(payload["geometry_bond_index"], int).T.tolist()}
    labels = payload.get("predicted_label_seq_ids")

    def connected(left):
        if left < 0 or left + 1 >= len(sequence):
            return False
        i, j = lookup.get((left, ATOM_IDS["O3'"])), lookup.get((left + 1, ATOM_IDS["P"]))
        if i is None or j is None or tuple(sorted((i, j))) not in bonds:
            return False
        if labels is not None:
            try:
                return int(labels[left + 1]) == int(labels[left]) + 1
            except (ValueError, TypeError):
                return False
        return True

    found = {t: [] for t in TORSIONS}
    for r, symbol in enumerate(sequence):
        definitions = {
            "alpha": ((r-1, "O3'"), (r, "P"), (r, "O5'"), (r, "C5'")),
            "beta": ((r, "P"), (r, "O5'"), (r, "C5'"), (r, "C4'")),
            "gamma": ((r, "O5'"), (r, "C5'"), (r, "C4'"), (r, "C3'")),
            "delta": ((r, "C5'"), (r, "C4'"), (r, "C3'"), (r, "O3'")),
            "epsilon": ((r, "C4'"), (r, "C3'"), (r, "O3'"), (r+1, "P")),
            "zeta": ((r, "C3'"), (r, "O3'"), (r+1, "P"), (r+1, "O5'")),
            "eta": ((r-1, "C4'"), (r, "P"), (r, "C4'"), (r+1, "P")),
            "theta": ((r, "P"), (r, "C4'"), (r+1, "P"), (r+1, "C4'")),
        }
        if symbol in "AG":
            definitions["chi"] = ((r, "O4'"), (r, "C1'"), (r, "N9"), (r, "C4"))
        elif symbol in "CU":
            definitions["chi"] = ((r, "O4'"), (r, "C1'"), (r, "N1"), (r, "C2"))
        for name, atoms in definitions.items():
            rs = [a[0] for a in atoms]
            if any(not connected(left) for left in range(min(rs), max(rs))):
                continue
            ids = [lookup.get((residue, ATOM_IDS[atom])) for residue, atom in atoms]
            if any(i is None for i in ids):
                continue
            if name not in {"eta", "theta"} and any(
                    tuple(sorted(pair)) not in bonds for pair in zip(ids, ids[1:])):
                continue
            found[name].append(ids)
    return {name: np.asarray(ids, dtype=int).reshape(-1, 4) for name, ids in found.items()}


def clash_metrics(pos, numbers, exclusions):
    """All unique heavy-atom pairs within 4.5 A; no 64-neighbor truncation."""
    radii = np.array([RADII.get(int(z), 0.0) for z in numbers])
    pairs = cKDTree(pos).query_pairs(4.5, output_type="ndarray")
    if len(pairs):
        n = len(pos)
        excluded = np.sort(exclusions, axis=0)
        hashes = excluded[0] * n + excluded[1]
        keep = ~np.isin(pairs[:, 0] * n + pairs[:, 1], hashes)
        keep &= (radii[pairs[:, 0]] > 0) & (radii[pairs[:, 1]] > 0)
        pairs = pairs[keep]
    if not len(pairs):
        return dict(clash_pair_count=0, clash_pairs_per_1000_atoms=0.0,
                    clash_atom_fraction=0.0, clash_overlap_mse_a2=0.0, clash_eligible_pairs=0)
    distance = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    overlap = np.maximum(0, 0.8 * radii[pairs].sum(1) - distance)
    clash = overlap > 0
    return dict(clash_pair_count=int(clash.sum()), clash_pairs_per_1000_atoms=1000 * clash.sum() / len(pos),
                clash_atom_fraction=len(np.unique(pairs[clash])) / len(pos),
                clash_overlap_mse_a2=float(np.mean(overlap ** 2)), clash_eligible_pairs=len(pairs))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def read_scores(paths):
    scores, records = {}, {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in pd.read_csv(path, sep="\t", keep_default_na=False).to_dict("records"):
            split = str(row.get("split", ""))
            pdb = str(row.get("pdb_id", "")).upper()
            seed, sample = number(row.get("seed")), number(row.get("sample"))
            score = number(row.get("ranking_score"))
            if not pdb or not all(np.isfinite(x) for x in (seed, sample, score)):
                continue
            key = (split, pdb, int(seed), int(sample))
            if key in scores and not math.isclose(scores[key], score, abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"conflicting ranking_score for {key}")
            scores[key], records[key] = score, str(path)
    return scores, records


def original_score(payload, task):
    if math.isfinite(task["ranking_score"]):
        return task["ranking_score"], task["score_source"]
    score = number(payload.get("ranking_score"))
    if math.isfinite(score):
        return score, "pt.ranking_score"
    suffix = f"_summary_confidence_sample_{task['sample']}.json"
    locations = []
    source = payload.get("source_confidence_json")
    if source:
        # full-data and summary-confidence names share the same sample identity.
        p = Path(source)
        locations.append(p.with_name(f"{task['pdb_id'].lower()}{suffix}"))
    root = Path(task["prediction_root"])
    for split in dict.fromkeys((task["split"], "val", "test", "train")):
        for pdb in dict.fromkeys((task["pdb_id"].lower(), task["pdb_id"].upper())):
            locations.append(root / split / pdb / f"seed_{task['seed']}" / "predictions" / f"{pdb}{suffix}")
    values = {}
    for p in locations:
        if p.is_file():
            value = number(json.loads(p.read_text(encoding="utf-8")).get("ranking_score"))
            if math.isfinite(value):
                values[str(p.resolve())] = value
    if values:
        if max(values.values()) - min(values.values()) > 1e-8:
            raise ValueError(f"conflicting original confidence JSONs: {values}")
        return next(iter(values.values())), next(iter(values))
    return np.nan, "missing"


def extract_one(task):
    torch.set_num_threads(1)
    base = {k: task[k] for k in ("key", "path", "size", "mtime_ns", "pool", "split", "pdb_id", "seed", "sample")}
    try:
        p = torch.load(task["path"], map_location="cpu", weights_only=True)
        if p.get("schema_version") != 2:
            raise ValueError(f"expected schema_version=2, got {p.get('schema_version')}")
        pos, native = array(p["pos_pred"], float), array(p["pos"], float)
        mask = array(p["target_mask"], bool).reshape(-1)
        ri = array(p["residue_index"], int).reshape(-1)
        an = array(p["atom_name_id"], int).reshape(-1)
        z = array(p["atomic_numbers"], int).reshape(-1)
        sequence = p["sequence"]
        if (pos.shape != (len(mask), 3) or native.shape != pos.shape
                or len(ri) != len(pos) or len(an) != len(pos) or len(z) != len(pos)
                or not np.isfinite(pos).all() or not np.isfinite(native[mask]).all()
                or ri.min() < 0 or ri.max() >= len(sequence)):
            raise ValueError("invalid coordinates, atom attributes or residue indices")
        rmsd, errors = fitted_errors(pos, native, mask)
        count = np.bincount(ri[mask], minlength=len(sequence))
        rsq = np.bincount(ri[mask], weights=errors[mask] ** 2, minlength=len(sequence))
        rerr = np.sqrt(rsq[count > 0] / count[count > 0])
        plddt = array(p["atom_plddt"], float).reshape(-1)
        if len(plddt) != len(pos) or not np.isfinite(plddt).all() or plddt.min() < 0 or plddt.max() > 1:
            raise ValueError("schema-v2 atom_plddt must have one value per atom in [0,1]")
        rc = np.bincount(ri, minlength=len(sequence))
        residue_plddt = np.bincount(ri, weights=plddt, minlength=len(sequence))[rc > 0] / rc[rc > 0]
        ranking, score_source = original_score(p, task)
        row = dict(base, status="ok", error="", sequence=sequence,
                   native_chain_id=p.get("native_chain_id", ""), length=len(sequence), atom_count=len(pos),
                   observed_fraction=float(mask.mean()), rmsd=rmsd,
                   stored_rmsd=number(p.get("pre_refinement_aligned_rmsd")),
                   residue_error_q90=float(np.quantile(rerr, 0.9)), good_residue_fraction=float((rerr < 0.5).mean()),
                   plddt_mean=float(plddt.mean()), plddt_residue_mean=float(residue_plddt.mean()),
                   plddt_q10=float(np.quantile(residue_plddt, 0.1)), plddt_low_fraction=float((residue_plddt < 0.5).mean()),
                   ranking_score=ranking, score_source=score_source,
                   rg_a=float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, axis=1)))),
                   gc_fraction=(sequence.count("G") + sequence.count("C")) / len(sequence),
                   modified_fraction=float(array(p.get("modified_residue_mask", np.zeros(len(sequence)))).mean()),
                   dataset_rmsd_limit=number(p.get("dataset_max_pre_refinement_rmsd")),
                   generator_version=p.get("generator_version", ""))
        for symbol in "ACGU":
            row[f"fraction_{symbol}"] = sequence.count(symbol) / len(sequence)
        for name, source in (("pae", "token_pair_pae"), ("pde", "token_pair_pde"), ("contact", "contact_probs")):
            row[f"{name}_mean"] = float(array(p[source], float).mean()) if source in p else np.nan
        bonds = array(p["geometry_bond_index"], int).reshape(2, -1)
        ideal = array(p["ideal_bond_length"], float).reshape(-1)
        if bonds.shape[1] != len(ideal):
            raise ValueError("bond count differs from ideal length count")
        if len(ideal):
            berr = np.linalg.norm(pos[bonds[0]] - pos[bonds[1]], axis=1) - ideal
            row["bond_rmse_a"] = float(np.sqrt(np.mean(berr ** 2)))
            row["bond_bad_fraction"] = float((np.abs(berr) > 0.2).mean())
        else:
            row["bond_rmse_a"] = row["bond_bad_fraction"] = np.nan
        exclusions = array(p["clash_exclusion_index"], int).reshape(2, -1)
        row.update(clash_metrics(pos, z, exclusions))
        mapping = array(p["prediction_to_native_residue_index"], int)
        for name, ids in torsion_indices(p).items():
            angles = dihedral(pos[ids])
            hist = angle_hist(angles)
            row[f"pred_{name}_count"] = int(np.isfinite(angles).sum())
            finite = angles[np.isfinite(angles)]
            for harmonic in (1, 2):
                for kind, function in (("sin", np.sin), ("cos", np.cos)):
                    row[f"pred_{name}_{kind}{harmonic}"] = float(function(harmonic * finite).mean()) if len(finite) else np.nan
            for i, value in enumerate(hist):
                row[f"pred_{name}_hist_{i:02d}"] = value
            mapped = mapping[ri[ids]]
            res = ri[ids]
            # Gaps in native sequence must not produce a fabricated torsion.
            valid = mask[ids].all(1) & ((mapped[:, 1:] - mapped[:, :-1]) == (res[:, 1:] - res[:, :-1])).all(1)
            nat = dihedral(native[ids[valid]])
            paired = np.isfinite(nat) & np.isfinite(angles[valid])
            err = np.degrees(circular_error(angles[valid][paired], nat[paired]))
            row[f"{name}_paired_count"] = len(err)
            row[f"{name}_error_mean_deg"] = float(err.mean()) if len(err) else np.nan
            for i, value in enumerate(angle_hist(nat)):
                row[f"native_{name}_hist_{i:02d}"] = value
        embedding = None
        if task["embedding"]:
            e = array(p["rnafm_embedding"], float)
            if e.ndim != 2 or len(e) != len(sequence) or not np.isfinite(e).all():
                raise ValueError("invalid RNA-FM residue embeddings")
            embedding = dict(pool=task["pool"], split=task["split"], pdb_id=task["pdb_id"],
                             sequence=sequence, length=len(sequence), **{f"fm_{i:03d}": v for i, v in enumerate(e.mean(0))})
        return row, embedding
    except Exception as exc:
        return dict(base, status="error", error=f"{type(exc).__name__}: {exc}"), None


def write_tsv(path, frame):
    frame.to_csv(path, sep="\t", index=False, float_format="%.8g", na_rep="")


def discover(root, pool, splits, args, scores, sources):
    tasks = []
    for split in splits:
        directory = root / split
        if not directory.is_dir():
            if pool == "current":
                raise FileNotFoundError(directory)
            continue
        for pdb_dir in sorted(directory.iterdir()):
            if not pdb_dir.is_dir():
                continue
            paths = sorted(pdb_dir.rglob("*.pt"))
            if args.limit_per_pdb:
                paths = paths[:args.limit_per_pdb]
            for i, path in enumerate(paths):
                rel = path.relative_to(directory)
                if len(rel.parts) != 3 or not re.fullmatch(r"seed_\d+", rel.parts[1]) or not re.fullmatch(r"sample_\d+\.pt", rel.parts[2]):
                    raise ValueError(f"unexpected PT layout: {path}")
                seed = int(rel.parts[1][5:])
                sample = int(rel.parts[2][7:-3])
                pdb = pdb_dir.name.upper()
                scorekey = split, pdb, seed, sample
                stat = path.stat()
                tasks.append(dict(key=f"{pool}/{split}/{pdb}/{seed}/{sample}", path=str(path.absolute()),
                                  size=stat.st_size, mtime_ns=stat.st_mtime_ns, pool=pool, split=split, pdb_id=pdb,
                                  seed=seed, sample=sample, prediction_root=str(args.prediction_root.expanduser()),
                                  ranking_score=scores.get(scorekey, np.nan), score_source=sources.get(scorekey, "missing"), embedding=(i == 0)))
    return tasks


def extract(args):
    out = args.output_dir
    fingerprint = hashlib.sha256(json.dumps(dict(version=VERSION, data_root=str(args.data_root.expanduser().resolve()),
        archive_root=str(args.archive_root.expanduser().resolve()), include_archive=args.include_archive,
        score_tables=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in args.score_table],
        prediction_root=str(args.prediction_root.expanduser().resolve()), limit_per_pdb=args.limit_per_pdb), sort_keys=True).encode()).hexdigest()
    meta_path = out / "extraction.json"
    features = out / "features.tsv"
    embeddings = out / "rnafm.tsv"
    if features.exists():
        if not args.resume:
            raise ValueError("features.tsv exists; use --resume or a new output directory")
        if not meta_path.is_file() or json.loads(meta_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("resume configuration differs; use a new output directory")
    scores, sources = read_scores(args.score_table)
    tasks = discover(args.data_root.expanduser(), "current", ("train", "val", "test"), args, scores, sources)
    if args.include_archive:
        archive = args.archive_root.expanduser()
        if not archive.is_dir():
            raise FileNotFoundError(archive)
        tasks += discover(archive, "archive", ("val", "test"), args, scores, sources)
    if len({t["key"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate PT identities")
    for split in ("train", "val", "test"):
        if not any(t["split"] == split and t["pool"] == "current" for t in tasks):
            raise ValueError(f"empty current {split}")
    current_rank = pd.DataFrame(tasks)
    counts = current_rank[current_rank.pool == "current"].groupby(["split", "pdb_id"]).size()
    if any(counts.loc[split].max() != 1 for split in ("val", "test")):
        raise ValueError("current val/test must already contain exactly one frozen rank-1 PT per PDB")
    old = {}
    old_embed = pd.read_csv(embeddings, sep="\t", keep_default_na=False) if embeddings.exists() else pd.DataFrame()
    if features.exists():
        for row in pd.read_csv(features, sep="\t", keep_default_na=False, dtype={"mtime_ns": str}).to_dict("records"):
            old[row["key"]] = row
    # File identity check; failed or score-missing rows are retried on resume.
    reusable, pending = [], []
    for t in tasks:
        r = old.get(t["key"])
        embed_ok = not t["embedding"] or (not old_embed.empty and ((old_embed.pool == t["pool"]) &
                    (old_embed.split == t["split"]) & (old_embed.pdb_id == t["pdb_id"])).any())
        if r and r["status"] == "ok" and number(r.get("ranking_score")) == number(r.get("ranking_score")) and int(r["size"]) == t["size"] and int(r["mtime_ns"]) == t["mtime_ns"] and embed_ok:
            reusable.append(r)
        else:
            pending.append(t)
    meta = dict(version=VERSION, fingerprint=fingerprint, discovered=len(tasks), reused=len(reusable),
                limit_per_pdb=args.limit_per_pdb, data_root=str(args.data_root.expanduser()),
                archive_root=str(args.archive_root.expanduser()), include_archive=args.include_archive)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Rewrite cached rows to remove stale identities; append each result as it completes.
    fields = list(extract_one(pending[0])[0]) if pending else list(reusable[0])
    if pending and "length" not in fields:
        # Use a complete column template even if the first file is corrupt.
        template = {k: "" for k in ("key", "path", "size", "mtime_ns", "pool", "split", "pdb_id", "seed", "sample", "status", "error", "sequence", "native_chain_id")}
        for k in SCALARS + INPUT_FEATURES + ORACLE_FEATURES:
            template[k] = ""
        for t in TORSIONS:
            for k in [f"pred_{t}_count", f"{t}_paired_count"] + [f"{prefix}_{t}_hist_{i:02d}" for prefix in ("pred", "native") for i in range(HIST_BINS)]:
                template[k] = ""
        fields = list(template)
    fields = list(dict.fromkeys(fields + ["dataset_rmsd_limit", "generator_version", "score_source", "clash_pair_count", "clash_eligible_pairs"] + [f"fraction_{s}" for s in "ACGU"]))
    started = time.monotonic()
    print(f"EXTRACT files={len(tasks)} cached={len(reusable)} pending={len(pending)} workers={args.workers}", flush=True)
    embed_rows = old_embed.to_dict("records")
    with features.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in reusable)
        handle.flush()
        executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        results = executor.map(extract_one, pending, chunksize=4) if executor else map(extract_one, pending)
        try:
            for i, (row, embed) in enumerate(results, 1):
                writer.writerow(row)
                if embed is not None:
                    embed_rows = [r for r in embed_rows if (r["pool"], r["split"], r["pdb_id"]) != (embed["pool"], embed["split"], embed["pdb_id"])]
                    embed_rows.append(embed)
                if i % 100 == 0 or i == len(pending):
                    handle.flush()
                    write_tsv(embeddings, pd.DataFrame(embed_rows))
                    print(f"PROGRESS {i}/{len(pending)} elapsed={time.monotonic()-started:.0f}s", flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True)
            write_tsv(embeddings, pd.DataFrame(embed_rows))
    meta["elapsed_seconds"] = time.monotonic() - started
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def make_views(data, factor=2.0, threshold=50):
    current = data[data.pool == "current"].copy()
    train = current[current.split == "train"].copy()
    views = {}
    def add(name, frame, weights):
        frame = frame.copy()
        frame["weight"] = np.asarray(weights, float)
        views[name] = frame
    add("train_candidates", train, np.ones(len(train)))
    exposure_length = train["sampling_length"] if "sampling_length" in train else train.length
    add("train_effective", train, np.where(exposure_length > threshold, factor, 1.0))
    add("train_pdb_equal", train, 1 / train.groupby("pdb_id").pdb_id.transform("size"))
    eligible = train.groupby("pdb_id").ranking_score.transform(lambda x: x.notna().all())
    rank = train[eligible].sort_values(["ranking_score", "seed", "sample"], ascending=[False, True, True]).drop_duplicates("pdb_id")
    add("train_rank1", rank, np.ones(len(rank)))
    for split in ("val", "test"):
        selected = current[current.split == split]
        if selected.groupby("pdb_id").size().max() != 1:
            raise ValueError(f"{split} is not a frozen rank-1 dataset")
        add(f"{split}_rank1", selected, np.ones(len(selected)))
        archived = data[(data.pool == "archive") & (data.split == split)]
        if len(archived):
            add(f"{split}_all_pdb_equal", archived, 1 / archived.groupby("pdb_id").pdb_id.transform("size"))
    return views


def weighted_quantile(x, weights, q):
    order = np.argsort(x)
    x, weights = np.asarray(x)[order], np.asarray(weights)[order]
    return np.interp(q, (np.cumsum(weights) - weights * 0.5) / weights.sum(), x)


def weighted_ks(a, b, wa, wb):
    def cdf(x, w, grid):
        order = np.argsort(x)
        x, w = np.asarray(x)[order], np.asarray(w)[order]
        cumulative = np.r_[0, np.cumsum(w) / w.sum()]
        return cumulative[np.searchsorted(x, grid, side="right")]
    grid = np.unique(np.r_[a, b])
    return float(np.max(np.abs(cdf(a, wa, grid) - cdf(b, wb, grid))))


def bootstrap_mean_difference(a, b, metric, iterations, rng, paired=False):
    """Resample PDB groups, retaining all candidate rows and exposure weights."""
    def groups(frame):
        f = frame[frame[metric].notna()].copy()
        f["wx"] = f[metric] * f.weight
        g = f.groupby("pdb_id")[["wx", "weight"]].sum()
        return g
    ga, gb = groups(a), groups(b)
    if paired:
        common = ga.index.intersection(gb.index)
        ga, gb = ga.loc[common], gb.loc[common]
    if min(len(ga), len(gb)) < 2 or iterations == 0:
        return np.nan, np.nan
    ga, gb = ga.to_numpy(), gb.to_numpy()
    estimates = []
    for _ in range(iterations):
        ia = rng.integers(len(ga), size=len(ga))
        ib = ia if paired else rng.integers(len(gb), size=len(gb))
        aa, bb = ga[ia].sum(0), gb[ib].sum(0)
        estimates.append(bb[0] / bb[1] - aa[0] / aa[1])
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def bounded_reference(train, targets, columns):
    """Train-only imputation/scaling; do not let test alter preprocessing."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    usable = [c for c in columns if c in train and train[c].notna().sum() >= 2 and train[c].std() > 1e-10]
    if not usable:
        raise ValueError("no nonconstant training features")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    ref = scaler.fit_transform(imputer.fit_transform(train[usable]))
    transformed = scaler.transform(imputer.transform(targets[usable]))
    config = dict(features=usable, medians=imputer.statistics_.tolist(), means=scaler.mean_.tolist(), scales=scaler.scale_.tolist(),
                  excluded_features=[c for c in columns if c not in usable])
    return ref, transformed, config


def representative_train(train, cap, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for _, group in train.groupby("pdb_id", sort=True):
        indices.extend(rng.choice(group.index.to_numpy(), min(cap, len(group)), replace=False))
    return train.loc[indices].copy()


def embeddings_and_coverage(data, out, args):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    current = data[data.pool == "current"]
    train = representative_train(current[current.split == "train"], args.embedding_per_pdb, args.seed)
    target = current[current.split != "train"]
    all_points = pd.concat([train, target], ignore_index=True)
    configs, warnings = {}, []
    # Fourier moments keep +pi/-pi adjacent in embedding space. Each torsion
    # has four moment columns, with a 1/sqrt(4) block weight.
    for space, columns in (("input", INPUT_FEATURES), ("oracle", INPUT_FEATURES + ORACLE_FEATURES)):
        ref, points, config = bounded_reference(train, all_points, columns)
        block_scale = np.array([0.5 if re.fullmatch(r"pred_.+_(sin|cos)[12]", c) else 1.0 for c in config["features"]])
        ref, points = ref * block_scale, points * block_scale
        pca = PCA(n_components=min(2, len(ref), ref.shape[1]), random_state=args.seed).fit(ref)
        pc = pca.transform(points)
        config.update(block_scale=block_scale.tolist(), pca_components=pca.components_.tolist(),
                      explained_variance_ratio=pca.explained_variance_ratio_.tolist(), fit_pdbs=train.pdb_id.nunique(), fit_candidates=len(train))
        configs[space] = config
        info = all_points[["key", "pool", "split", "pdb_id", "seed", "sample", "length", "rmsd", "plddt_residue_mean", "ranking_score"]].copy()
        info["pc1"] = pc[:, 0]
        info["pc2"] = pc[:, 1] if pc.shape[1] > 1 else 0
        write_tsv(out / f"embedding_{space}.tsv", info)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for split, color in (("train", "#4477aa"), ("val", "#228833"), ("test", "#cc6677")):
            select = info.split == split
            axes[0].scatter(info.loc[select, "pc1"], info.loc[select, "pc2"], s=9 if split == "train" else 25,
                            alpha=0.25 if split == "train" else 0.9, label=split, c=color)
        axes[0].legend()
        for ax, col in zip(axes[1:], ("length", "rmsd")):
            scatter = ax.scatter(info.pc1, info.pc2, c=info[col], s=10, alpha=0.65, cmap="viridis")
            fig.colorbar(scatter, ax=ax, label=col)
        for ax in axes:
            explained = pca.explained_variance_ratio_
            ax.set(xlabel=f"PC1 ({explained[0]:.1%})", ylabel=f"PC2 ({explained[1] if len(explained) > 1 else 0:.1%})")
        fig.suptitle(f"{space}: PCA fitted on train only; representative candidates")
        fig.tight_layout()
        fig.savefig(out / f"pca_{space}.png", dpi=170)
        plt.close(fig)
        if space == "input":
            coverage = nearest_coverage(ref, points[len(train):], train.pdb_id.to_numpy(), args.seed)
            info_target = target[["key", "split", "pdb_id", "length", "rmsd"]].reset_index(drop=True)
            for k, v in coverage.items():
                if k != "threshold":
                    info_target[k] = v
            info_target["reference_q95"] = coverage["threshold"]
            write_tsv(out / "input_coverage.tsv", info_target)
            configs["coverage"] = {"distance": "Euclidean in full scaled input space, not 2D PCA", "threshold": coverage["threshold"],
                                   "calibration": "up to 2000 train candidates, nearest other-PDB neighbor",
                                   "reference_per_pdb_cap": args.embedding_per_pdb}
            if args.umap:
                try:
                    import umap
                except ImportError:
                    warnings.append("UMAP requested but umap-learn is absent; PCA and full-space coverage completed.")
                else:
                    if len(ref) >= 4:
                        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(ref)-1), random_state=args.seed,
                                            transform_seed=args.seed, n_jobs=1).fit(ref)
                        uv = np.vstack([reducer.embedding_, reducer.transform(points[len(train):])])
                        uinfo = info.drop(columns=["pc1", "pc2"]).copy()
                        uinfo["umap1"], uinfo["umap2"] = uv[:, 0], uv[:, 1]
                        write_tsv(out / "embedding_umap.tsv", uinfo)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        for split, group in uinfo.groupby("split"):
                            ax.scatter(group.umap1, group.umap2, label=split, alpha=0.5, s=15)
                        ax.legend(); ax.set(xlabel="UMAP1", ylabel="UMAP2", title="UMAP fitted on train only")
                        fig.tight_layout(); fig.savefig(out / "umap_input.png", dpi=170); plt.close(fig)
    fm_path = out / "rnafm.tsv"
    if fm_path.is_file():
        fm = pd.read_csv(fm_path, sep="\t")
        fm = fm[fm.pool == "current"].drop_duplicates(["split", "pdb_id"])
        ft = fm[fm.split == "train"]
        if len(ft) >= 2:
            ref, points, config = bounded_reference(ft, fm, [c for c in fm if c.startswith("fm_")])
            pca = PCA(n_components=min(2, len(ref), ref.shape[1]), random_state=args.seed).fit(ref)
            pc = pca.transform(points)
            info = fm[["split", "pdb_id", "sequence", "length"]].copy()
            info["pc1"], info["pc2"] = pc[:, 0], pc[:, 1] if pc.shape[1] > 1 else 0
            write_tsv(out / "embedding_rnafm.tsv", info)
            fq = fm[fm.split != "train"]
            _, queries, _ = bounded_reference(ft, fq, [c for c in fm if c.startswith("fm_")])
            fcov = nearest_coverage(ref, queries, ft.pdb_id.to_numpy(), args.seed)
            finfo = fq[["split", "pdb_id", "length"]].reset_index(drop=True)
            for k, v in fcov.items():
                if k != "threshold":
                    finfo[k] = v
            finfo["reference_q95"] = fcov["threshold"]
            write_tsv(out / "rnafm_coverage.tsv", finfo)
            fig, ax = plt.subplots(figsize=(6, 5))
            for split, group in info.groupby("split"):
                ax.scatter(group.pc1, group.pc2, label=split, alpha=0.5, s=15)
            ax.legend(); ax.set(xlabel="PC1", ylabel="PC2", title="Mean RNA-FM; one point per PDB; train-fitted PCA")
            fig.tight_layout(); fig.savefig(out / "pca_rnafm.png", dpi=170); plt.close(fig)
            configs["rnafm"] = dict(config, explained_variance_ratio=pca.explained_variance_ratio_.tolist(), coverage_q95=fcov["threshold"])
    (out / "preprocessing.json").write_text(json.dumps(configs, indent=2, allow_nan=False), encoding="utf-8")
    return warnings


def nearest_coverage(ref, query, pdbs, seed):
    if len(np.unique(pdbs)) < 2:
        raise ValueError("nearest coverage requires at least two training PDBs")
    tree = cKDTree(ref)
    rng = np.random.default_rng(seed)
    calibration = rng.choice(len(ref), size=min(2000, len(ref)), replace=False)
    # k exceeds every PDB's representative count, guaranteeing an other-PDB neighbor.
    k = min(len(ref), int(pd.Series(pdbs).value_counts().max()) + 1)
    distance, index = tree.query(ref[calibration], k=k)
    distance = np.where(pdbs[index] != pdbs[calibration, None], distance, np.inf)
    threshold = float(np.quantile(distance.min(1), 0.95))
    qdist, qindex = tree.query(query, k=1)
    return dict(nearest_distance=qdist, nearest_train_pdb=pdbs[qindex],
                outside_reference_q95=qdist > threshold, threshold=threshold)


def plots(views, out):
    import matplotlib.pyplot as plt
    chosen = [k for k in ("train_effective", "train_pdb_equal", "train_rank1", "val_rank1", "test_rank1") if len(views[k])]
    metrics = ["length", "rmsd", "plddt_residue_mean", "ranking_score", "clash_pairs_per_1000_atoms", "bond_rmse_a"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.flat, metrics):
        for name in chosen:
            f = views[name].dropna(subset=[metric]).sort_values(metric)
            if len(f):
                mass = f.groupby(metric, sort=True).weight.sum()
                x, y = mass.index.to_numpy(), np.cumsum(mass.to_numpy()) / mass.sum()
                ax.step(np.r_[x[0], x], np.r_[0, y], where="post", label=name)
        ax.set(xlabel=metric, ylabel="Weighted ECDF")
        if metric in {"length", "clash_pairs_per_1000_atoms", "bond_rmse_a"}:
            ax.set_xscale("symlog", linthresh=1 if metric != "bond_rmse_a" else 0.01)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "distributions.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    angle_edges = np.linspace(-180, 180, HIST_BINS + 1)
    for ax, torsion in zip(axes.flat, TORSIONS):
        columns = [f"pred_{torsion}_hist_{i:02d}" for i in range(HIST_BINS)]
        for name in chosen:
            f = views[name].dropna(subset=columns)
            if len(f):
                h = np.average(f[columns].to_numpy(), axis=0, weights=f.weight)
                ax.stairs(h, angle_edges, baseline=None, label=name)
        ax.set(title=torsion, xlabel="Degrees", ylabel="Per-structure normalized mass", xlim=(-180, 180))
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "torsions.png", dpi=170); plt.close(fig)


def circular_hist_wasserstein(p, q):
    """Earth mover distance on a ring of 12 equally spaced angular bins."""
    flow = np.cumsum(np.asarray(p) - np.asarray(q))
    return float(np.abs(flow - np.median(flow)).sum() * 360 / HIST_BINS)


def link_performance(args, out):
    """Join only exact candidate identities; never match just by PDB name."""
    if not args.performance_table:
        return None
    coverage = pd.read_csv(out / "input_coverage.tsv", sep="\t")
    features = pd.read_csv(out / "features.tsv", sep="\t")
    identity = ["split", "pdb_id", "seed", "sample"]
    selected = features[(features.pool == "current") & features.split.isin(["val", "test"]) & (features.status == "ok")]
    records = []
    for path in args.performance_table:
        f = pd.read_csv(path.expanduser(), sep="\t").rename(columns={"protenix_seed": "seed", "protenix_sample": "sample"})
        required = identity + ["input_aligned_rmsd", "refined_aligned_rmsd", "aligned_improvement"]
        if not set(required).issubset(f):
            raise ValueError(f"performance table missing candidate identities/RMSD fields: {path}")
        f.pdb_id = f.pdb_id.str.upper()
        f["performance_source"] = str(path)
        records.append(f)
    perf = pd.concat(records, ignore_index=True)
    if perf.duplicated(identity).any():
        raise ValueError("multiple evaluation rows for same candidate; compare checkpoints in separate report directories")
    match = selected[["key"] + identity].merge(perf, on=identity, how="inner", validate="one_to_one")
    match = coverage.merge(match.drop(columns=["split", "pdb_id", "length"], errors="ignore"), on="key", how="inner", validate="one_to_one")
    if len(match) and (np.abs(match.rmsd - match.input_aligned_rmsd) > 0.01).any():
        raise ValueError("performance input RMSD differs from extracted candidate; check data/native/metric versions")
    write_tsv(out / "coverage_performance.tsv", match)
    rows = []
    for (split, outside), g in match.groupby(["split", "outside_reference_q95"]):
        rows.append(dict(split=split, outside_reference_q95=outside, pdbs=g.pdb_id.nunique(),
                         mean_input_rmsd=g.input_aligned_rmsd.mean(), mean_improvement=g.aligned_improvement.mean(),
                         median_improvement=g.aligned_improvement.median(), worsened_fraction=(g.aligned_improvement < -0.1).mean()))
    write_tsv(out / "coverage_performance_summary.tsv", pd.DataFrame(rows))
    return dict(matched_candidates=len(match), selected_candidates=len(selected),
                interpretation="Observational association only; confounded by length/quality and historical BF16 computation")


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    out = args.output_dir
    data = pd.read_csv(out / "features.tsv", sep="\t", keep_default_na=False)
    for column in SCALARS + INPUT_FEATURES + ORACLE_FEATURES + ["dataset_rmsd_limit"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for name in TORSIONS:
        for column in [f"{prefix}_{name}_hist_{i:02d}" for prefix in ("pred", "native") for i in range(HIST_BINS)]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    failures = data[data.status != "ok"]
    write_tsv(out / "extraction_errors.tsv", failures)
    data = data[data.status == "ok"].copy()
    if data.empty:
        raise ValueError("no valid extracted samples")
    sampling_mismatches = 0
    if args.train_length_table:
        lengths = pd.read_csv(args.train_length_table.expanduser(), sep="\t", usecols=["PDB_ID", "RNA_LENGTH"])
        lengths.PDB_ID = lengths.PDB_ID.str.upper()
        lengths.RNA_LENGTH = pd.to_numeric(lengths.RNA_LENGTH, errors="coerce")
        lengths = lengths.dropna().drop_duplicates()
        if lengths.PDB_ID.duplicated().any() or (lengths.RNA_LENGTH < 1).any():
            raise ValueError("invalid/conflicting train length table")
        mapping = lengths.set_index("PDB_ID").RNA_LENGTH
        data["sampling_length"] = data.pdb_id.map(mapping)
        train_rows = (data.pool == "current") & (data.split == "train")
        if data.loc[train_rows, "sampling_length"].isna().any():
            raise ValueError("training PDB missing from train length table")
        sampling_mismatches = int((data.loc[train_rows, "sampling_length"] != data.loc[train_rows, "length"]).sum())
    leaks = []
    current = data[data.pool == "current"]
    for field in ("pdb_id", "sequence"):
        for value, group in current.groupby(field):
            if group.split.nunique() > 1:
                leaks.append(dict(field=field, value=value, splits=",".join(sorted(group.split.unique()))))
    write_tsv(out / "exact_split_overlaps.tsv", pd.DataFrame(leaks, columns=["field", "value", "splits"]))
    audits = []
    for (pool, split, pdb), group in data.groupby(["pool", "split", "pdb_id"]):
        seeds = group.groupby("seed")["sample"].apply(set)
        complete = len(seeds) == 50 and all(samples == {0, 1, 2, 3} for samples in seeds)
        nscore = group.ranking_score.notna().sum()
        audits.append(dict(pool=pool, split=split, pdb_id=pdb, candidate_count=len(group), score_count=nscore,
                           complete_50x4=int(complete), length_variants=group.length.nunique(), sequence_variants=group.sequence.nunique(),
                           rmsd_gt30_count=int((group.rmsd > 30).sum()), rmsd_limit_min=group.dataset_rmsd_limit.min()))
    write_tsv(out / "pdb_audit.tsv", pd.DataFrame(audits))
    selection_audit = []
    for split in ("val", "test"):
        selected = current[current.split == split]
        archived = data[(data.pool == "archive") & (data.split == split)]
        for _, row in selected.iterrows():
            group = archived[archived.pdb_id == row.pdb_id]
            best = group.sort_values(["ranking_score", "seed", "sample"], ascending=[False, True, True]).head(1)
            checkable = len(group) > 0 and group.ranking_score.notna().all() and pd.notna(row.ranking_score)
            matched = bool(checkable and int(best.iloc[0].seed) == int(row.seed) and int(best.iloc[0]["sample"]) == int(row["sample"]))
            selection_audit.append(dict(split=split, pdb_id=row.pdb_id, archived_candidates=len(group),
                                       selected_seed=row.seed, selected_sample=row["sample"],
                                       score_checkable=int(checkable), matches_archive_rank1=int(matched) if checkable else ""))
    write_tsv(out / "selection_audit.tsv", pd.DataFrame(selection_audit))
    views = make_views(data, args.long_factor, args.long_threshold)
    summaries, shifts, strata, torsion_shifts = [], [], [], []
    rng = np.random.default_rng(args.seed)
    for name, frame in views.items():
        for metric in SCALARS:
            f = frame.dropna(subset=[metric])
            if len(f):
                summaries.append(dict(view=name, metric=metric, candidates=len(f), pdbs=f.pdb_id.nunique(),
                    mean=np.average(f[metric], weights=f.weight), **{f"q{q}": weighted_quantile(f[metric], f.weight, q/100) for q in (5, 25, 50, 75, 95)},
                    missing_fraction=1-len(f)/max(1, len(frame))))
        f = frame.copy()
        f["length_bin"] = pd.cut(f.length, [0, 50, 100, 200, np.inf], labels=["<=50", "51-100", "101-200", ">200"])
        f["rmsd_bin"] = pd.cut(f.rmsd, [-np.inf, 2, 5, 10, 20, 30, np.inf], right=False, labels=["<2", "2-5", "5-10", "10-20", "20-30", ">=30"])
        for field in ("length_bin", "rmsd_bin"):
            for label, g in f.groupby(field, observed=True):
                result = dict(view=name, stratum=field, bin=str(label), candidates=len(g), pdbs=g.pdb_id.nunique(), weight_share=g.weight.sum()/f.weight.sum())
                for metric in ("rmsd", "plddt_residue_mean", "ranking_score", "clash_pairs_per_1000_atoms", "bond_rmse_a"):
                    h = g.dropna(subset=[metric])
                    result[f"mean_{metric}"] = np.average(h[metric], weights=h.weight) if len(h) else np.nan
                strata.append(result)
    pairs = [(reference, f"{split}_rank1") for reference in ("train_effective", "train_pdb_equal", "train_rank1") for split in ("val", "test")]
    pairs += [("val_rank1", "test_rank1")]
    pairs += [(f"{split}_all_pdb_equal", f"{split}_rank1") for split in ("val", "test") if f"{split}_all_pdb_equal" in views]
    for left, right in pairs:
        print(f"COMPARE {left} -> {right}", flush=True)
        for t in TORSIONS:
            columns = [f"pred_{t}_hist_{i:02d}" for i in range(HIST_BINS)]
            a, b = views[left].dropna(subset=columns), views[right].dropna(subset=columns)
            if len(a) and len(b):
                p = np.average(a[columns], weights=a.weight, axis=0)
                q = np.average(b[columns], weights=b.weight, axis=0)
                torsion_shifts.append(dict(reference=left, target=right, torsion=t,
                                          reference_pdbs=a.pdb_id.nunique(), target_pdbs=b.pdb_id.nunique(),
                                          circular_wasserstein_deg=circular_hist_wasserstein(p, q)))
        for metric in SCALARS:
            a, b = views[left].dropna(subset=[metric]), views[right].dropna(subset=[metric])
            if not len(a) or not len(b):
                continue
            difference = np.average(b[metric], weights=b.weight)-np.average(a[metric], weights=a.weight)
            ci = bootstrap_mean_difference(views[left], views[right], metric, args.bootstrap, rng, paired="_all_pdb_equal" in left)
            shifts.append(dict(reference=left, target=right, metric=metric, reference_pdbs=a.pdb_id.nunique(), target_pdbs=b.pdb_id.nunique(),
                               ks_effect=weighted_ks(a[metric], b[metric], a.weight, b.weight),
                               wasserstein=wasserstein_distance(a[metric], b[metric], a.weight, b.weight),
                               mean_difference=difference, ci95_low=ci[0], ci95_high=ci[1],
                               outside_reference_range_fraction=np.average((b[metric]<a[metric].min()) | (b[metric]>a[metric].max()), weights=b.weight)))
    write_tsv(out / "distribution_summary.tsv", pd.DataFrame(summaries))
    write_tsv(out / "shift_metrics.tsv", pd.DataFrame(shifts))
    write_tsv(out / "strata.tsv", pd.DataFrame(strata))
    write_tsv(out / "torsion_shift.tsv", pd.DataFrame(torsion_shifts))
    plots(views, out)
    warnings = embeddings_and_coverage(data, out, args)
    performance = link_performance(args, out)
    if sampling_mismatches:
        warnings.append(f"{sampling_mismatches} train PTs differ in length from the sampler audit table; train_effective uses the audit length.")
    if not args.train_length_table:
        warnings.append("No sampler length table supplied; train_effective assumes PT sequence length matches the training audit length.")
    missing_scores = int(data.ranking_score.isna().sum())
    selection_mismatch = [r for r in selection_audit if r["score_checkable"] and not r["matches_archive_rank1"]]
    if missing_scores:
        warnings.append(f"{missing_scores} candidates lack ranking_score; train_rank1 excludes every PDB with any missing score.")
    if len(failures):
        warnings.append(f"{len(failures)} PT extraction errors; distributions use successful PTs only. Inspect extraction_errors.tsv.")
    if selection_mismatch:
        warnings.append(f"{len(selection_mismatch)} current selected PTs disagree with archived rank-1. Inspect selection_audit.tsv.")
    if leaks:
        warnings.append("Exact PDB/sequence overlap detected. Sequence matches do not replace a homology-cluster audit.")
    metadata = json.loads((out / "extraction.json").read_text(encoding="utf-8"))
    if metadata.get("limit_per_pdb"):
        warnings.append("SMOKE ONLY: extraction truncated per PDB; train_rank1 is highest among this subset, not all 200 candidates.")
    if any(a["length_variants"] > 1 or a["sequence_variants"] > 1 for a in audits):
        warnings.append("Some PDBs have inconsistent length/sequence across candidates; inspect pdb_audit.tsv.")
    rmsd_difference = np.abs(data.rmsd - data.stored_rmsd)
    counts = {name: dict(candidates=len(frame), pdbs=int(frame.pdb_id.nunique())) for name, frame in views.items()}
    summary = dict(version=VERSION, views=counts, extraction_errors=len(failures), missing_scores=missing_scores,
                   rmsd_gt30_current=int((current.rmsd > 30).sum()), stored_rmsd_disagreement_gt_0_01=int((rmsd_difference > 0.01).sum()),
                   long_factor=args.long_factor, long_threshold=args.long_threshold, bootstrap_unit="PDB", warnings=warnings,
                   sampler_length_table=str(args.train_length_table) if args.train_length_table else None,
                   performance_link=performance,
                   selection_mismatches=len(selection_mismatch), limitations=["No causal attribution from distribution plots",
                   "Exact sequence check does not detect all homology", "Input summaries/mean RNA-FM are lossy representations, not the model's complete input",
                   "Nearest-neighbor coverage is a heuristic; dependent on feature set and reference cap",
                   "PDB bootstrap does not account for homology between PDBs", "Current rank1 verified against surviving PT archive, not the original missing/filtered candidates"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 数据集分布诊断", "", "输入为现有 PT；不修改数据、划分、候选或模型。val/test 使用已经冻结的最高 ranking_score 候选。", "",
             "| 分布 | PT 数 | PDB 数 |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['candidates']} | {v['pdbs']} |" for k, v in counts.items()]
    lines += ["", f"训练采样：长度 >{args.long_threshold} nt 的候选权重为 {args.long_factor}；其余为 1。与 V3 配置一致时才代表实际训练曝光。",
              "", "## 需要先核对", ""]
    lines += [f"- {warning}" for warning in warnings] or ["- 抽取与分数检查未发现警告；仍需检查原始 200 候选和同源关系。"]
    lines += [f"- 当前集合 RMSD >30 Å 的候选：{summary['rmsd_gt30_current']}；存储/重算 RMSD 差值 >0.01 Å：{summary['stored_rmsd_disagreement_gt_0_01']}。",
              "", "## 分布差异最大的指标（按 KS 幅度排序）", "", "| 比较 | 指标 | KS | 均值差：目标−训练 | PDB bootstrap 95% CI |", "|---|---|---:|---:|---|"]
    top = sorted([r for r in shifts if r["reference"] == "train_effective"], key=lambda r: -r["ks_effect"])[:12]
    lines += [f"| {r['target']} | {r['metric']} | {r['ks_effect']:.3f} | {r['mean_difference']:.3g} | [{r['ci95_low']:.3g}, {r['ci95_high']:.3g}] |" for r in top]
    lines += ["", "## 解释边界", "", "- train_effective 是按 PT 采样且长链加权；train_pdb_equal 每个 PDB 等权；train_rank1 是训练端也选最高分，用于识别选择效应，未改变训练数据。",
              "- 标准化、缺失值填补、PCA 和可选 UMAP 只拟合训练参考。input 空间是无 native 的输入摘要，不含 ranking_score；oracle 空间含 RMSD/二面角误差，属于监督难度诊断。RNA-FM 空间每个 PDB 一个均值向量。二面角降维使用 sin/cos 的一、二阶矩，跨 ±180° 连续。这些摘要不是模型完整输入。",
              "- input_coverage.tsv 在完整输入特征空间计算最近距离，以训练中跨 PDB 距离的第 95 百分位作参考阈值。二维图分开不等于不可学习；图重合也不证明覆盖充分。",
              "- clash 为重原子 0.8×范德华半径和阈值，排除 PT 的 1–2/1–3 配对；统计全部邻居、不截断到 64，因此不是训练 clash loss 或 MolProbity clashscore。",
              "- 二面角按 12 箱周期直方图比较；native 误差取圆周最短距离，缺原子、序列断点和退化角不计为零。每结构先归一化，避免长链天然占更大权重。",
              "- bootstrap 重采样整个 PDB，保留其候选；不同 PDB 的同源性未消除。没有把数万个候选当独立样本给出 KS p 值。",
              "- RMSD >30 Å 的保存策略、部分 PDB 不足 200 个候选会影响分布。rank1 比较仅覆盖现有可用 PT；不能宣称已核实原始所有 200 个预测。",
              "- 当前模型已有 BF16 图均值错误记录。分布差异和该数值问题应分别检验；本报告不能给出模型差的因果结论。",
              "", "## 后续验证", "", "保持 val/test PDB 划分。用 val_rank1 选 checkpoint/早停，用归档 val 全候选作稳定性面板；同时考察宏平均 RMSD 改善、低误差输入回退率和局部几何恶化。按长度、置信度、输入误差固定分层，并报告每层 PDB 数。",
              "测试集用于本次已声明的事后诊断，不能继续承担调参/选权重。若后续针对测试集暴露的问题反复迭代，需要新的独立终测集。",
              "", "![分布](distributions.png)", "", "![二面角](torsions.png)", "", "![输入空间](pca_input.png)", "", "![监督难度空间](pca_oracle.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT {out / 'report.md'}", flush=True)
    for warning in warnings:
        print(f"WARNING {warning}", flush=True)
    return bool(len(failures) or selection_mismatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Data_PT_V2")
    parser.add_argument("--archive-root", type=Path, default=Path.home() / "Data_PT_V2_all_candidates_20261013")
    parser.add_argument("--prediction-root", type=Path, default=Path.home() / "Data_V2")
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--score-table", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit-per-pdb", type=int, default=0, help="Smoke ONLY: cap extraction, invalidates full-pool rank1 comparison")
    parser.add_argument("--embedding-per-pdb", type=int, default=8)
    parser.add_argument("--long-factor", type=float, default=2.0)
    parser.add_argument("--long-threshold", type=int, default=50)
    parser.add_argument("--train-length-table", type=Path, help="Same PDB_ID/RNA_LENGTH audit table used by the training sampler")
    parser.add_argument("--performance-table", type=Path, action="append", default=[], help="Optional exact-candidate evaluation samples.tsv; repeat for val/test")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap", action="store_true", help="Optional umap-learn, train-fitted")
    args = parser.parse_args()
    if args.workers < 1 or args.limit_per_pdb < 0 or args.embedding_per_pdb < 1 or args.bootstrap < 0 or args.long_threshold < 1 or not math.isfinite(args.long_factor) or args.long_factor < 1:
        parser.error("invalid extraction, sampling or bootstrap settings")
    args.score_table = [p.expanduser() for p in args.score_table]
    args.output_dir = args.output_dir.expanduser().resolve()
    roots = [args.data_root.expanduser().resolve(), args.archive_root.expanduser().resolve(), args.prediction_root.expanduser().resolve()]
    if any(args.output_dir == root or root in args.output_dir.parents for root in roots):
        parser.error("output-dir must be outside all data/prediction roots")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        extract(args)
    bad = report(args)
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
