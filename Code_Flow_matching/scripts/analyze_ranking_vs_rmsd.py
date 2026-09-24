#!/usr/bin/env python3
"""Compare Protenix ranking_score with high input RMSD using PT build logs.

Reads manifest.tsv (RMSD <= limit) and filtered_samples.tsv (RMSD > limit).
Does not inspect or modify model inputs or choose a score cutoff automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite(value: str) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def auc_low_score_predicts_bad(rows: list[dict]) -> float | None:
    # Mann-Whitney AUC with half credit for score ties.
    bad = [row["score"] for row in rows if row["bad"]]
    good = [row["score"] for row in rows if not row["bad"]]
    if not bad or not good:
        return None
    good_sorted = sorted(good)
    from bisect import bisect_left, bisect_right
    wins = sum(len(good_sorted) - bisect_right(good_sorted, score)
               + 0.5 * (bisect_right(good_sorted, score) - bisect_left(good_sorted, score))
               for score in bad)
    return wins / (len(bad) * len(good))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-root", type=Path,
                        help="Use Protenix summary confidence JSON when older PT logs lack ranking_score")
    parser.add_argument("--rmsd-threshold", type=float, default=30.0)
    args = parser.parse_args()
    run = args.pt_run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    score_files = {}
    if args.prediction_root is not None:
        root = args.prediction_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*_summary_confidence_sample_*.json"):
            parts = path.relative_to(root).parts
            match = re.search(r"_summary_confidence_sample_(\d+)\.json$", path.name)
            seed = next((part[5:] for part in parts if part.startswith("seed_") and part[5:].isdigit()), None)
            if len(parts) >= 4 and parts[0] in {"train", "val", "test"} and seed and match:
                key = (parts[0], parts[1].upper(), seed, match.group(1))
                if key in score_files:
                    raise ValueError(f"duplicate ranking score file for {key}")
                score_files[key] = path
    by_key = {}
    for name in ("manifest.tsv", "filtered_samples.tsv"):
        for row in read_tsv(run / name):
            key = (row["split"].lower(), row["pdb_id"].upper(), str(row["seed"]), str(row["sample"]))
            if key in by_key:
                raise ValueError(f"candidate appears in both PT reports: {key}")
            rmsd = finite(row.get("pre_refinement_aligned_rmsd", ""))
            score = finite(row.get("ranking_score", ""))
            if score is None and key in score_files:
                try:
                    score = finite(json.loads(score_files[key].read_text(encoding="utf-8")).get("ranking_score"))
                except (OSError, ValueError, TypeError):
                    score = None
            if rmsd is None:
                raise ValueError(f"missing or invalid RMSD: {key}")
            by_key[key] = {"split": key[0], "pdb_id": key[1], "seed": key[2],
                           "sample": key[3], "rmsd": rmsd, "score": score,
                           "bad": rmsd > args.rmsd_threshold,
                           "pt_category": "rmsd_filtered" if name == "filtered_samples.tsv" else "main"}
    rows = list(by_key.values())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "pdb_id", "seed", "sample",
                                  "rmsd", "score", "bad", "pt_category"), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["pdb_id"], row["seed"], row["sample"])))
    summary = {}
    pdb_groups = defaultdict(list)
    for row in rows:
        pdb_groups[(row["split"], row["pdb_id"])].append(row)
    pdb_rows = []
    for (split, pdb_id), group in sorted(pdb_groups.items()):
        bad_scores = [row["score"] for row in group if row["bad"] and row["score"] is not None]
        good_scores = [row["score"] for row in group if not row["bad"] and row["score"] is not None]
        pdb_rows.append({"split": split, "pdb_id": pdb_id, "candidates": len(group),
                         "rmsd_gt_threshold": sum(row["bad"] for row in group),
                         "score_median_bad": percentile(bad_scores, 0.5),
                         "score_median_good": percentile(good_scores, 0.5)})
    with (output / "pdb_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "pdb_id", "candidates",
                                  "rmsd_gt_threshold", "score_median_bad", "score_median_good"), delimiter="\t")
        writer.writeheader()
        writer.writerows(pdb_rows)
    for split in ("train", "val", "test"):
        group = [row for row in rows if row["split"] == split]
        scored = [row for row in group if row["score"] is not None]
        bad_scores = [row["score"] for row in scored if row["bad"]]
        good_scores = [row["score"] for row in scored if not row["bad"]]
        summary[split] = {
            "candidates": len(group), "pdbs": len({row["pdb_id"] for row in group}),
            "rmsd_gt_threshold": sum(row["bad"] for row in group),
            "ranking_score_available": len(scored),
            "score_median_rmsd_gt_threshold": percentile(bad_scores, 0.5),
            "score_median_rmsd_le_threshold": percentile(good_scores, 0.5),
            "auc_low_score_predicts_bad": auc_low_score_predicts_bad(scored),
        }
    # Candidate-level threshold sweep. Select any cutoff using train/val only.
    thresholds = sorted({row["score"] for row in rows if row["split"] == "train" and row["score"] is not None})
    if len(thresholds) > 201:
        thresholds = sorted({thresholds[round(i * (len(thresholds) - 1) / 200)] for i in range(201)})
    curve = []
    for threshold in thresholds:
        for split in ("train", "val", "test"):
            group = [row for row in rows if row["split"] == split and row["score"] is not None]
            bad_total = sum(row["bad"] for row in group)
            good_total = len(group) - bad_total
            bad_removed = sum(row["bad"] and row["score"] < threshold for row in group)
            good_removed = sum(not row["bad"] and row["score"] < threshold for row in group)
            curve.append({"split": split, "score_cutoff_exclusive": threshold,
                          "bad_recall": bad_removed / bad_total if bad_total else "",
                          "good_removed_fraction": good_removed / good_total if good_total else "",
                          "bad_removed": bad_removed, "good_removed": good_removed})
    with (output / "score_cutoff_curve.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "score_cutoff_exclusive", "bad_recall",
                                  "good_removed_fraction", "bad_removed", "good_removed"), delimiter="\t")
        writer.writeheader()
        writer.writerows(curve)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
