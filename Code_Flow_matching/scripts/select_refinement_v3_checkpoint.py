"""Select one V3 checkpoint from validation runs before touching test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def dataset_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (
                row["pdb_id"], row["native_chain_id"], row["protenix_seed"],
                row["protenix_sample"], row["sample_id"], row["sample_path"],
            )
            digest.update("\t".join(key).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.screen_root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["split"] != "val":
            raise ValueError(f"Non-validation input: {path}")
        rows.append({
            "checkpoint": summary["checkpoint"],
            "validation_dir": str(path.parent.resolve()),
            "candidate_count": summary["candidate_count"],
            "pdb_chain_count": summary["pdb_chain_count"],
            "dataset_signature_sha256": dataset_signature(path.parent / "samples.tsv"),
            "input_rmsd_macro_a": summary["pdb_chain_macro_input_rmsd_mean"],
            "refined_rmsd_macro_a": summary["pdb_chain_macro_refined_rmsd_mean"],
            "improvement_macro_a": summary["pdb_chain_macro_improvement_mean"],
            "win_rate_macro": summary["pdb_chain_win_rate"],
        })
    if not rows:
        raise SystemExit(f"No validation summaries under {args.screen_root}")
    if len({row["checkpoint"] for row in rows}) != len(rows):
        raise ValueError("Duplicate checkpoints in validation runs")
    if len({(row["candidate_count"], row["pdb_chain_count"]) for row in rows}) != 1:
        raise ValueError("Validation runs cover different numbers of candidates or chains")
    if len({row["dataset_signature_sha256"] for row in rows}) != 1:
        raise ValueError("Validation runs contain different candidate identities")
    baseline = rows[0]["input_rmsd_macro_a"]
    if any(abs(row["input_rmsd_macro_a"] - baseline) > 1e-4 for row in rows):
        raise ValueError("Validation runs have different input RMSD baselines")
    if not all(math.isfinite(row["refined_rmsd_macro_a"]) for row in rows):
        raise ValueError("A validation score is not finite")
    rows.sort(key=lambda row: (row["refined_rmsd_macro_a"], row["checkpoint"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "checkpoint_comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    selected = rows[0]["checkpoint"]
    (args.output_dir / "selected_checkpoint.txt").write_text(selected + "\n", encoding="utf-8")
    print(f"SELECTED_CHECKPOINT={selected}")
    print("Selection criterion: lowest PDB-chain macro refined RMSD on validation")


if __name__ == "__main__":
    main()
