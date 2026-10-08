"""Summarize paired RNA-backbone RMSD for epoch-49 rank-1 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "pdb_id", "native_chain_id", "predicted_chain_id", "length",
    "protenix_seed", "protenix_sample", "observed_backbone_atom_count",
    "input_backbone_aligned_rmsd", "refined_backbone_aligned_rmsd",
    "backbone_aligned_improvement", "input_aligned_rmsd",
    "refined_aligned_rmsd", "aligned_improvement",
)
KEY_FIELDS = ("pdb_id", "native_chain_id", "predicted_chain_id", "protenix_seed", "protenix_sample")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}; upload the updated evaluator")
        return list(reader)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def summary_for(rows: list[dict[str, str]], split: str) -> tuple[dict, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pdb_id"], row["native_chain_id"])].append(row)
    chain_rows = []
    for (pdb_id, native_chain), members in sorted(groups.items()):
        values = {name: [float(row[name]) for row in members] for name in FIELDS[7:]}
        if not all(math.isfinite(value) for series in values.values() for value in series):
            raise ValueError(f"{split} {pdb_id} {native_chain}: non-finite RMSD")
        chain_rows.append({
            "split": split, "pdb_id": pdb_id, "native_chain_id": native_chain,
            "candidate_count": len(members), "length_nt": int(members[0]["length"]),
            "observed_backbone_atoms": int(members[0]["observed_backbone_atom_count"]),
            "input_backbone_rmsd_a": mean(values["input_backbone_aligned_rmsd"]),
            "refined_backbone_rmsd_a": mean(values["refined_backbone_aligned_rmsd"]),
            "backbone_improvement_a": mean(values["backbone_aligned_improvement"]),
            "input_all_atom_rmsd_a": mean(values["input_aligned_rmsd"]),
            "refined_all_atom_rmsd_a": mean(values["refined_aligned_rmsd"]),
            "all_atom_improvement_a": mean(values["aligned_improvement"]),
        })
    deltas = [row["backbone_improvement_a"] for row in chain_rows]
    return {
        "split": split,
        "candidate_count": len(rows),
        "pdb_chain_count": len(chain_rows),
        "backbone_definition": "P OP1 OP2 OP3 O5' C5' C4' O4' C3' O3' C2' O2' C1'",
        "alignment": "independent proper-rotation Kabsch fit on observed backbone atoms",
        "input_backbone_rmsd_macro_a": mean([row["input_backbone_rmsd_a"] for row in chain_rows]),
        "refined_backbone_rmsd_macro_a": mean([row["refined_backbone_rmsd_a"] for row in chain_rows]),
        "backbone_improvement_macro_a": mean(deltas),
        "backbone_improvement_median_a": median(deltas),
        "backbone_win_count": sum(delta > 1e-6 for delta in deltas),
        "backbone_worsen_count": sum(delta < -1e-6 for delta in deltas),
        "backbone_win_rate": mean([float(delta > 1e-6) for delta in deltas]),
        "input_all_atom_rmsd_macro_a": mean([row["input_all_atom_rmsd_a"] for row in chain_rows]),
        "refined_all_atom_rmsd_macro_a": mean([row["refined_all_atom_rmsd_a"] for row in chain_rows]),
        "all_atom_improvement_macro_a": mean([row["all_atom_improvement_a"] for row in chain_rows]),
    }, chain_rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")

    split_rows = {
        "val": read_rows(args.val_dir / "samples.tsv"),
        "test": read_rows(args.test_dir / "samples.tsv"),
    }
    summaries = []
    chains = []
    for split, rows in split_rows.items():
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError(f"{split}: empty or mismatched samples.tsv")
        keys = [tuple(row[field] for field in KEY_FIELDS) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{split}: duplicate candidate keys")
        if any(int(row["observed_backbone_atom_count"]) < 3 for row in rows):
            raise ValueError(f"{split}: fewer than three observed backbone atoms in a sample")
        summary, chain_rows = summary_for(rows, split)
        summaries.append(summary)
        chains.extend(chain_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "per_pdb_chain.tsv", chains)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    lines = [
        "# Epoch 49 rank1：RNA 骨架 RMSD 前后对照", "",
        "骨架原子：P、OP1、OP2、OP3，以及 O5′、C5′、C4′、O4′、C3′、O3′、C2′、O2′、C1′。",
        "只纳入有真实结构对应坐标的原子；输入和精修结果分别用这些骨架原子独立做 Kabsch 刚体对齐。",
        "改善量 = 精修前 RMSD − 精修后 RMSD，正值表示提升。", "",
        "| 数据集 | PDB 链数 | 骨架前 (Å) | 骨架后 (Å) | 平均改善 (Å) | 中位改善 (Å) | 改善链数 / 比例 | 全原子前 (Å) | 全原子后 (Å) |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['split']} | {s['pdb_chain_count']} | {s['input_backbone_rmsd_macro_a']:.4f} | "
            f"{s['refined_backbone_rmsd_macro_a']:.4f} | {s['backbone_improvement_macro_a']:+.4f} | "
            f"{s['backbone_improvement_median_a']:+.4f} | {s['backbone_win_count']} / "
            f"{s['backbone_win_rate']:.1%} | {s['input_all_atom_rmsd_macro_a']:.4f} | "
            f"{s['refined_all_atom_rmsd_macro_a']:.4f} |"
        )
    lines += ["", "逐 PDB 链结果见 `per_pdb_chain.tsv`。全原子列用于对照，采用全原子独立对齐，不能和骨架列直接相减解释为某类原子的贡献。", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for s in summaries:
        print(f"BACKBONE_SUMMARY split={s['split']} chains={s['pdb_chain_count']} "
              f"before={s['input_backbone_rmsd_macro_a']:.4f} "
              f"after={s['refined_backbone_rmsd_macro_a']:.4f} "
              f"improvement={s['backbone_improvement_macro_a']:+.4f}")


if __name__ == "__main__":
    main()
