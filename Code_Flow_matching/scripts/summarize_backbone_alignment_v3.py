"""Compare backbone-fit and all-observed-atom-fit backbone RMSD for rank-1 V3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    ("backbone_fit", "input_backbone_aligned_rmsd", "refined_backbone_aligned_rmsd"),
    ("all_atom_fit_backbone", "input_backbone_after_all_atom_fit_rmsd",
     "refined_backbone_after_all_atom_fit_rmsd"),
)


def read_samples(directory: Path, split: str) -> list[dict[str, str]]:
    path = directory / "samples.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"split", "sample_id", "pdb_id", "native_chain_id",
                    "observed_backbone_atom_count", "input_aligned_rmsd",
                    "refined_aligned_rmsd"}
        for _, before, after in METRICS:
            required.update((before, after))
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or ())) }")
        rows = list(reader)
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"{path}: empty or wrong split")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate sample IDs")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["candidate_count"] != len(rows) or summary["split"] != split:
        raise ValueError(f"{path}: evaluation summary mismatch")
    return rows


def analyze(rows: list[dict[str, str]], split: str) -> tuple[dict, list[dict]]:
    # Rank-1 should contain exactly one candidate per PDB chain.
    keys = [(row["pdb_id"], row["native_chain_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{split}: expected one rank-1 candidate per PDB chain")
    details = []
    for row in sorted(rows, key=lambda item: (item["pdb_id"], item["native_chain_id"])):
        if int(row["observed_backbone_atom_count"]) < 3:
            raise ValueError(f"{row['sample_id']}: fewer than three backbone atoms")
        detail = {"split": split, "sample_id": row["sample_id"],
                  "pdb_id": row["pdb_id"], "native_chain_id": row["native_chain_id"],
                  "observed_backbone_atom_count": int(row["observed_backbone_atom_count"])}
        for name, before, after in METRICS:
            initial, final = float(row[before]), float(row[after])
            if not (math.isfinite(initial) and math.isfinite(final)):
                raise ValueError(f"{row['sample_id']}: nonfinite {name}")
            detail[f"{name}_before_a"] = initial
            detail[f"{name}_after_a"] = final
            detail[f"{name}_improvement_a"] = initial - final
        detail["all_atom_before_a"] = float(row["input_aligned_rmsd"])
        detail["all_atom_after_a"] = float(row["refined_aligned_rmsd"])
        # Fitting on the scored backbone subset must be no worse than fitting on
        # a larger atom set and then scoring just that subset.
        for stage in ("before", "after"):
            if detail[f"backbone_fit_{stage}_a"] > detail[f"all_atom_fit_backbone_{stage}_a"] + 2e-4:
                raise ValueError(f"{row['sample_id']}: Kabsch subset optimality violated")
        details.append(detail)

    result = {"split": split, "pdb_chain_count": len(details),
              "candidate_count": len(rows),
              "definition": "native-observed atoms only; independent proper-rotation Kabsch fits for input and refined"}
    for name, _, _ in METRICS:
        improvements = [item[f"{name}_improvement_a"] for item in details]
        result[f"{name}_before_mean_a"] = statistics.mean(item[f"{name}_before_a"] for item in details)
        result[f"{name}_after_mean_a"] = statistics.mean(item[f"{name}_after_a"] for item in details)
        result[f"{name}_improvement_mean_a"] = statistics.mean(improvements)
        result[f"{name}_improvement_median_a"] = statistics.median(improvements)
        result[f"{name}_improved_count"] = sum(delta > 1e-6 for delta in improvements)
        result[f"{name}_worsened_count"] = sum(delta < -1e-6 for delta in improvements)
    result["both_methods_improved_count"] = sum(
        item["backbone_fit_improvement_a"] > 1e-6 and
        item["all_atom_fit_backbone_improvement_a"] > 1e-6 for item in details
    )
    result["methods_disagree_on_direction_count"] = sum(
        (item["backbone_fit_improvement_a"] > 1e-6) !=
        (item["all_atom_fit_backbone_improvement_a"] > 1e-6) for item in details
    )
    return result, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summaries, details = [], []
    for split, directory in (("val", args.val_dir), ("test", args.test_dir)):
        summary, per_chain = analyze(read_samples(directory, split), split)
        summaries.append(summary)
        details.extend(per_chain)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_pdb_chain.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(details)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    lines = [
        "# V3 rank-1：两种骨架 RMSD 对齐口径", "",
        "两种口径均只使用 native 有对应坐标的原子，输入和精修结果各自独立做刚体 Kabsch 对齐。",
        "骨架原子为 P、OP1、OP2、OP3、O5′、C5′、C4′、O4′、C3′、O3′、C2′、O2′、C1′。",
        "`骨架对齐`先筛骨架再对齐；`全原子对齐后取骨架`先用全部可观测原子对齐，再在同一变换下只算骨架 RMSD。",
        "改善量 = 精修前 − 精修后；正数表示精修改善。", "",
        "| split | 口径 | 链数 | 前 (Å) | 后 (Å) | 平均改善 (Å) | 中位改善 (Å) | 改善链数 | 变差链数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        for name, label in (("backbone_fit", "骨架对齐"),
                            ("all_atom_fit_backbone", "全原子对齐后取骨架")):
            lines.append(
                f"| {summary['split']} | {label} | {summary['pdb_chain_count']} | "
                f"{summary[f'{name}_before_mean_a']:.4f} | {summary[f'{name}_after_mean_a']:.4f} | "
                f"{summary[f'{name}_improvement_mean_a']:+.4f} | "
                f"{summary[f'{name}_improvement_median_a']:+.4f} | "
                f"{summary[f'{name}_improved_count']} | {summary[f'{name}_worsened_count']} |"
            )
    lines += ["", "逐链数值见 `per_pdb_chain.tsv`。两种 RMSD 的绝对值不同，是对齐目标不同；比较精修前后变化时须保持同一口径。", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for summary in summaries:
        print(f"BACKBONE_ALIGNMENT_SUMMARY split={summary['split']} "
              f"chains={summary['pdb_chain_count']} "
              f"backbone_fit_improvement={summary['backbone_fit_improvement_mean_a']:+.4f} "
              f"all_atom_fit_backbone_improvement={summary['all_atom_fit_backbone_improvement_mean_a']:+.4f}")


if __name__ == "__main__":
    main()
