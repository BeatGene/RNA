"""Create paired validation/test RNA refinement analysis from V3 TSV results.

Only Python's standard library is required, so this can also run locally after
copying evaluation results back from the GPU server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


RMSD_BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, math.inf))
PHYSICS = ("bond", "clash", "plane")


def number(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def median(values) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def fraction(values) -> float:
    return mean(float(value) for value in values)


def fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_sample_rows(directory: Path) -> tuple[list[dict], dict]:
    with (directory / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with (directory / "samples.tsv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != summary["candidate_count"]:
        raise ValueError(f"candidate count mismatch in {directory}")
    for row in rows:
        for field in (
            "input_aligned_rmsd", "refined_aligned_rmsd", "aligned_improvement",
            "length", "mean_plddt", "input_bond_loss_a2", "refined_bond_loss_a2",
            "input_clash_loss_a2", "refined_clash_loss_a2",
            "input_plane_loss_a2", "refined_plane_loss_a2",
        ):
            row[field] = number(row.get(field, "nan"))
    if any(row["split"] != summary["split"] for row in rows):
        raise ValueError(f"split mismatch in {directory}")
    return rows, summary


def chain_rows(samples: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in samples:
        groups[(row["pdb_id"], row["native_chain_id"])].append(row)
    result = []
    for (pdb_id, chain), group in sorted(groups.items()):
        item = {
            "pdb_id": pdb_id,
            "native_chain_id": chain,
            "sample_count": len(group),
            "length": median(row["length"] for row in group),
            "input_aligned_rmsd": mean(row["input_aligned_rmsd"] for row in group),
            "refined_aligned_rmsd": mean(row["refined_aligned_rmsd"] for row in group),
        }
        item["aligned_improvement"] = item["input_aligned_rmsd"] - item["refined_aligned_rmsd"]
        for name in PHYSICS:
            for phase in ("input", "refined"):
                field = f"{phase}_{name}_loss_a2"
                item[field] = mean(row[field] for row in group)
        result.append(item)
    return result


def length_bin(length: float) -> str:
    if length <= 50:
        return "<=50 nt"
    if length <= 100:
        return "51-100 nt"
    if length <= 200:
        return "101-200 nt"
    return ">200 nt"


def rmsd_bin(value: float) -> str:
    for low, high in RMSD_BINS:
        if low <= value < high:
            return f"[{low},{high if math.isfinite(high) else 'inf'})"
    return "invalid"


def metric_row(split: str, group: str, unit: str, rows: list[dict], denominator: int) -> dict:
    before = [row["input_aligned_rmsd"] for row in rows]
    after = [row["refined_aligned_rmsd"] for row in rows]
    deltas = [a - b for a, b in zip(before, after)]
    return {
        "split": split,
        "group": group,
        "unit": unit,
        "count": len(rows),
        "share": len(rows) / denominator if denominator else math.nan,
        "input_rmsd_mean_a": mean(before),
        "refined_rmsd_mean_a": mean(after),
        "improvement_mean_a": mean(deltas),
        "input_rmsd_median_a": median(before),
        "refined_rmsd_median_a": median(after),
        "improvement_median_a": median(deltas),
        "win_rate": fraction(delta > 1e-6 for delta in deltas),
        "worsen_rate": fraction(delta < -1e-6 for delta in deltas),
        "damage_ge_0.5_rate": fraction(delta <= -0.5 for delta in deltas),
        "post_rmsd_ge_2_rate": fraction(value >= 2 for value in after),
    }


def physics_rows(split: str, group: str, unit: str, rows: list[dict]) -> list[dict]:
    result = []
    for name in PHYSICS:
        field_before = f"input_{name}_loss_a2"
        field_after = f"refined_{name}_loss_a2"
        paired = [
            (row[field_before], row[field_after]) for row in rows
            if math.isfinite(row[field_before]) and math.isfinite(row[field_after])
        ]
        result.append({
            "split": split,
            "group": group,
            "unit": unit,
            "loss": name,
            "count": len(paired),
            "input_loss_a2": mean(before for before, _ in paired),
            "refined_loss_a2": mean(after for _, after in paired),
            "change_after_minus_before_a2": mean(after - before for before, after in paired),
            "worsen_rate": fraction(after > before + 1e-8 for before, after in paired),
        })
    return result


def distribution_rows(split: str, unit: str, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    distribution = []
    transitions = []
    for low, high in RMSD_BINS:
        label = rmsd_bin(low)
        group = [row for row in rows if rmsd_bin(row["input_aligned_rmsd"]) == label]
        distribution.append({
            "split": split,
            "unit": unit,
            "input_bin_a": label,
            "count": len(group),
            "share": len(group) / len(rows) if rows else math.nan,
            "input_rmsd_mean_a": mean(row["input_aligned_rmsd"] for row in group),
            "refined_rmsd_mean_a": mean(row["refined_aligned_rmsd"] for row in group),
            "improvement_mean_a": mean(row["aligned_improvement"] for row in group),
            "win_rate": fraction(row["aligned_improvement"] > 1e-6 for row in group),
        })
        for after_low, _ in RMSD_BINS:
            after_label = rmsd_bin(after_low)
            count = sum(rmsd_bin(row["refined_aligned_rmsd"]) == after_label for row in group)
            transitions.append({
                "split": split,
                "unit": unit,
                "input_bin_a": label,
                "refined_bin_a": after_label,
                "count": count,
                "within_input_bin_rate": count / len(group) if group else math.nan,
            })
    return distribution, transitions


def histogram_rows(split: str, unit: str, rows: list[dict]) -> list[dict]:
    result = []
    for low, _ in RMSD_BINS:
        label = rmsd_bin(low)
        input_count = sum(rmsd_bin(row["input_aligned_rmsd"]) == label for row in rows)
        refined_count = sum(rmsd_bin(row["refined_aligned_rmsd"]) == label for row in rows)
        result.append({
            "split": split, "unit": unit, "bin_a": label,
            "input_count": input_count, "refined_count": refined_count,
            "input_share": input_count / len(rows) if rows else math.nan,
            "refined_share": refined_count / len(rows) if rows else math.nan,
        })
    return result


def foldbench_overlap(chains: list[dict], targets_path: Path) -> list[dict]:
    if not targets_path.is_file():
        return []
    with targets_path.open("r", encoding="utf-8", newline="") as handle:
        targets = list(csv.DictReader(handle))
    target_map = {
        (row["pdb_id"].split("-", 1)[0].upper(), row["chain_id"]): row["pdb_id"]
        for row in targets
    }
    result = []
    for row in chains:
        key = row["pdb_id"].upper(), row["native_chain_id"]
        if key in target_map:
            result.append({
                "pdb_id": row["pdb_id"],
                "native_chain_id": row["native_chain_id"],
                "foldbench_target_id": target_map[key],
                "sample_count": row["sample_count"],
                "input_rmsd_mean_a": row["input_aligned_rmsd"],
                "refined_rmsd_mean_a": row["refined_aligned_rmsd"],
                "improvement_mean_a": row["aligned_improvement"],
            })
    return result


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(fmt(row[field]) for field in fields) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foldbench-targets", type=Path, help="FoldBench targets/monomer_rna.csv")
    args = parser.parse_args()
    data = {}
    for split, directory in (("val", args.val_dir), ("test", args.test_dir)):
        samples, summary = read_sample_rows(directory)
        if summary["split"] != split:
            raise ValueError(f"Expected {split} in {directory}")
        for name in PHYSICS:
            if not any(math.isfinite(row[f"input_{name}_loss_a2"]) and
                       math.isfinite(row[f"refined_{name}_loss_a2"]) for row in samples):
                raise ValueError(f"{directory} is missing full physical loss results for {name}")
        data[split] = (samples, chain_rows(samples), summary)
    if data["val"][2]["checkpoint"] != data["test"][2]["checkpoint"]:
        raise ValueError("Validation and test results use different checkpoints")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, physics, distributions, transitions, histograms = [], [], [], [], []
    for split, (samples, chains, _) in data.items():
        for unit, rows in (("candidate", samples), ("PDB-chain macro", chains)):
            groups = {"all": rows, "input RMSD <2 Å": [r for r in rows if r["input_aligned_rmsd"] < 2]}
            for label in ("<=50 nt", "51-100 nt", "101-200 nt", ">200 nt"):
                groups[f"length {label}"] = [r for r in rows if length_bin(r["length"]) == label]
            for low, _ in RMSD_BINS:
                label = rmsd_bin(low)
                groups[f"input RMSD {label} Å"] = [r for r in rows if rmsd_bin(r["input_aligned_rmsd"]) == label]
            for group_name, group_rows in groups.items():
                metrics.append(metric_row(split, group_name, unit, group_rows, len(rows)))
                physics.extend(physics_rows(split, group_name, unit, group_rows))
            drows, trows = distribution_rows(split, unit, rows)
            distributions.extend(drows)
            transitions.extend(trows)
            histograms.extend(histogram_rows(split, unit, rows))

    metric_fields = list(metrics[0])
    physics_fields = list(physics[0])
    distribution_fields = list(distributions[0])
    transition_fields = list(transitions[0])
    histogram_fields = list(histograms[0])
    write_tsv(args.output_dir / "rmsd_comparison.tsv", metrics, metric_fields)
    write_tsv(args.output_dir / "physical_comparison.tsv", physics, physics_fields)
    write_tsv(args.output_dir / "rmsd_distribution.tsv", distributions, distribution_fields)
    write_tsv(args.output_dir / "rmsd_transitions.tsv", transitions, transition_fields)
    write_tsv(args.output_dir / "rmsd_histogram.tsv", histograms, histogram_fields)

    overlap = foldbench_overlap(data["test"][1], args.foldbench_targets) if args.foldbench_targets else []
    if args.foldbench_targets:
        write_tsv(
            args.output_dir / "foldbench_overlap.tsv", overlap,
            ["pdb_id", "native_chain_id", "foldbench_target_id", "sample_count",
             "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a"],
        )

    low_chains = []
    for split, (_, chains, _) in data.items():
        for row in chains:
            if row["input_aligned_rmsd"] < 2:
                low_chains.append({
                    "split": split,
                    "pdb_id": row["pdb_id"],
                    "native_chain_id": row["native_chain_id"],
                    "sample_count": row["sample_count"],
                    "input_rmsd_mean_a": row["input_aligned_rmsd"],
                    "refined_rmsd_mean_a": row["refined_aligned_rmsd"],
                    "improvement_mean_a": row["aligned_improvement"],
                    **{f"{phase}_{name}_loss_a2": row[f"{phase}_{name}_loss_a2"]
                       for name in PHYSICS for phase in ("input", "refined")},
                })
    low_chains.sort(key=lambda row: (row["split"], row["improvement_mean_a"]))
    write_tsv(args.output_dir / "low_input_rmsd_chains.tsv", low_chains, [
        "split", "pdb_id", "native_chain_id", "sample_count",
        "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a",
        *[f"{phase}_{name}_loss_a2" for name in PHYSICS for phase in ("input", "refined")],
    ])

    overall = [r for r in metrics if r["group"] == "all"]
    low = [r for r in metrics if r["group"] == "input RMSD <2 Å"]
    strata = [
        r for r in metrics if r["unit"] == "PDB-chain macro"
        and (r["group"].startswith("length ") or r["group"].startswith("input RMSD ["))
    ]
    physical_all = [r for r in physics if r["group"] == "all" and r["unit"] == "PDB-chain macro"]
    physical_low = [r for r in physics if r["group"] == "input RMSD <2 Å" and r["unit"] == "PDB-chain macro"]
    test_distribution = [r for r in distributions if r["split"] == "test"]
    test_histogram = [r for r in histograms if r["split"] == "test"]
    lines = [
        "# V3 RNA refinement evaluation", "",
        f"Checkpoint: `{data['val'][2]['checkpoint']}`", "",
        "主 RMSD 是 native observed atoms 的逐候选 Kabsch 对齐 RMSD，单位 Å；改善量 = 输入 − refinement 后。",
        "同一链的候选相关，链层面的宏平均是主要统计单位。分层按输入值固定，避免 refinement 后跨组导致口径改变。RMSD 仅用 native 可观测原子；物理 loss 则按训练定义使用预测结构的全部原子。", "",
        "## 验证集与测试集总览", "",
        markdown_table(overall, ["split", "unit", "count", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a", "win_rate", "worsen_rate"]), "",
        "## 输入 RMSD <2 Å", "",
        "比例分母分别为该 split 的全部候选或全部 PDB 链；链按候选的平均输入 RMSD 判定。", "",
        markdown_table(low, ["split", "unit", "count", "share", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a", "worsen_rate", "damage_ge_0.5_rate", "post_rmsd_ge_2_rate"]), "",
        "输入 <2 Å 且恶化最明显的 PDB 链（完整清单见 `low_input_rmsd_chains.tsv`）：", "",
        markdown_table([r for r in low_chains if r["split"] == "val"][:15] +
                       [r for r in low_chains if r["split"] == "test"][:15],
                       ["split", "pdb_id", "native_chain_id", "sample_count", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a"]), "",
        "## 原始长度与输入 RMSD 分层（PDB 链宏平均）", "",
        markdown_table(strata, ["split", "group", "count", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a", "win_rate"]), "",
        "## 测试集输入 RMSD 分布及去向", "",
        "前后使用相同分箱，下表直接比较各箱占比。", "",
        markdown_table(test_histogram, ["unit", "bin_a", "input_count", "refined_count", "input_share", "refined_share"]), "",
        "每行按 refinement 前的 RMSD 分箱，展示同一批结构之后的 RMSD；完整跨箱矩阵见 `rmsd_transitions.tsv`。", "",
        markdown_table(test_distribution, ["unit", "input_bin_a", "count", "share", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a", "win_rate"]), "",
        "## 物理结构指标（PDB 链宏平均）", "",
        "数值为训练代码中的未加权键长、位阻、碱基平面 loss，单位 Å²；越小越好。位阻 loss 是 cutoff 内非排除原子对的平均平方重叠，随邻域构成变化。它们不是完整的化学有效性检查。", "",
        markdown_table(physical_all, ["split", "loss", "count", "input_loss_a2", "refined_loss_a2", "change_after_minus_before_a2", "worsen_rate"]), "",
        "输入 RMSD <2 Å 链的物理 loss：", "",
        markdown_table(physical_low, ["split", "loss", "count", "input_loss_a2", "refined_loss_a2", "change_after_minus_before_a2", "worsen_rate"]), "",
    ]
    if args.foldbench_targets:
        lines += [
            "## FoldBench RNA 单体重叠", "",
            f"测试集匹配 {len(overlap)} 个 FoldBench monomer_rna 的 PDB+链。此处仍是本项目 RMSD，不是 FoldBench lDDT 分数。", "",
            markdown_table(overlap, ["foldbench_target_id", "sample_count", "input_rmsd_mean_a", "refined_rmsd_mean_a", "improvement_mean_a"]), "",
        ]
    lines += [
        "## 文件", "",
        "`rmsd_comparison.tsv` 包含总览、长度、输入 RMSD 和 <2 Å 子集；`low_input_rmsd_chains.tsv` 为低输入误差链清单；`physical_comparison.tsv` 包含对应候选/链级物理 loss；`rmsd_histogram.tsv`、`rmsd_distribution.tsv` 与 `rmsd_transitions.tsv` 记录前后分布。", "",
        "评估只覆盖 Data_PT_V2 中实际存在的 .pt 候选；FoldBench 分数需使用原始/精修结构文件由 FoldBench OpenStructure 流程另算，并按预先指定的排名分数选每靶标一个候选。", "",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"ANALYSIS_COMPLETE output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
