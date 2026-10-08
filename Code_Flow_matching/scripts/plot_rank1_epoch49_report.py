"""Create presentation figures and a compact audit for the rank-1 epoch-49 evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "evaluation" / "v3_rank1_epoch49_20261008"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 220,
    }
)
COLORS = {"input": "#74859A", "refined": "#187A89", "negative": "#D8754C"}


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


samples = {
    "val": rows("val_final/samples.tsv"),
    "test": rows("test_locked/samples.tsv"),
}
hist = rows("analysis/rmsd_histogram.tsv")
comparison = rows("analysis/rmsd_comparison.tsv")
physical = rows("analysis/physical_comparison.tsv")

# 1. Distribution in identical, mutually exclusive RMSD bins.
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=False, layout="constrained")
for ax, split, label in zip(axes, ("val", "test"), ("验证集", "测试集")):
    data = [r for r in hist if r["split"] == split and r["unit"] == "candidate"]
    x = np.arange(len(data))
    width = 0.38
    before = [int(r["input_count"]) for r in data]
    after = [int(r["refined_count"]) for r in data]
    input_bars = ax.bar(x - width / 2, before, width, label="精修前", color=COLORS["input"])
    refined_bars = ax.bar(x + width / 2, after, width, label="精修后", color=COLORS["refined"])
    ax.set_xticks(x, ["<2", "2–5", "5–10", "10–20", "20–30", "≥30"])
    ax.set_xlabel("对齐 RMSD (Å)")
    ax.set_ylabel("RNA 链数")
    ax.set_title(f"{label} (n={len(samples[split])})")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(before + after) * 1.22)
    ax.bar_label(input_bars, labels=[str(n) for n in before], padding=3, fontsize=9)
    ax.bar_label(refined_bars, labels=[str(n) for n in after], padding=3, fontsize=9)
fig.suptitle("Rank-1 候选：精修前后 RMSD 分布", weight="bold")
save(fig, "rank1_rmsd_distribution_epoch49.png")

# 2. Length strata: paired mean before/after, with group counts.
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
groups = ["length <=50 nt", "length 51-100 nt", "length 101-200 nt", "length >200 nt"]
labels = ["≤50", "51–100", "101–200", ">200"]
for ax, split, title in zip(axes, ("val", "test"), ("验证集", "测试集")):
    data = {r["group"]: r for r in comparison if r["split"] == split and r["unit"] == "candidate"}
    x = np.arange(len(groups))
    width = 0.38
    before = [float(data[g]["input_rmsd_mean_a"]) for g in groups]
    after = [float(data[g]["refined_rmsd_mean_a"]) for g in groups]
    ns = [int(data[g]["count"]) for g in groups]
    input_bars = ax.bar(x - width / 2, before, width, label="精修前", color=COLORS["input"])
    refined_bars = ax.bar(x + width / 2, after, width, label="精修后", color=COLORS["refined"])
    ax.bar_label(input_bars, labels=[f"{value:.2f}" for value in before], padding=3, fontsize=9)
    ax.bar_label(refined_bars, labels=[f"{value:.2f}" for value in after], padding=3, fontsize=9)
    ax.set_xticks(x, [f"{label} nt\n(n={n})" for label, n in zip(labels, ns)])
    ax.set_xlabel("链长 (nt)")
    ax.set_ylabel("平均对齐 RMSD (Å)")
    ax.set_ylim(0, max(before + after) * 1.22)
    ax.set_title(title)
fig.suptitle("按链长分层：精修前后 RMSD", weight="bold")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.86, 0.83), frameon=False)
fig.subplots_adjust(left=0.08, right=0.84, bottom=0.18, top=0.79, wspace=0.32)
save(fig, "rank1_length_before_after_epoch49.png")

# 3. Match the original report's horizontal, diverging stratum chart.
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))
strata = (
    ("input RMSD [0,2) Å", "[0,2)"),
    ("input RMSD [2,5) Å", "[2,5)"),
    ("input RMSD [5,10) Å", "[5,10)"),
    ("input RMSD [10,20) Å", "[10,20)"),
    ("input RMSD [20,30) Å", "[20,30)"),
    ("input RMSD [30,inf) Å", "≥30"),
)
for ax, split, title in zip(axes, ("val", "test"), ("验证集", "测试集")):
    by_group = {
        row["group"]: row for row in comparison
        if row["split"] == split and row["unit"] == "PDB-chain macro"
    }
    if any(group not in by_group for group, _ in strata):
        raise ValueError(f"Missing input RMSD stratum for {split}")
    counts = [int(by_group[group]["count"]) for group, _ in strata]
    values = [float(by_group[group]["improvement_mean_a"]) for group, _ in strata]
    for (group, _), count, value in zip(strata, counts, values):
        if count:
            row = by_group[group]
            before = float(row["input_rmsd_mean_a"])
            after = float(row["refined_rmsd_mean_a"])
            if not math.isfinite(value) or abs(before - after - value) > 1e-5:
                raise ValueError(f"Inconsistent input RMSD stratum: {split} {group}")
    y = np.arange(len(strata))
    safe_values = [value if count else 0 for value, count in zip(values, counts)]
    colors = [COLORS["refined"] if value >= 0 else COLORS["negative"] for value in safe_values]
    bars = ax.barh(y, safe_values, color=colors, height=0.56)
    for index, (count, value) in enumerate(zip(counts, values)):
        if not count:
            ax.text(0.02, index, "无样本", va="center", fontsize=9, color="#64748b")
            continue
        shift = 0.025 if value >= 0 else -0.025
        ax.text(
            value + shift, index, f"{value:+.3f}",
            ha="left" if value >= 0 else "right", va="center",
            fontsize=10, color="#334155",
        )
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_yticks(y, [f"{label} Å  (n={count})" for (_, label), count in zip(strata, counts)])
    ax.invert_yaxis()
    ax.set_xlim(-0.60, 1.38)
    ax.set_xlabel("平均改善量：输入 − 精修后（Å）")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle("按输入 RMSD 分层：哪些起点得到修正｜epoch 49（rank-1）", fontsize=17, weight="bold", y=0.99)
fig.text(
    0.5, 0.015,
    "每条 PDB 链仅 1 个候选；按精修前 RMSD 固定分箱，对箱内链等权平均。蓝绿色为改善，橙色为变差。",
    ha="center", fontsize=10, color="#475569",
)
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
save(fig, "rank1_input_rmsd_strata_epoch49.png")

# 4. High quality input subset. Right panel is conditional on input RMSD < 2 Å.
low = {split: [r for r in group if float(r["input_aligned_rmsd"]) < 2] for split, group in samples.items()}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
x = np.arange(2)
before = [np.mean([float(r["input_aligned_rmsd"]) for r in low[s]]) for s in ("val", "test")]
after = [np.mean([float(r["refined_aligned_rmsd"]) for r in low[s]]) for s in ("val", "test")]
axes[0].bar(x - 0.19, before, 0.38, label="精修前", color=COLORS["input"])
axes[0].bar(x + 0.19, after, 0.38, label="精修后", color=COLORS["refined"])
axes[0].set_xticks(x, [f"验证集\n{len(low['val'])}/103", f"测试集\n{len(low['test'])}/67"])
axes[0].set_ylabel("该子集平均 RMSD (Å)")
axes[0].set_title("高质量输入的平均 RMSD")
axes[0].legend(frameon=False)
axes[0].set_ylim(0, 2.2)
for i, (a, b) in enumerate(zip(before, after)):
    axes[0].text(i, max(a, b) + 0.08, f"{a:.3f}→{b:.3f}", ha="center", fontsize=9)
worsen = [sum(float(r["refined_aligned_rmsd"]) > float(r["input_aligned_rmsd"]) for r in low[s]) for s in ("val", "test")]
cross = [sum(float(r["refined_aligned_rmsd"]) >= 2 for r in low[s]) for s in ("val", "test")]
n = [len(low[s]) for s in ("val", "test")]
axes[1].bar(x - 0.19, [100 * a / b for a, b in zip(worsen, n)], 0.38, label="RMSD 变差", color=COLORS["negative"])
axes[1].bar(x + 0.19, [100 * a / b for a, b in zip(cross, n)], 0.38, label="跨到 ≥2 Å", color="#936BB0")
axes[1].set_xticks(x, ["验证集", "测试集"])
axes[1].set_ylabel("占输入 <2 Å 子集比例 (%)")
axes[1].set_ylim(0, 90)
axes[1].set_title("右图分母：各自输入 <2 Å 的链数")
axes[1].legend(frameon=False)
for i in range(2):
    axes[1].text(i - 0.19, 100 * worsen[i] / n[i] + 2, f"{worsen[i]}/{n[i]}", ha="center", fontsize=9)
    axes[1].text(i + 0.19, 100 * cross[i] / n[i] + 2, f"{cross[i]}/{n[i]}", ha="center", fontsize=9)
fig.suptitle("输入 RMSD <2 Å：单独检查高质量输入", weight="bold")
save(fig, "rank1_low_rmsd_epoch49.png")

# Audit values that are not present in the precomputed analysis tables.
audit = {}
for split, group in samples.items():
    sorted_group = sorted(group, key=lambda r: float(r["aligned_improvement"]), reverse=True)
    improvement = np.array([float(r["aligned_improvement"]) for r in group])
    audit[split] = {
        "count": len(group),
        "improve_ge_0.1": int((improvement >= 0.1).sum()),
        "improve_ge_0.5": int((improvement >= 0.5).sum()),
        "worsen_ge_0.1": int((improvement <= -0.1).sum()),
        "worsen_ge_0.5": int((improvement <= -0.5).sum()),
        "top5_improvement_share": float(sum(float(r["aligned_improvement"]) for r in sorted_group[:5]) / improvement.sum()),
        "remaining_mean_without_top5": float(np.mean([float(r["aligned_improvement"]) for r in sorted_group[5:]])),
        "top5": [
            {k: r[k] for k in ("pdb_id", "native_chain_id", "length", "input_aligned_rmsd", "refined_aligned_rmsd", "aligned_improvement")}
            for r in sorted_group[:5]
        ],
        "bottom3": [
            {k: r[k] for k in ("pdb_id", "native_chain_id", "length", "input_aligned_rmsd", "refined_aligned_rmsd", "aligned_improvement")}
            for r in sorted_group[-3:]
        ],
    }
print(json.dumps(audit, ensure_ascii=False, indent=2))
