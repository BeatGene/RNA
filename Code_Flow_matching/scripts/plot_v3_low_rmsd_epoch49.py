"""Plot epoch-49 outcomes for candidates with input RMSD below 2 Angstrom."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
SOURCE = EVALUATION / "v3_20261008" / "analysis" / "rmsd_comparison.tsv"
OUTPUT = EVALUATION / "figures"


def main() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    selected = {}
    for split in ("val", "test"):
        row = next(
            row for row in records
            if row["split"] == split and row["unit"] == "candidate"
            and row["group"] == "input RMSD <2 Å"
        )
        selected[split] = row

    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    labels = ("验证集", "测试集")
    inputs = [float(selected[s]["input_rmsd_mean_a"]) for s in ("val", "test")]
    refined = [float(selected[s]["refined_rmsd_mean_a"]) for s in ("val", "test")]
    counts = [int(selected[s]["count"]) for s in ("val", "test")]
    crossing_rates = [float(selected[s]["post_rmsd_ge_2_rate"]) for s in ("val", "test")]
    crossing_counts = [round(n * rate) for n, rate in zip(counts, crossing_rates)]

    x = np.arange(2)
    width = 0.34
    before_bars = axes[0].bar(x - width / 2, inputs, width,
                              color="#64748b", label="输入")
    after_bars = axes[0].bar(x + width / 2, refined, width,
                             color="#2563eb", label="精修后")
    for bars, values in ((before_bars, inputs), (after_bars, refined)):
        for bar, value in zip(bars, values):
            axes[0].annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, color="#334155",
            )
    axes[0].set_xticks(x, [f"{name}\n(n={n:,})" for name, n in zip(labels, counts)])
    axes[0].set_ylim(0, 2.35)
    axes[0].set_ylabel("该子集平均 RMSD（Å）")
    axes[0].set_title("平均 RMSD：输入与精修后", fontsize=14, weight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    crossing_bars = axes[1].bar(x, np.array(crossing_rates) * 100,
                                width=0.48, color="#ef6c42")
    for bar, n, crossed, rate in zip(crossing_bars, counts, crossing_counts, crossing_rates):
        axes[1].annotate(
            f"{crossed:,}/{n:,}\n{rate * 100:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color="#334155",
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 43)
    axes[1].set_ylabel("从 <2 Å 变为 ≥2 Å 的比例（%）")
    axes[1].set_title("跨出高质量区间的比例", fontsize=14, weight="bold")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("输入 RMSD <2 Å 的候选：精修后发生了什么｜epoch 49",
                 fontsize=17, weight="bold", y=0.99)
    fig.text(
        0.5, 0.015,
        "两图均只统计输入 RMSD <2 Å 的候选；右图分母分别为验证集 2,630 个、测试集 780 个候选。",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("svg", {})):
        destination = OUTPUT / f"v3_low_rmsd_epoch49.{suffix}"
        fig.savefig(destination, bbox_inches="tight", facecolor="white", **options)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
