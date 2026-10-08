"""Plot PDB-chain RMSD before and after V3 epoch-49 refinement by RNA length."""

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
GROUPS = (
    ("length <=50 nt", "≤50 nt"),
    ("length 51-100 nt", "51–100 nt"),
    ("length 101-200 nt", "101–200 nt"),
    ("length >200 nt", ">200 nt"),
)


def main() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))

    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    colors = ("#64748b", "#2563eb")

    for ax, split, title in zip(axes, ("val", "test"), ("验证集", "测试集")):
        rows = {
            row["group"]: row for row in records
            if row["split"] == split and row["unit"] == "PDB-chain macro"
        }
        if any(key not in rows for key, _ in GROUPS):
            raise ValueError(f"Missing length strata in {SOURCE}: {split}")
        x = np.arange(len(GROUPS))
        width = 0.35
        before = [float(rows[key]["input_rmsd_mean_a"]) for key, _ in GROUPS]
        after = [float(rows[key]["refined_rmsd_mean_a"]) for key, _ in GROUPS]
        bars_before = ax.bar(x - width / 2, before, width, label="输入", color=colors[0])
        bars_after = ax.bar(x + width / 2, after, width, label="精修后", color=colors[1])

        for bars, values in ((bars_before, before), (bars_after, after)):
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color="#334155",
                )
        labels = [f"{label}\n(n={rows[key]['count']})" for key, label in GROUPS]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 23)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("链级平均 RMSD（Å）")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("按链长分层：精修前后 RMSD｜epoch 49", fontsize=17, weight="bold", y=0.99)
    fig.text(
        0.5, 0.015,
        "每条 PDB 链先对其候选取平均，再对同一长度组的链等权平均；数值越低越好。",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("svg", {})):
        destination = OUTPUT / f"v3_length_before_after_epoch49.{suffix}"
        fig.savefig(destination, bbox_inches="tight", facecolor="white", **options)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
