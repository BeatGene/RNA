"""Plot before/after RMSD bins for the selected V3 checkpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
RESULT = EVALUATION / "v3_20261008"
OUTPUT = EVALUATION / "figures"
BINS = ("[0,2)", "[2,5)", "[5,10)", "[10,20)", "[20,30)", "[30,inf)")
BIN_LABELS = ("0–2", "2–5", "5–10", "10–20", "20–30", "≥30")


def main() -> None:
    with (RESULT / "analysis" / "rmsd_histogram.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))

    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    colors = ("#64748b", "#2563eb")

    for ax, split, folder, title in zip(
        axes,
        ("val", "test"),
        ("val_final", "test_locked"),
        ("验证集", "测试集"),
    ):
        summary = json.loads((RESULT / folder / "summary.json").read_text(encoding="utf-8"))
        rows = {
            row["bin_a"]: row for row in records
            if row["split"] == split and row["unit"] == "PDB-chain macro"
        }
        if set(rows) != set(BINS):
            raise ValueError(f"RMSD bins are incomplete for {split}: {sorted(rows)}")
        total = summary["pdb_chain_count"]
        before_count = [int(rows[key]["input_count"]) for key in BINS]
        after_count = [int(rows[key]["refined_count"]) for key in BINS]
        if sum(before_count) != total or sum(after_count) != total:
            raise ValueError(f"RMSD bin counts do not match {split} PDB-chain count")

        x = np.arange(len(BINS))
        width = 0.35
        for offset, counts, label, color in (
            (-width / 2, before_count, "输入", colors[0]),
            (+width / 2, after_count, "精修后", colors[1]),
        ):
            bars = ax.bar(x + offset, np.array(counts) * 100 / total, width,
                          label=label, color=color)
            for bar, count in zip(bars, counts):
                if count:
                    ax.annotate(
                        str(count),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="#334155",
                    )
        ax.set_xticks(x, BIN_LABELS)
        ax.set_ylim(0, 48)
        ax.set_xlabel("RMSD 分箱（Å）")
        ax.set_title(f"{title}（{total} 条 RNA 链）", fontsize=14, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("PDB 链占比（%）")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("精修前后 RMSD 分布｜epoch 49", fontsize=17, weight="bold", y=0.99)
    fig.text(
        0.5, 0.015,
        "统计口径：每条 PDB 链先对候选 RMSD 取平均，再按该链的均值分箱；柱顶数字为链数。",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("svg", {})):
        destination = OUTPUT / f"v3_rmsd_distribution_epoch49.{suffix}"
        fig.savefig(destination, bbox_inches="tight", facecolor="white", **options)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
