"""Plot epoch-49 mean RMSD improvement by input RMSD stratum."""

from __future__ import annotations

import csv
import math
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
    ("input RMSD [0,2) Å", "[0,2)"),
    ("input RMSD [2,5) Å", "[2,5)"),
    ("input RMSD [5,10) Å", "[5,10)"),
    ("input RMSD [10,20) Å", "[10,20)"),
    ("input RMSD [20,30) Å", "[20,30)"),
    ("input RMSD [30,inf) Å", "≥30"),
)


def main() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))

    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))

    for ax, split, title in zip(axes, ("val", "test"), ("验证集", "测试集")):
        rows = {
            row["group"]: row for row in records
            if row["split"] == split and row["unit"] == "PDB-chain macro"
        }
        if any(key not in rows for key, _ in GROUPS):
            raise ValueError(f"Missing input RMSD strata for {split}")
        counts = [int(rows[key]["count"]) for key, _ in GROUPS]
        values = [float(rows[key]["improvement_mean_a"]) for key, _ in GROUPS]
        before = [float(rows[key]["input_rmsd_mean_a"]) for key, _ in GROUPS]
        after = [float(rows[key]["refined_rmsd_mean_a"]) for key, _ in GROUPS]
        for count, value, source, refined in zip(counts, values, before, after):
            if count and (not math.isfinite(value) or abs((source - refined) - value) > 1e-5):
                raise ValueError(f"Inconsistent RMSD stratum in {split}")

        y = np.arange(len(GROUPS))
        colors = ["#2563eb" if value >= 0 else "#ef6c42" for value in values]
        safe_values = [value if count else 0 for value, count in zip(values, counts)]
        bars = ax.barh(y, safe_values, color=colors, height=0.56)
        for index, (bar, count, value) in enumerate(zip(bars, counts, values)):
            if not count:
                ax.text(0.02, index, "无样本", va="center", fontsize=9,
                        color="#64748b")
                continue
            shift = 0.025 if value >= 0 else -0.025
            ax.text(
                value + shift, index, f"{value:+.3f}",
                ha="left" if value >= 0 else "right", va="center",
                fontsize=10, color="#334155",
            )
        ax.axvline(0, color="#334155", linewidth=1)
        ax.set_yticks(y, [f"{label} Å  (n={count})"
                          for (_, label), count in zip(GROUPS, counts)])
        ax.invert_yaxis()
        ax.set_xlim(-0.60, 1.38)
        ax.set_xlabel("平均改善量：输入 − 精修后（Å）")
        ax.set_title(title, fontsize=14, weight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle("按输入 RMSD 分层：哪些起点得到修正｜epoch 49", fontsize=17,
                 weight="bold", y=0.99)
    fig.text(
        0.5, 0.015,
        "每条 PDB 链先对候选 RMSD 取平均；按精修前 RMSD 固定分箱，再对箱内的链等权平均。蓝色为改善，橙色为变差。",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("svg", {})):
        destination = OUTPUT / f"v3_input_rmsd_strata_epoch49.{suffix}"
        fig.savefig(destination, bbox_inches="tight", facecolor="white", **options)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
