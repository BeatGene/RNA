#!/usr/bin/env python3
"""Plot per-PT mean absolute circular torsion errors from the shift summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    source = pd.read_csv(root / "distribution_summary.tsv", sep="\t")
    angles = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "eta", "theta"]
    labels = ["α", "β", "γ", "δ", "ε", "ζ", "χ", "η", "θ"]
    views = ["train_effective", "train_rank1", "val_rank1", "test_rank1"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    metrics = [f"{a}_error_mean_deg" for a in angles]
    selected = source[source.view.isin(views) & source.metric.isin(metrics)]
    if selected.duplicated(["view", "metric"]).any() or len(selected) != len(views) * len(angles):
        raise ValueError("Missing or duplicate torsion summary rows")
    values = selected.pivot(index="metric", columns="view", values="mean").reindex(index=metrics, columns=views)
    if not np.isfinite(values.to_numpy()).all() or ((values < 0) | (values > 180)).any().any():
        raise ValueError("Invalid circular angle error means")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 19,
                         "axes.labelsize": 16, "xtick.labelsize": 18, "ytick.labelsize": 13})
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x, width = np.arange(len(angles)), .19
    for i, (view, color) in enumerate(zip(views, colors)):
        ax.bar(x + (i - 1.5) * width, values[view], width=width, color=color, label=view)
    ax.set_xticks(x, labels)
    ax.set(xlabel="Torsion type", ylabel="Mean absolute circular error (°)",
           title="Input torsion errors relative to native", ylim=(0, values.to_numpy().max() * 1.24))
    ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "Each PT is averaged over valid paired angles first; train_effective then uses sampling weights.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, 1))
    out = root / "ppt_panels" / "torsion_native_error.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(out)
    print(values.round(4).to_string())


if __name__ == "__main__":
    main()
