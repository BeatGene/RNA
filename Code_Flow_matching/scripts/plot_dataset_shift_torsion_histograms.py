#!/usr/bin/env python3
"""Redraw saved 30-degree input torsion histograms as piecewise-constant stairs."""
from __future__ import annotations

import argparse
import json
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
    metadata = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if metadata["extraction_errors"] or metadata["missing_scores"]:
        raise ValueError("Resolve extraction/score errors before reproducing plots")
    if any("differ in length" in w or "No sampler length" in w for w in metadata["warnings"]):
        raise ValueError("Reconcile sampler lengths first")
    angles = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "eta", "theta"]
    hist_columns = [f"pred_{a}_hist_{i:02d}" for a in angles for i in range(12)]
    data = pd.read_csv(root / "features.tsv", sep="\t",
                       usecols=["pool", "split", "pdb_id", "seed", "sample", "ranking_score", "length"] + hist_columns,
                       dtype={c: float for c in hist_columns})
    data = data[data.pool == "current"]
    train = data[data.split == "train"].copy()
    train["weight"] = np.where(train.length > metadata["long_threshold"], metadata["long_factor"], 1.)
    pdb_equal = train.copy()
    pdb_equal["weight"] = 1 / pdb_equal.groupby("pdb_id").pdb_id.transform("size")
    rank1 = train.sort_values(["ranking_score", "seed", "sample"], ascending=[False, True, True]).drop_duplicates("pdb_id").copy()
    rank1["weight"] = 1.
    views = {"train_effective": train, "train_pdb_equal": pdb_equal, "train_rank1": rank1}
    for split in ("val", "test"):
        f = data[data.split == split].copy()
        if f.pdb_id.duplicated().any():
            raise ValueError(f"{split} is not selected rank1")
        f["weight"] = 1.
        views[f"{split}_rank1"] = f
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    edges = np.linspace(-180, 180, 13)
    # Both values belong to exactly the same [-90, -60) interval.
    assert np.searchsorted(edges, -70, side="right") == np.searchsorted(edges, -75, side="right")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14})
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.7))
    for ax, angle in zip(axes.flat, angles):
        columns = [f"pred_{angle}_hist_{i:02d}" for i in range(12)]
        for (name, frame), color in zip(views.items(), colors):
            values = frame[columns].to_numpy()
            valid = np.isfinite(values).all(axis=1)
            if not valid.any():
                continue
            heights = np.average(values[valid], axis=0, weights=frame.weight.to_numpy()[valid])
            assert np.isclose(heights.sum(), 1., atol=1e-7) and (heights >= 0).all()
            ax.stairs(heights, edges, baseline=None, label=name, color=color, linewidth=1.7)
        ax.set(title=angle, xlabel="Angle (°)", ylabel="Mean within-PT bin fraction", xlim=(-180, 180))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_ylim(bottom=-.015)
        ax.grid(axis="y", color="#e8e8e8", linewidth=.6)
        ax.set_axisbelow(True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Input torsion distributions: 30° bins, constant height within each bin", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .965))
    out = root / "ppt_panels" / "torsions_step.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(out)
    print("VERIFIED: every histogram sums to 1; -70° and -75° use the same bin height.")


if __name__ == "__main__":
    main()
