#!/usr/bin/env python3
"""Export three full-data weighted ECDF panels for the dataset-shift PPT."""
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
    out = root / "ppt_panels"
    metadata = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if any("differ in length" in w or "No sampler length" in w for w in metadata["warnings"]):
        raise ValueError("Reconcile PT/sampler lengths before reproducing exposure weights")
    columns = ["pool", "split", "pdb_id", "seed", "sample", "length", "rmsd", "plddt_residue_mean", "ranking_score"]
    data = pd.read_csv(root / "features.tsv", sep="\t", usecols=columns)
    data = data[data.pool == "current"].copy()
    if data[columns].isna().any().any():
        raise ValueError("Incomplete current sample features")
    train = data[data.split == "train"].copy()
    train["weight"] = np.where(train.length > metadata["long_threshold"], metadata["long_factor"], 1.)
    rank1 = train.sort_values(["ranking_score", "seed", "sample"], ascending=[False, True, True]).drop_duplicates("pdb_id").copy()
    rank1["weight"] = 1.
    views = {"train_effective": train, "train_rank1": rank1}
    for split in ("val", "test"):
        frame = data[data.split == split].copy()
        if frame.pdb_id.duplicated().any():
            raise ValueError(f"{split} has more than one selected candidate per PDB")
        frame["weight"] = 1.
        views[f"{split}_rank1"] = frame
    colors = {"train_effective": "#1f77b4", "train_rank1": "#2ca02c", "val_rank1": "#d62728", "test_rank1": "#9467bd"}
    panels = [("length", "Length", "RNA length (nt)", "upper left"),
              ("rmsd", "RMSD", "Input RMSD (Å)", "lower right"),
              ("plddt_residue_mean", "pLDDT", "Mean residue pLDDT (0–1)", "upper left")]
    reference = pd.read_csv(root / "distribution_summary.tsv", sep="\t")
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17,
                         "axes.titlesize": 19, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.linewidth": 1.1, "savefig.facecolor": "white"})
    for metric, title, xlabel, location in panels:
        fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
        for name, frame in views.items():
            # Check both sample coverage and weighted mean against the original report.
            expected = reference[(reference.view == name) & (reference.metric == metric)].iloc[0]
            assert len(frame) == int(expected.candidates), (name, metric, "candidate count")
            assert np.isclose(np.average(frame[metric], weights=frame.weight), expected["mean"], rtol=1e-7, atol=1e-7), (name, metric, "weighted mean")
            mass = frame.groupby(metric, sort=True).weight.sum()
            x = mass.index.to_numpy()
            y = np.cumsum(mass.to_numpy()) / mass.sum()
            assert np.isclose(y[-1], 1.) and np.all(np.diff(y) >= 0)
            ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=colors[name], linewidth=2.4, label=name)
        if metric == "length":
            ax.set_xscale("symlog", linthresh=1)
        elif metric == "rmsd":
            ax.set_xlim(0, data.rmsd.max() * 1.04)
        else:
            ax.set_xlim(0, 1)
        ax.set(title=title, xlabel=xlabel, ylabel="Weighted ECDF", ylim=(-0.015, 1.025))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#e6e6e6", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(loc=location, fontsize=12.5, framealpha=.95)
        filename = "plddt.png" if metric == "plddt_residue_mean" else f"{metric}.png"
        fig.savefig(out / filename, dpi=300)
        plt.close(fig)
        print(f"SAVED {out / filename}")
    print("VERIFIED: four curves only; full candidate counts and weighted means match distribution_summary.tsv.")


if __name__ == "__main__":
    main()
