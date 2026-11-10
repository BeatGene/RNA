#!/usr/bin/env python3
"""Export the split-colored input PCA panel using existing saved coordinates."""
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
    data = pd.read_csv(root / "embedding_input.tsv", sep="\t")
    config = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))["input"]
    if data.key.duplicated().any() or not np.isfinite(data[["pc1", "pc2"]]).all().all():
        raise ValueError("Invalid saved PCA coordinates")
    train = data[data.split == "train"]
    if len(train) != config["fit_candidates"] or train.pdb_id.nunique() != config["fit_pdbs"]:
        raise ValueError("Training point counts disagree with saved PCA configuration")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17,
                         "axes.titlesize": 19, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for split, color in (("train", "#4477aa"), ("val", "#228833"), ("test", "#cc6677")):
        f = data[data.split == split]
        ax.scatter(f.pc1, f.pc2, label=split, c=color, s=12 if split == "train" else 35,
                   alpha=.25 if split == "train" else .9, edgecolors="none")
    variance = config["explained_variance_ratio"]
    ax.set(title="Input feature PCA", xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})")
    ax.legend(loc="upper left", fontsize=14)
    out = root / "ppt_panels" / "pca_input_split.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(out)
    print("SAVED: original coordinates and all saved points; PCA was not refitted.")


if __name__ == "__main__":
    main()
