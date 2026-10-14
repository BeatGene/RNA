"""Summarize fixed-checkpoint loss gradient competition from the server probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_rank1_pymol_diagnostics import read_tsv, write_tsv


LOSSES = ("bond_loss", "clash_loss", "plane_loss", "mobility_loss",
          "no_regret_loss", "protect_loss", "velocity_budget_loss")


def summarize(root):
    rows = read_tsv(root / "loss_gradients.tsv")
    by_pdb = {}
    for row in rows:
        by_pdb.setdefault(row["pdb_id"], {})[row["loss"]] = row
    normalized, summary = [], []
    for pdb, losses in by_pdb.items():
        if set(losses) != {"flow_matching_loss", *LOSSES}:
            raise ValueError(f"missing gradient terms: {pdb}")
        baseline = float(losses["flow_matching_loss"]["weighted_parameter_gradient_norm"])
        raw_baseline = float(losses["flow_matching_loss"]["weighted_raw_output_gradient_norm"])
        if baseline <= 0 or raw_baseline <= 0:
            raise ValueError(f"zero baseline gradient: {pdb}")
        pieces = {}
        for name in LOSSES:
            row = losses[name]
            ratio = float(row["weighted_parameter_gradient_norm"]) / baseline
            cosine = float(row["parameter_gradient_cosine_vs_coordinate"]) if row["parameter_gradient_cosine_vs_coordinate"] else 0
            cosine = float(np.clip(cosine, -1, 1))
            projection = ratio * cosine
            pieces[name] = ratio, cosine, projection
            normalized.append({
                "pdb_id": pdb, "loss": name,
                "weighted_parameter_norm_ratio_to_coordinate": ratio,
                "parameter_cosine_vs_coordinate": cosine if ratio > 0 else "",
                "parameter_projection_ratio_to_coordinate": projection,
                "weighted_raw_output_norm_ratio_to_coordinate": float(row["weighted_raw_output_gradient_norm"]) / raw_baseline,
                "weighted_loss_value": row["weighted_value"],
            })
        summary.append({
            "pdb_id": pdb, "coordinate_parameter_norm": baseline,
            "no_regret_norm_ratio": pieces["no_regret_loss"][0],
            "no_regret_cosine": pieces["no_regret_loss"][1],
            "no_regret_projection_ratio": pieces["no_regret_loss"][2],
            "protect_norm_ratio": pieces["protect_loss"][0],
            "protect_cosine": pieces["protect_loss"][1] if pieces["protect_loss"][0] > 0 else "",
            "budget_norm_ratio": pieces["velocity_budget_loss"][0],
            "geometry_sum_of_norms_ratio_upper_bound": sum(pieces[name][0] for name in ("bond_loss", "clash_loss", "plane_loss")),
            "all_losses_parallel_component_vs_coordinate": 1 + sum(item[2] for item in pieces.values()),
            "coordinate_plus_no_regret_norm_ratio": float(np.sqrt(max(0, 1 + pieces["no_regret_loss"][0] ** 2 + 2 * pieces["no_regret_loss"][2]))),
        })
    write_tsv(root / "normalized_gradients.tsv", normalized)
    write_tsv(root / "gradient_summary.tsv", summary)
    plot(root, by_pdb, normalized)
    for row in summary:
        print(row)


def plot(root, by_pdb, normalized):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["bond", "clash", "plane", "mobility", "no-regret", "protect", "budget"]
    pdbs = list(by_pdb)
    indexed = {(row["pdb_id"], row["loss"]): row for row in normalized}
    ratios = np.array([[indexed[pdb, name]["weighted_parameter_norm_ratio_to_coordinate"] for name in LOSSES] for pdb in pdbs])
    cosines = np.array([[indexed[pdb, name]["parameter_cosine_vs_coordinate"] if ratios[i, j] > 0 else np.nan
                         for j, name in enumerate(LOSSES)] for i, pdb in enumerate(pdbs)], float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    first = axes[0].imshow(np.log10(np.maximum(ratios, 1e-6)), cmap="YlOrRd", vmin=-6, vmax=0, aspect="auto")
    second = axes[1].imshow(cosines, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for k, axis in enumerate(axes):
        axis.set_xticks(range(len(LOSSES)), labels, rotation=40, ha="right")
        axis.set_yticks(range(len(pdbs)), pdbs)
        for i in range(len(pdbs)):
            for j in range(len(LOSSES)):
                value = ratios[i, j] if k == 0 else cosines[i, j]
                text = (f"{value:.2f}" if value >= 0.01 else f"{value:.1e}") if k == 0 else (f"{value:+.2f}" if np.isfinite(value) else "--")
                axis.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    axes[0].set_title("Weighted parameter gradient norm / coordinate norm")
    axes[1].set_title("Gradient cosine vs coordinate: red = conflict")
    fig.colorbar(first, ax=axes[0], label="log10 norm ratio")
    fig.colorbar(second, ax=axes[1], label="cosine")
    fig.suptitle("Epoch 49: five diagnostic targets, BF16 (checkpoint-local gradients)")
    fig.tight_layout()
    fig.savefig(root / "gradient_competition.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=Path(__file__).resolve().parents[1] /
                        "evaluation/rank1_mobility_probe_20261013_104047")
    summarize(parser.parse_args().probe_dir)
