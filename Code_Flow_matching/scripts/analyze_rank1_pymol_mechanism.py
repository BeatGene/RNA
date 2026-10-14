"""Diagnose movement, direction and chemical distortion in a local triplet bundle.

Reads completed exports only; does not train or select a model. Residual scale
curves and residue rigid projections are descriptive interventions on fixed
predictions, not a validation-selected deployment procedure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_rank1_pymol_diagnostics import (
    BASE_RING_NAMES, fit_coordinates, read_cif, read_tsv, write_json, write_tsv,
)


def f(row, key):
    return float(row[key])


def masked_rms_by_residue(error, residues, observed, count):
    squares = np.bincount(residues[observed], weights=(error[observed] ** 2).sum(axis=1), minlength=count)
    counts = np.bincount(residues[observed], minlength=count)
    valid = counts > 0
    return np.sqrt(squares / np.maximum(counts, 1)), valid


def coordinate_geometry(positions, bonds, ideals, rings):
    distances = np.linalg.norm(positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=1)
    bond_loss = float(np.square(distances - ideals).mean())
    planes = []
    for indices in rings:
        if len(indices) >= 4:
            points = positions[indices]
            singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            planes.append(float(singular[-1] ** 2 / len(points)))
    return bond_loss, float(np.mean(planes)) if planes else 0.0


def target_analysis(bundle, row, scales):
    root = bundle / row["split"] / row["pdb_id"]
    atoms = read_tsv(root / "atom_metrics.tsv")
    residues_table = read_tsv(root / "residue_metrics.tsv")
    bonds_table = read_tsv(root / "bond_metrics.tsv")
    _, _, source = read_cif(root / "source_frame/before.cif")
    _, _, refined = read_cif(root / "source_frame/after.cif")
    _, _, native = read_cif(root / "source_frame/native.cif")
    observed = np.array([int(atom["native_observed"]) for atom in atoms], bool)
    residue_index = np.array([int(atom["residue"]) - 1 for atom in atoms])
    names = np.array([atom["atom_name"] for atom in atoms])
    count = int(row["length"])
    if len(atoms) != len(source) or len(native) != observed.sum():
        raise ValueError(f"atom mapping count mismatch: {root}")
    target = source.copy()
    target[observed] = native
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    refined_c = refined - refined.mean(axis=0)
    residual = refined_c - source_c
    target_displacement = target_c - source_c
    movement = np.linalg.norm(residual, axis=1)
    # Rounding can add a tiny centroid error; the predictor itself is centered.
    before_fitted, pre_rmsd, _ = fit_coordinates(source, target, observed)
    after_fitted, post_rmsd, _ = fit_coordinates(refined, target, observed)
    if abs(pre_rmsd - f(row, "input_aligned_rmsd")) > 0.001 or abs(post_rmsd - f(row, "refined_aligned_rmsd")) > 0.001:
        raise ValueError(f"CIF metrics disagree with manifest: {root}")
    if abs(float(np.sqrt(np.square(target_displacement[observed]).sum(axis=1).mean())) - f(row, "input_frame_rmsd")) > 0.001:
        raise ValueError(f"reconstructed training target frame disagrees: {root}")
    bonds = np.array([[int(b["atom_id_1"]) - 1, int(b["atom_id_2"]) - 1] for b in bonds_table])
    ideals = np.array([f(b, "ideal_length_a") for b in bonds_table])
    groups = [np.flatnonzero(residue_index == r) for r in range(count)]
    rings = [indices[np.isin(names[indices], BASE_RING_NAMES)] for indices in groups]
    # Project each source residue onto the predicted residue by a proper rigid fit.
    # The transformation is determined by source and prediction only.
    projected = refined.copy()
    rigid_error = np.zeros(count)
    for r, indices in enumerate(groups):
        rigid, error, _ = fit_coordinates(source[indices], refined[indices], np.ones(len(indices), bool))
        projected[indices] = rigid
        rigid_error[r] = error
    _, projected_rmsd, _ = fit_coordinates(projected, target, observed)
    projected_bond, projected_plane = coordinate_geometry(projected, bonds, ideals, rings)
    source_errors, valid = masked_rms_by_residue(source_c - target_c, residue_index, observed, count)
    refined_errors, _ = masked_rms_by_residue(refined_c - target_c, residue_index, observed, count)
    gate_targets = np.clip((source_errors - 0.3) / 1.2, 0, 1)
    protected = valid & (source_errors < 0.5)
    protected_atoms = observed & protected[residue_index]
    no_regret = float(np.maximum(refined_errors[valid] - source_errors[valid], 0).mean())
    protect_loss = float(np.square(residual[protected_atoms]).sum(axis=1).mean()) if protected_atoms.any() else 0.0
    u, d = target_displacement[observed], residual[observed]
    dot = float((u * d).sum())
    norm = float(np.sqrt(np.square(u).sum() * np.square(d).sum()))
    cos = dot / norm if norm > 1e-12 else 0.0
    denominator = float(np.square(d).sum())
    quadratic_alpha = dot / denominator if denominator > 1e-12 else 0.0
    relative_bond_motion = np.linalg.norm(residual[bonds[:, 0]] - residual[bonds[:, 1]], axis=1)
    aligned_move = np.linalg.norm(after_fitted - before_fitted, axis=1)
    pre_bond, post_bond = np.array([f(b, "input_abs_error_a") for b in bonds_table]), np.array([f(b, "refined_abs_error_a") for b in bonds_table])
    result = {"split": row["split"], "pdb_id": row["pdb_id"], "length_nt": count,
              "input_rmsd_a": pre_rmsd, "refined_rmsd_a": post_rmsd, "improvement_a": pre_rmsd - post_rmsd,
              "movement_rms_a": float(np.sqrt(np.square(movement).mean())),
              "movement_mean_a": float(movement.mean()), "movement_median_a": float(np.median(movement)),
              "movement_p95_a": float(np.quantile(movement, 0.95)), "movement_max_a": float(movement.max()),
              "atom_fraction_move_lt_0.01": float((movement < 0.01).mean()),
              "atom_fraction_move_lt_0.1": float((movement < 0.1).mean()),
              "atom_fraction_move_gt_1": float((movement > 1).mean()),
              "atom_fraction_move_gt_3": float((movement > 3).mean()),
              "displacement_target_cosine": cos, "quadratic_optimal_alpha_fixed_frame": quadratic_alpha,
              "bond_relative_motion_rms_a": float(np.sqrt(np.square(relative_bond_motion).mean())),
              "residue_nonrigid_rms_a": float(np.sqrt(np.average(np.square(rigid_error), weights=[len(g) for g in groups]))),
              "residue_nonrigid_median_a": float(np.median(rigid_error)),
              "input_bond_rmse_a": float(np.sqrt(f(row, "input_bond_loss_a2"))),
              "refined_bond_rmse_a": float(np.sqrt(f(row, "refined_bond_loss_a2"))),
              "bond_fraction_error_gt_0.1_before": float((pre_bond > 0.1).mean()),
              "bond_fraction_error_gt_0.1_after": float((post_bond > 0.1).mean()),
              "bond_fraction_error_gt_0.3_after": float((post_bond > 0.3).mean()),
              "bond_fraction_error_gt_0.5_after": float((post_bond > 0.5).mean()),
              "input_bond_max_error_a": float(pre_bond.max()), "refined_bond_max_error_a": float(post_bond.max()),
              "expected_gate_target_mean": float(gate_targets[valid].mean()),
              "residue_fraction_target_gate_eq_1": float((gate_targets[valid] == 1).mean()),
              "protected_residue_fraction": float(protected.sum() / valid.sum()),
              "no_regret_loss_a": no_regret, "protect_loss_a2": protect_loss,
              "coordinate_l2_before_a": float(np.linalg.norm(u, axis=1).mean()),
              "coordinate_l2_after_a": float(np.linalg.norm(d - u, axis=1).mean()),
              "geometry_weighted_before": sum(weight * f(row, "input_" + name + "_loss_a2") for weight, name in ((0.1, "bond"), (0.01, "clash"), (0.1, "plane"))),
              "geometry_weighted_after": sum(weight * f(row, "refined_" + name + "_loss_a2") for weight, name in ((0.1, "bond"), (0.01, "clash"), (0.1, "plane"))),
              "mean_plddt": f(row, "mean_plddt"), "ranking_score": f(row, "ranking_score"),
              "residue_rigid_projection_rmsd_a": projected_rmsd,
              "residue_rigid_projection_bond_loss_a2": projected_bond,
              "residue_rigid_projection_plane_loss_a2": projected_plane,
              "fitted_before_after_movement_rms_a": float(np.sqrt(np.square(aligned_move).mean())),
              "native_observed_fraction": f(row, "observed_atom_fraction")}
    alpha_rows = []
    for alpha in scales:
        candidate = source + alpha * (refined - source)
        _, rmsd, _ = fit_coordinates(candidate, target, observed)
        bond, plane = coordinate_geometry(candidate, bonds, ideals, rings)
        alpha_rows.append({"split": row["split"], "pdb_id": row["pdb_id"], "alpha": alpha,
                           "rmsd_a": rmsd, "improvement_a": pre_rmsd - rmsd,
                           "bond_loss_a2": bond, "plane_loss_a2": plane})
    best = min(alpha_rows, key=lambda item: item["rmsd_a"])
    result["oracle_best_alpha_in_grid"] = best["alpha"]
    result["oracle_best_rmsd_in_grid_a"] = best["rmsd_a"]
    extra_residues = []
    for r, res in enumerate(residues_table):
        extra_residues.append({"split": row["split"], "pdb_id": row["pdb_id"], **res,
                               "nonrigid_residue_rms_a": float(rigid_error[r]),
                               "training_frame_input_error_a": float(source_errors[r]) if valid[r] else "",
                               "training_frame_refined_error_a": float(refined_errors[r]) if valid[r] else "",
                               "training_mobility_target": float(gate_targets[r]) if valid[r] else "",
                               "protect_mask": int(protected[r])})
    return result, alpha_rows, extra_residues


def summary_group(rows):
    keys = ("input_rmsd_a", "refined_rmsd_a", "improvement_a", "movement_rms_a", "movement_mean_a",
            "movement_median_a", "bond_relative_motion_rms_a", "residue_nonrigid_rms_a",
            "displacement_target_cosine", "atom_fraction_move_lt_0.1", "atom_fraction_move_gt_1",
            "input_bond_rmse_a", "refined_bond_rmse_a", "bond_fraction_error_gt_0.3_after",
            "expected_gate_target_mean", "protected_residue_fraction", "coordinate_l2_before_a",
            "coordinate_l2_after_a", "geometry_weighted_before", "geometry_weighted_after",
            "no_regret_loss_a", "protect_loss_a2", "residue_rigid_projection_rmsd_a")
    return {"count": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in keys},
            "movement_rms_median_a": float(np.median([row["movement_rms_a"] for row in rows])),
            "targets_move_rms_lt_0.01": sum(row["movement_rms_a"] < 0.01 for row in rows),
            "targets_move_rms_lt_0.1": sum(row["movement_rms_a"] < 0.1 for row in rows),
            "targets_move_rms_gt_1": sum(row["movement_rms_a"] > 1 for row in rows),
            "targets_worse_ge_0.1": sum(row["improvement_a"] <= -0.1 for row in rows),
            "oracle_best_alpha_0": sum(row["oracle_best_alpha_in_grid"] == 0 for row in rows),
            "oracle_best_alpha_lt_1": sum(row["oracle_best_alpha_in_grid"] < 1 for row in rows),
            "oracle_best_alpha_gt_1": sum(row["oracle_best_alpha_in_grid"] > 1 for row in rows)}


def make_figures(rows, alpha_rows, residue_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for split, marker, color in (("val", "o", "#507cb9"), ("test", "^", "#e18b38")):
        subset = [row for row in rows if row["split"] == split]
        axes[0, 0].scatter([row["length_nt"] for row in subset],
                           [max(row["movement_rms_a"], 1e-6) for row in subset],
                           marker=marker, color=color, alpha=0.75, label=split, s=26)
        axes[0, 1].scatter([row["movement_rms_a"] for row in subset],
                           [row["refined_bond_rmse_a"] for row in subset],
                           marker=marker, color=color, alpha=0.75, s=26)
        scales = sorted({row["alpha"] for row in alpha_rows})
        average = [np.mean([row["improvement_a"] for row in alpha_rows if row["split"] == split and row["alpha"] == a]) for a in scales]
        axes[1, 0].plot(scales, average, marker=marker, color=color, label=split)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="RNA length (nt)", ylabel="Raw displacement RMS (A)", title="A. Amount of motion")
    axes[0, 0].axhline(0.1, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].legend()
    axes[0, 1].set(xlabel="Raw displacement RMS (A)", ylabel="Refined bond-length RMSE (A)", title="B. Motion and bond distortion")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].axvline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].set(xlabel="Scale of the fixed predicted residual", ylabel="Mean input - refined RMSD (A)", title="C. Diagnostic scale sweep (uses native)")
    axes[1, 0].legend()
    test = [row for row in rows if row["split"] == "test"]
    scatter = axes[1, 1].scatter([row["displacement_target_cosine"] for row in test], [row["improvement_a"] for row in test],
                                c=[row["movement_rms_a"] for row in test], cmap="viridis", s=35)
    fig.colorbar(scatter, ax=axes[1, 1], label="Displacement RMS (A)")
    axes[1, 1].axhline(0, color="gray", linewidth=0.8)
    axes[1, 1].axvline(0, color="gray", linewidth=0.8)
    axes[1, 1].set(xlabel="Cosine: prediction vs native target displacement", ylabel="RMSD improvement (A)", title="D. Test: direction and improvement")
    fig.tight_layout()
    fig.savefig(output / "movement_and_geometry.png", dpi=170)
    plt.close(fig)
    examples = [("test", "8JHP"), ("val", "7POF"), ("test", "9IOS"), ("test", "7ZJ4")]
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 12))
    for k, (split, pdb) in enumerate(examples):
        subset = [row for row in residue_rows if row["split"] == split and row["pdb_id"] == pdb]
        x = [int(row["residue"]) for row in subset]
        to_values = lambda field: [float(row[field]) if row[field] != "" else np.nan for row in subset]
        axes[k, 0].plot(x, to_values("input_rms_error_all_atom_fit_a"), color="#507cb9", label="before")
        axes[k, 0].plot(x, to_values("refined_rms_error_all_atom_fit_a"), color="#e18b38", label="after")
        axes[k, 0].set(title=f"{split}/{pdb}: native error", ylabel="Error (A)")
        axes[k, 0].legend()
        axes[k, 1].plot(x, to_values("mean_raw_movement_a"), color="#6c4ca0", label="total movement")
        axes[k, 1].plot(x, to_values("nonrigid_residue_rms_a"), color="#c14551", label="nonrigid residue RMS")
        axes[k, 1].set(title="Motion vs internal deformation", ylabel="Distance (A)")
        axes[k, 1].legend()
        axes[k, 2].plot(x, to_values("input_mean_abs_bond_error_a"), color="#507cb9", label="before")
        axes[k, 2].plot(x, to_values("refined_mean_abs_bond_error_a"), color="#e18b38", label="after")
        axes[k, 2].set(title="Local bond error", ylabel="Mean absolute error (A)")
        axes[k, 2].legend()
        for axis in axes[k]:
            axis.set_xlabel("Sequential residue")
    fig.tight_layout()
    fig.savefig(output / "representative_residue_diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[2] / "pymol_bundle")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation/local_rank1_diagnosis_20261013")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, alpha_rows, residue_rows = [], [], []
    scales = (0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)
    manifest = read_tsv(args.bundle / "manifest.tsv")
    for index, row in enumerate(manifest):
        result, alphas, residues = target_analysis(args.bundle, row, scales)
        rows.append(result)
        alpha_rows.extend(alphas)
        residue_rows.extend(residues)
        if (index + 1) % 25 == 0:
            print(f"ANALYZED {index+1}/{len(manifest)}", flush=True)
    write_tsv(output / "target_diagnostics.tsv", rows)
    write_tsv(output / "residual_scale_sweep.tsv", alpha_rows)
    write_tsv(output / "residue_diagnostics.tsv", residue_rows)
    summary = {}
    grouped = []
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = summary_group(split_rows)
        for name, select in (
            ("all", lambda row: True), ("length_le50", lambda row: row["length_nt"] <= 50),
            ("length_51_100", lambda row: 50 < row["length_nt"] <= 100),
            ("length_101_200", lambda row: 100 < row["length_nt"] <= 200),
            ("length_gt200", lambda row: row["length_nt"] > 200),
            ("input_lt2", lambda row: row["input_rmsd_a"] < 2),
            ("worse_ge0.1", lambda row: row["improvement_a"] < -0.1),
            ("better_ge0.5", lambda row: row["improvement_a"] > 0.5),
        ):
            subset = [row for row in split_rows if select(row)]
            if subset:
                grouped.append({"split": split, "group": name, **summary_group(subset)})
    write_tsv(output / "group_diagnostics.tsv", grouped)
    write_json(output / "summary.json", summary)
    make_figures(rows, alpha_rows, residue_rows, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DIAGNOSIS_READY {output}")


if __name__ == "__main__":
    main()
