#!/usr/bin/env python3
"""Build a FoldBench-style strict-rank-1 RNA multi-metric report.

This is a reporting-only program.  It reads ``reports/rank1_targets.csv``
created by ``summarize_foldbench_rna_rmsd.py`` and does not rerun
OpenStructure.  LDDT is the primary endpoint; TM-score, GDT-TS and C3' RMSD
are complementary continuous metrics.  No binary RNA-folding success rule is
invented.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LENGTH_ORDER = ("1-20", "21-50", "51-100", "101-200", ">200")
CHAIN_ORDER = ("1", "2", "3", "4", ">=5")
TIME_ORDER = ("pre_or_on_cutoff", "post_cutoff")

METRICS = {
    "lddt": {
        "source": "lddt",
        "label": "LDDT",
        "axis": "LDDT (higher is better)",
        "unit": "",
        "log": False,
    },
    "tm_score": {
        "source": "tm_score",
        "label": "TM-score",
        "axis": "TM-score (higher is better)",
        "unit": "",
        "log": False,
    },
    "gdt_ts": {
        "source": "oligo_gdtts",
        "label": "GDT-TS",
        "axis": "GDT-TS (higher is better)",
        "unit": "",
        "log": False,
    },
    "rmsd": {
        "source": "rmsd",
        "label": "C3' RMSD",
        "axis": "C3' RMSD (Å; lower is better)",
        "unit": "Å",
        "log": True,
    },
}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def save_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight")


def finite_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    valid = np.isfinite(values) & values.ge(0)
    if metric in {"lddt", "tm_score", "gdt_ts"}:
        valid &= values.le(1.000001)
    return values[valid]


def load_rank1(path: Path, expected_targets: Optional[int]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Strict rank-1 table not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "pdb_id",
        "seed",
        "sample",
        "ranking_score",
        "time_group",
        "rna_total_length",
        "length_group",
        "chain_count_group",
        *(item["source"] for item in METRICS.values()),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame["pdb_id"] = frame["pdb_id"].astype(str).str.upper()
    if frame["pdb_id"].duplicated().any():
        duplicates = frame.loc[frame["pdb_id"].duplicated(False), "pdb_id"].tolist()
        raise ValueError(f"Duplicate PDB IDs in strict rank-1 table: {duplicates[:20]}")
    if expected_targets is not None and len(frame) != expected_targets:
        raise ValueError(f"Expected {expected_targets} targets, found {len(frame)}")
    for name, definition in METRICS.items():
        frame[name] = pd.to_numeric(frame[definition["source"]], errors="coerce")
    frame["rna_total_length"] = pd.to_numeric(
        frame["rna_total_length"], errors="coerce"
    )
    frame["length_group"] = frame["length_group"].astype("string")
    frame["chain_count_group"] = frame["chain_count_group"].astype("string")
    return frame.sort_values("pdb_id", ignore_index=True)


def bootstrap_mean_ci(
    values: pd.Series,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan
    if len(array) == 1 or replicates < 2:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    batch_size = max(1, min(250, replicates))
    cursor = 0
    while cursor < replicates:
        batch = min(batch_size, replicates - cursor)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        means[cursor : cursor + batch] = array[indices].mean(axis=1)
        cursor += batch
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def group_definitions(rank1: pd.DataFrame):
    result = [
        ("Overall", "All targets", rank1),
        (
            "Time",
            "Pre/on cutoff (≤2021-09-30)",
            rank1[rank1["time_group"].eq("pre_or_on_cutoff")],
        ),
        (
            "Time",
            "Post-cutoff (>2021-09-30)",
            rank1[rank1["time_group"].eq("post_cutoff")],
        ),
    ]
    for label in LENGTH_ORDER:
        result.append(
            (
                "RNA length",
                f"{label} nt",
                rank1[rank1["length_group"].eq(label)],
            )
        )
    for label in CHAIN_ORDER:
        result.append(
            (
                "RNA chain count",
                label,
                rank1[rank1["chain_count_group"].eq(label)],
            )
        )
    return result


def build_summary(
    rank1: pd.DataFrame,
    *,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    rows = []
    seed_base = 20260821
    for group_index, (section, group, subset) in enumerate(group_definitions(rank1)):
        for metric_index, (metric, definition) in enumerate(METRICS.items()):
            values = finite_values(subset, metric)
            q25 = values.quantile(0.25) if len(values) else np.nan
            q75 = values.quantile(0.75) if len(values) else np.nan
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                replicates=bootstrap_replicates,
                seed=seed_base + group_index * 101 + metric_index,
            )
            rows.append(
                {
                    "section": section,
                    "group": group,
                    "metric": metric,
                    "metric_label": definition["label"],
                    "unit": definition["unit"],
                    "n_total": len(subset),
                    "n_valid": len(values),
                    "n_missing": len(subset) - len(values),
                    "coverage_percent": 100 * len(values) / len(subset)
                    if len(subset)
                    else np.nan,
                    "mean": values.mean() if len(values) else np.nan,
                    "mean_ci95_low": ci_low,
                    "mean_ci95_high": ci_high,
                    "median": values.median() if len(values) else np.nan,
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25 if len(values) else np.nan,
                    "p90": values.quantile(0.90) if len(values) else np.nan,
                    "minimum": values.min() if len(values) else np.nan,
                    "maximum": values.max() if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_primary_lddt(
    rank1: pd.DataFrame,
    summary: pd.DataFrame,
    figures: Path,
) -> None:
    plt = setup_matplotlib()
    definitions = [
        ("All", rank1, "#4C78A8"),
        (
            "Pre/on cutoff",
            rank1[rank1["time_group"].eq("pre_or_on_cutoff")],
            "#54A24B",
        ),
        (
            "Post-cutoff",
            rank1[rank1["time_group"].eq("post_cutoff")],
            "#E45756",
        ),
    ]
    arrays = [finite_values(subset, "lddt").to_numpy() for _, subset, _ in definitions]
    labels = [
        f"{label}\nvalid={len(values)}/{len(subset)}"
        for (label, subset, _), values in zip(definitions, arrays)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), constrained_layout=True)
    violin = axes[0].violinplot(
        arrays,
        positions=np.arange(1, 4),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.75,
    )
    for body, (_, _, color) in zip(violin["bodies"], definitions):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.30)
    box = axes[0].boxplot(
        arrays,
        positions=np.arange(1, 4),
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, (_, _, color) in zip(box["boxes"], definitions):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    rng = np.random.default_rng(20260821)
    for index, (values, (_, _, color)) in enumerate(zip(arrays, definitions), start=1):
        axes[0].scatter(
            index + rng.uniform(-0.18, 0.18, size=len(values)),
            values,
            s=6,
            alpha=0.12,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
    axes[0].set_xticks(np.arange(1, 4), labels)
    axes[0].set_ylabel("Strict rank-1 LDDT")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("A. Distribution of the primary RNA metric")
    axes[0].grid(True, axis="y", alpha=0.22)

    summary_groups = [
        "All targets",
        "Pre/on cutoff (≤2021-09-30)",
        "Post-cutoff (>2021-09-30)",
    ]
    rows = [
        summary[(summary["metric"].eq("lddt")) & summary["group"].eq(group)].iloc[0]
        for group in summary_groups
    ]
    means = np.array([row["mean"] for row in rows])
    lows = np.array([row["mean_ci95_low"] for row in rows])
    highs = np.array([row["mean_ci95_high"] for row in rows])
    colors = [item[2] for item in definitions]
    positions = np.arange(3)
    for x, mean, low, high, color in zip(positions, means, lows, highs, colors):
        axes[1].errorbar(
            x,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="none",
            ecolor=color,
            elinewidth=2.2,
            capsize=5,
        )
        axes[1].scatter(x, mean, s=85, color=color, zorder=3)
    for x, mean, row in zip(positions, means, rows):
        axes[1].text(
            x,
            min(0.99, mean + 0.055),
            f"{mean:.3f}\n95% CI {row['mean_ci95_low']:.3f}–{row['mean_ci95_high']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axes[1].set_xticks(positions, [item[0] for item in definitions])
    axes[1].set_ylabel("Mean strict rank-1 LDDT")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("B. Mean LDDT with bootstrap 95% CI")
    axes[1].grid(True, axis="y", alpha=0.22)
    fig.suptitle("FoldBench-style RNA folding performance: primary LDDT", fontsize=14)
    fig.text(
        0.5,
        -0.025,
        "Rank-1 is selected by the highest model ranking_score before consulting ground-truth metrics. "
        "Missing LDDT values are excluded only from LDDT summaries and remain in the reported denominator.",
        ha="center",
        fontsize=8.2,
    )
    save_figure(fig, figures / "Figure1_primary_LDDT")
    plt.close(fig)


def _metric_box_panel(
    ax,
    rank1: pd.DataFrame,
    *,
    metric: str,
    column: str,
    order: tuple[str, ...],
    title: str,
) -> None:
    definition = METRICS[metric]
    arrays = []
    labels = []
    rng = np.random.default_rng(20260821 + sum(ord(char) for char in metric + column))
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"]
    for index, group in enumerate(order, start=1):
        subset = rank1[rank1[column].eq(group)]
        values = finite_values(subset, metric).to_numpy()
        if definition["log"]:
            values = values[values > 0]
        arrays.append(values)
        labels.append(f"{group}\nn={len(values)}/{len(subset)}")
        if len(values):
            ax.scatter(
                index + rng.uniform(-0.16, 0.16, size=len(values)),
                values,
                s=5,
                alpha=0.11,
                color=colors[index - 1],
                edgecolors="none",
                rasterized=True,
            )
    box = ax.boxplot(
        arrays,
        positions=np.arange(1, len(order) + 1),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    for index, values in enumerate(arrays, start=1):
        if len(values):
            ax.scatter(index, np.mean(values), marker="D", s=27, color="white", edgecolor="black", zorder=4)
    ax.set_xticks(np.arange(1, len(order) + 1), labels)
    ax.set_ylabel(definition["axis"])
    ax.set_title(title)
    if definition["log"]:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", which="both", alpha=0.20)


def plot_stratified_metrics(rank1: pd.DataFrame, figures: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 19.5), constrained_layout=True)
    for row, metric in enumerate(METRICS):
        _metric_box_panel(
            axes[row, 0],
            rank1,
            metric=metric,
            column="length_group",
            order=LENGTH_ORDER,
            title=f"{METRICS[metric]['label']} by total RNA length",
        )
        _metric_box_panel(
            axes[row, 1],
            rank1,
            metric=metric,
            column="chain_count_group",
            order=CHAIN_ORDER,
            title=f"{METRICS[metric]['label']} by RNA chain count",
        )
    fig.suptitle(
        "Strict rank-1 RNA metrics stratified by target complexity",
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.012,
        "Box center: median; box: IQR; white diamond: mean. Dots show every valid target. "
        "Labels report metric-valid/total targets; RMSD uses a logarithmic y-axis.",
        ha="center",
        fontsize=8.5,
    )
    save_figure(fig, figures / "Figure2_metrics_by_length_and_chain_count")
    plt.close(fig)


def select_representatives(rank1: pd.DataFrame) -> pd.DataFrame:
    complete = rank1.copy()
    mask = np.ones(len(complete), dtype=bool)
    for metric in METRICS:
        values = pd.to_numeric(complete[metric], errors="coerce")
        mask &= np.isfinite(values) & values.ge(0)
    complete = complete.loc[mask].copy()
    if complete.empty:
        return pd.DataFrame()
    lddt_q25, lddt_q75 = complete["lddt"].quantile([0.25, 0.75])
    rmsd_q25, rmsd_q75 = complete["rmsd"].quantile([0.25, 0.75])
    selections = []

    def choose(category: str, subset: pd.DataFrame, sort_columns, ascending):
        if subset.empty:
            return
        used = {item["pdb_id"] for item in selections}
        unused = subset[~subset["pdb_id"].isin(used)]
        if not unused.empty:
            subset = unused
        row = subset.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
        item = row.to_dict()
        item["case_category"] = category
        selections.append(item)

    choose(
        "High local and high global accuracy",
        complete[(complete["lddt"] >= lddt_q75) & (complete["rmsd"] <= rmsd_q25)],
        ["rmsd", "lddt"],
        [True, False],
    )
    choose(
        "High LDDT but poor global fold",
        complete[(complete["lddt"] >= lddt_q75) & (complete["rmsd"] >= rmsd_q75)],
        ["rmsd", "lddt"],
        [False, False],
    )
    choose(
        "Low local and poor global accuracy",
        complete[(complete["lddt"] <= lddt_q25) & (complete["rmsd"] >= rmsd_q75)],
        ["rmsd", "lddt"],
        [False, True],
    )
    choose(
        "Difficult long-RNA target",
        complete[complete["rna_total_length"] > 200],
        ["rmsd", "lddt"],
        [False, True],
    )
    if not selections:
        return pd.DataFrame()
    columns = [
        "case_category",
        "pdb_id",
        "seed",
        "sample",
        "ranking_score",
        "time_group",
        "rna_total_length",
        "chain_count_group",
        "lddt",
        "tm_score",
        "gdt_ts",
        "rmsd",
        "prediction_path",
        "reference_path",
    ]
    result = pd.DataFrame(selections)
    return result[[column for column in columns if column in result.columns]]


def _rank_correlation(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 2:
        return np.nan
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def plot_local_global_relationships(
    rank1: pd.DataFrame,
    representatives: pd.DataFrame,
    figures: Path,
) -> None:
    plt = setup_matplotlib()
    colors = dict(zip(LENGTH_ORDER, ["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"]))
    markers = {"pre_or_on_cutoff": "o", "post_cutoff": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    for length_group in LENGTH_ORDER:
        for time_group in TIME_ORDER:
            subset = rank1[
                rank1["length_group"].eq(length_group)
                & rank1["time_group"].eq(time_group)
            ]
            rmsd_pair = subset[["rmsd", "lddt"]].apply(pd.to_numeric, errors="coerce").dropna()
            rmsd_pair = rmsd_pair[(rmsd_pair["rmsd"] > 0) & rmsd_pair["lddt"].between(0, 1)]
            axes[0].scatter(
                rmsd_pair["rmsd"],
                rmsd_pair["lddt"],
                s=13,
                alpha=0.38,
                color=colors[length_group],
                marker=markers[time_group],
                edgecolors="none",
                rasterized=True,
            )
            tm_pair = subset[["tm_score", "lddt"]].apply(pd.to_numeric, errors="coerce").dropna()
            tm_pair = tm_pair[tm_pair["tm_score"].between(0, 1) & tm_pair["lddt"].between(0, 1)]
            axes[1].scatter(
                tm_pair["tm_score"],
                tm_pair["lddt"],
                s=13,
                alpha=0.38,
                color=colors[length_group],
                marker=markers[time_group],
                edgecolors="none",
                rasterized=True,
            )
    pair_rmsd = rank1[["rmsd", "lddt"]].apply(pd.to_numeric, errors="coerce").dropna()
    pair_tm = rank1[["tm_score", "lddt"]].apply(pd.to_numeric, errors="coerce").dropna()
    rho_rmsd = _rank_correlation(pair_rmsd["rmsd"], pair_rmsd["lddt"])
    rho_tm = _rank_correlation(pair_tm["tm_score"], pair_tm["lddt"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Strict rank-1 C3' RMSD (Å, log scale)")
    axes[0].set_ylabel("Strict rank-1 LDDT")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"A. Local quality versus geometric deviation\nSpearman ρ={rho_rmsd:.3f}, n={len(pair_rmsd)}")
    axes[1].set_xlabel("Strict rank-1 TM-score")
    axes[1].set_ylabel("Strict rank-1 LDDT")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title(f"B. Local quality versus global topology\nSpearman ρ={rho_tm:.3f}, n={len(pair_tm)}")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.20)
    if not representatives.empty:
        for row in representatives.itertuples(index=False):
            axes[0].annotate(
                row.pdb_id,
                (row.rmsd, row.lddt),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
                weight="bold",
            )
            axes[1].annotate(
                row.pdb_id,
                (row.tm_score, row.lddt),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
                weight="bold",
            )
    from matplotlib.lines import Line2D

    length_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=f"{label} nt", markersize=6)
        for label in LENGTH_ORDER
    ]
    time_handles = [
        Line2D([0], [0], marker=markers[label], linestyle="", color="#555555", label=("Pre/on cutoff" if label == "pre_or_on_cutoff" else "Post-cutoff"), markersize=6)
        for label in TIME_ORDER
    ]
    axes[1].legend(handles=length_handles + time_handles, loc="lower right", ncol=2, frameon=True)
    fig.suptitle("Local versus global RNA folding accuracy", fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "Colors indicate total RNA length; circles and triangles indicate temporal groups. "
        "Annotated PDB IDs are automatically selected representative cases for later structure inspection.",
        ha="center",
        fontsize=8.3,
    )
    save_figure(fig, figures / "Figure3_local_vs_global_accuracy")
    plt.close(fig)


def autosize_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            width = min(
                60,
                max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2,
            )
            sheet.column_dimensions[column_cells[0].column_letter].width = max(10, width)
    workbook.save(path)


def build_missing_table(rank1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in rank1.itertuples(index=False):
        missing = []
        for metric in METRICS:
            value = getattr(row, metric)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)) or float(value) < 0:
                missing.append(metric)
        if missing:
            rows.append(
                {
                    "pdb_id": row.pdb_id,
                    "seed": row.seed,
                    "sample": row.sample,
                    "missing_metrics": ";".join(missing),
                    "time_group": row.time_group,
                    "rna_total_length": row.rna_total_length,
                    "length_group": row.length_group,
                    "chain_count_group": row.chain_count_group,
                    "evaluation_protocol": getattr(row, "evaluation_protocol", ""),
                    "eval_status": getattr(row, "eval_status", ""),
                    "eval_issue": getattr(row, "eval_issue", ""),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output: Path,
    *,
    root: Path,
    rank1: pd.DataFrame,
    summary: pd.DataFrame,
    representatives: pd.DataFrame,
    bootstrap_replicates: int,
) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    primary = summary[summary["metric"].eq("lddt")].copy()
    missing = build_missing_table(rank1)
    primary.to_csv(tables / "Table1_primary_LDDT.tsv", sep="\t", index=False)
    summary.to_csv(tables / "Table2_all_metrics.tsv", sep="\t", index=False)
    rank1.to_csv(tables / "strict_rank1_all_metrics.csv", index=False)
    missing.to_csv(tables / "TableS1_missing_metrics_by_target.tsv", sep="\t", index=False)
    representatives.to_csv(tables / "TableS2_representative_cases.tsv", sep="\t", index=False)

    definitions = pd.DataFrame(
        [
            {
                "item": "Evaluation label",
                "definition": "FoldBench-style strict-rank-1 evaluation on the curated PDB RNA target set; not the official FoldBench low-homology RNA-monomer dataset.",
            },
            {
                "item": "Rank-1 selection",
                "definition": "Highest Protenix ranking_score is selected before consulting any ground-truth metric; no lower-ranked fallback.",
            },
            {
                "item": "Primary endpoint",
                "definition": "OpenStructure all-atom LDDT, summarized as mean with bootstrap 95% CI plus median and IQR.",
            },
            {
                "item": "Complementary endpoints",
                "definition": "TM-score, GDT-TS (oligo_gdtts), and mapped nucleic-acid C3' RMSD.",
            },
            {
                "item": "Missing metrics",
                "definition": "Validity and coverage are computed independently for each metric. Rigid-only RMSD rescue does not create LDDT or TM-score values.",
            },
            {
                "item": "Success rate",
                "definition": "No binary RNA-folding success threshold is applied because FoldBench does not define one for RNA monomers.",
            },
            {
                "item": "Temporal cutoff",
                "definition": "post_cutoff means PDB release date after 2021-09-30.",
            },
        ]
    )
    workbook = output / "FoldBench_style_RNA_multimetric_report.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        definitions.to_excel(writer, sheet_name="README", index=False)
        primary.to_excel(writer, sheet_name="Primary_LDDT", index=False)
        summary.to_excel(writer, sheet_name="All_metrics", index=False)
        rank1.to_excel(writer, sheet_name="Strict_rank1_targets", index=False)
        missing.to_excel(writer, sheet_name="Missing_metrics", index=False)
        representatives.to_excel(writer, sheet_name="Representative_cases", index=False)
    autosize_workbook(workbook)

    lines = [
        "FoldBench-style Protenix RNA multi-metric report",
        "",
        f"Total strict rank-1 targets: {len(rank1)}",
        "No binary RNA folding success criterion was applied.",
        "",
    ]
    for group in ("All targets", "Post-cutoff (>2021-09-30)"):
        lines.append(group + ":")
        for metric, definition in METRICS.items():
            row = summary[(summary["group"].eq(group)) & summary["metric"].eq(metric)].iloc[0]
            # Keep the console summary ASCII-safe for non-UTF-8 login terminals.
            unit = " A" if definition["unit"] == "Å" else ""
            lines.append(
                f"  {definition['label']}: valid={int(row.n_valid)}/{int(row.n_total)} "
                f"({row.coverage_percent:.2f}%), mean={row['mean']:.4f}{unit}, "
                f"95% CI={row.mean_ci95_low:.4f}-{row.mean_ci95_high:.4f}{unit}, "
                f"median={row['median']:.4f}{unit}"
            )
        lines.append("")
    (output / "multimetric_report_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    captions = [
        "Figure 1. FoldBench-style strict rank-1 LDDT, the primary RNA folding endpoint. Distributions, means, bootstrap 95% confidence intervals, valid counts and total counts are shown for the full, pre/on-cutoff and post-cutoff sets.",
        "",
        "Figure 2. Strict rank-1 LDDT, TM-score, GDT-TS and C3' RMSD stratified by total RNA length and RNA chain count. Each metric uses its own valid-target denominator; RMSD is displayed on a logarithmic scale.",
        "",
        "Figure 3. Relationship between local and global RNA accuracy. LDDT is compared with C3' RMSD and TM-score; colors encode RNA length and marker shape encodes temporal group. Annotated targets are listed in Table S2 for structural inspection.",
    ]
    (output / "figure_captions.txt").write_text(
        "\n".join(captions) + "\n", encoding="utf-8"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_root": str(root),
        "input_rank1_table": str(root / "reports" / "rank1_targets.csv"),
        "output_root": str(output),
        "target_count": len(rank1),
        "bootstrap_replicates": bootstrap_replicates,
        "metrics": METRICS,
        "strict_rank1_before_ground_truth_metrics": True,
        "binary_success_rule_applied": False,
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    root = Path(args.rmsd_root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root.parent / "foldbench_style_multimetric_report"
    )
    output.mkdir(parents=True, exist_ok=True)
    rank1 = load_rank1(
        root / "reports" / "rank1_targets.csv",
        getattr(args, "expected_targets", None),
    )
    bootstrap_replicates = int(args.bootstrap_replicates)
    if bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    summary = build_summary(
        rank1,
        bootstrap_replicates=bootstrap_replicates,
    )
    representatives = select_representatives(rank1)
    figures = output / "figures"
    plot_primary_lddt(rank1, summary, figures)
    plot_stratified_metrics(rank1, figures)
    plot_local_global_relationships(rank1, representatives, figures)
    write_outputs(
        output,
        root=root,
        rank1=rank1,
        summary=summary,
        representatives=representatives,
        bootstrap_replicates=bootstrap_replicates,
    )
    print((output / "multimetric_report_summary.txt").read_text(encoding="utf-8"))
    print(f"Multi-metric report written to: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rmsd-root",
        default="~/Json_data/Foldbench_evaluation/rmsd",
        help="Existing FoldBench-style evaluation root containing reports/rank1_targets.csv.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-targets", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
