#!/usr/bin/env python3
"""Build the single-chain RNA Data_V1 chronological split.

The pipeline keeps only PDB entries containing exactly one RNA chain, joins
each entry to the strict rank-1 Protenix C3' RMSD table, applies the RMSD
cutoff only to the training period, and preserves the original test-set
homology/de-redundancy contract.

Dry-run is the default.  Execute mode creates empty lower-case PDB directories
under ``~/Data_V1/{train,val,test}``; it never copies or deletes CIF files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import split_rna_dataset as legacy


PIPELINE_VERSION = "3.0-single-chain-rmsd15"
DEFAULT_CIF_DIR = Path("~/pdb_data").expanduser()
DEFAULT_DATA_DIR = Path("~/Data_V1").expanduser()
DEFAULT_REPORT_ROOT = Path("~/Code/pipeline_reports").expanduser()
DEFAULT_RANK1_CSV = Path(
    "~/Json_data/Foldbench_evaluation/rmsd/reports/rank1_targets.csv"
).expanduser()
DEFAULT_RMSD_EXCLUSION_LIST = Path(
    "~/Json_data/Foldbench_evaluation/rmsd/reports/"
    "exclude_strict_rank1_rmsd_pdb.txt"
).expanduser()
DEFAULT_EXCLUSION_XLSX = legacy.DEFAULT_EXCLUSION_XLSX

EXPECTED_SOURCE_CIFS = 2246
EXPECTED_SOURCE_EXCLUSIONS = 5
EXPECTED_INCLUDED_PDBS = 2241
EXPECTED_RANK1_TARGETS = 2199
EXPECTED_RMSD_EXCLUSIONS = 32

HIT_COLUMNS = legacy.REPORT_COLUMNS["reference_redundancy_hits.tsv"]
SELECTION_COLUMNS = [
    "PDB_ID",
    "RELEASE_DATE",
    "INITIAL_SPLIT",
    "RNA_ENTITY_COUNT",
    "RNA_CHAIN_COUNT",
    "RNA_LENGTH",
    "STRICT_RANK1_RMSD_ANGSTROM",
    "RMSD_EVAL_STATUS",
    "FROZEN_RMSD_EXCLUSION",
    "SELECTION_STATUS",
    "SELECTION_REASON",
    "SELECTED_SPLIT",
]
MANIFEST_COLUMNS = [
    "PDB_ID",
    "RELEASE_DATE",
    "INITIAL_SPLIT",
    "STRICT_RANK1_RMSD_ANGSTROM",
    "SELECTION_STATUS",
    "FINAL_SPLIT",
    "FINAL_STATUS",
    "EXCLUSION_REASON",
    "RNA_CHAIN_ID",
    "TARGET_DIRECTORY",
]


@dataclass(frozen=True)
class Rank1Record:
    pdb_id: str
    release_date: date
    chain_count: int
    rmsd: float | None
    eval_status: str
    seed: str
    sample: str
    ranking_score: str

    @property
    def metric_valid(self) -> bool:
        return (
            self.eval_status == "SUCCESS"
            and self.rmsd is not None
            and math.isfinite(self.rmsd)
            and self.rmsd >= 0
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the single-chain Data_V1 split with a training-only strict "
            "rank-1 C3' RMSD cutoff and test-set homology filtering."
        )
    )
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--exclusion-xlsx", type=Path, default=DEFAULT_EXCLUSION_XLSX)
    parser.add_argument("--rank1-csv", type=Path, default=DEFAULT_RANK1_CSV)
    parser.add_argument(
        "--rmsd-exclusion-list", type=Path, default=DEFAULT_RMSD_EXCLUSION_LIST
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--train-end", type=date.fromisoformat, default=date(2021, 9, 30))
    parser.add_argument("--val-end", type=date.fromisoformat, default=date(2023, 12, 31))
    parser.add_argument("--train-rmsd-max", type=float, default=15.0)
    parser.add_argument("--min-seq-id", type=float, default=0.80)
    parser.add_argument("--min-query-cov", type=float, default=0.80)
    parser.add_argument("--min-target-cov", type=float, default=0.80)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 executable")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="create empty lower-case PDB directories; default is report-only dry-run",
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="development/testing only: skip fixed 2246/5/2241/2199/32 count checks",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="development/testing only: do not generate distribution figures",
    )
    args = parser.parse_args(argv)
    if args.train_end >= args.val_end:
        parser.error("--train-end must be earlier than --val-end")
    if not math.isfinite(args.train_rmsd_max) or args.train_rmsd_max < 0:
        parser.error("--train-rmsd-max must be a finite non-negative number")
    for name in ("min_seq_id", "min_query_cov", "min_target_cov"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def make_report_dir(root: Path, execute: bool, rmsd_max: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "EXECUTE" if execute else "DRYRUN"
    cutoff = f"{rmsd_max:g}A".replace(".", "p")
    base = f"DATA_SPLIT_V1_SINGLECHAIN_RMSD{cutoff}_{timestamp}_{mode}"
    candidate = root / base
    counter = 2
    while candidate.exists():
        candidate = root / f"{base}_{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def read_id_list(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"PDB ID list not found: {path}")
    result: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        pdb_id = legacy.normalize_pdb_id(text, f"{path}:{line_number}")
        if pdb_id in result:
            raise ValueError(f"Duplicate PDB ID {pdb_id} in {path}")
        result.add(pdb_id)
    return result


def optional_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def load_rank1_records(path: Path) -> dict[str, Rank1Record]:
    if not path.is_file():
        raise FileNotFoundError(f"Strict rank-1 CSV not found: {path}")
    required = {
        "pdb_id",
        "seed",
        "sample",
        "ranking_score",
        "eval_status",
        "rmsd",
        "release_date",
        "chain_count",
    }
    records: dict[str, Rank1Record] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            pdb_id = legacy.normalize_pdb_id(row["pdb_id"], f"{path}:{line_number}")
            if pdb_id in records:
                raise ValueError(f"Duplicate strict rank-1 PDB ID {pdb_id} in {path}")
            release_date = legacy.parse_iso_date(
                str(row["release_date"]).strip(), f"{path}:{line_number}"
            )
            try:
                raw_chain_count = float(str(row["chain_count"]).strip())
                chain_count = int(raw_chain_count)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid chain_count for {pdb_id} in {path}: {row['chain_count']!r}"
                ) from exc
            if chain_count < 1 or raw_chain_count != chain_count:
                raise ValueError(
                    f"Non-positive or non-integral chain_count for {pdb_id}: {raw_chain_count}"
                )
            records[pdb_id] = Rank1Record(
                pdb_id=pdb_id,
                release_date=release_date,
                chain_count=chain_count,
                rmsd=optional_float(row["rmsd"]),
                eval_status=str(row["eval_status"]).strip(),
                seed=str(row["seed"]).strip(),
                sample=str(row["sample"]).strip(),
                ranking_score=str(row["ranking_score"]).strip(),
            )
    return records


def entry_chain_count(entry: legacy.Entry) -> int:
    return sum(len(entity.chain_ids) for entity in entry.entities)


def entry_rna_length(entry: legacy.Entry) -> int:
    return sum(len(entity.search_sequence) * len(entity.chain_ids) for entity in entry.entities)


def select_entries(
    entries: dict[str, legacy.Entry],
    rank1: dict[str, Rank1Record],
    frozen_rmsd_exclusions: set[str],
    train_end: date,
    val_end: date,
    train_rmsd_max: float,
) -> tuple[dict[str, str], list[dict[str, Any]], Counter[str]]:
    assignments: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for pdb_id, entry in sorted(entries.items()):
        split = legacy.initial_split(entry.release.release_date, train_end, val_end)
        chain_count = entry_chain_count(entry)
        record = rank1.get(pdb_id)
        if record is not None:
            if record.release_date != entry.release.release_date:
                raise ValueError(
                    f"Release-date mismatch for {pdb_id}: CIF={entry.release.release_date}, "
                    f"rank1={record.release_date}"
                )
            if record.chain_count != chain_count:
                raise ValueError(
                    f"RNA-chain-count mismatch for {pdb_id}: CIF={chain_count}, "
                    f"rank1={record.chain_count}"
                )

        selected_split = ""
        if chain_count != 1:
            status = "EXCLUDE_MULTI_RNA_CHAIN"
            reason = f"RNA_CHAIN_COUNT={chain_count}; required exactly 1"
        elif record is None:
            status = "EXCLUDE_NO_STRICT_RANK1"
            reason = "PDB is absent from rank1_targets.csv"
        elif pdb_id in frozen_rmsd_exclusions:
            status = "EXCLUDE_FROZEN_RMSD"
            reason = "PDB is listed in exclude_strict_rank1_rmsd_pdb.txt"
        elif not record.metric_valid:
            status = "EXCLUDE_INVALID_RMSD"
            reason = f"eval_status={record.eval_status!r}, rmsd={record.rmsd!r}"
        elif split == "train" and record.rmsd > train_rmsd_max:
            status = "EXCLUDE_TRAIN_RMSD_CUTOFF"
            reason = f"strict rank-1 RMSD {record.rmsd:g} A > {train_rmsd_max:g} A"
        else:
            status = "SELECTED"
            reason = ""
            selected_split = split
            assignments[pdb_id] = split
        statuses[status] += 1
        rows.append(
            {
                "PDB_ID": pdb_id,
                "RELEASE_DATE": entry.release.release_date.isoformat(),
                "INITIAL_SPLIT": split,
                "RNA_ENTITY_COUNT": len(entry.entities),
                "RNA_CHAIN_COUNT": chain_count,
                "RNA_LENGTH": entry_rna_length(entry),
                "STRICT_RANK1_RMSD_ANGSTROM": record.rmsd if record else None,
                "RMSD_EVAL_STATUS": record.eval_status if record else "",
                "FROZEN_RMSD_EXCLUSION": pdb_id in frozen_rmsd_exclusions,
                "SELECTION_STATUS": status,
                "SELECTION_REASON": reason,
                "SELECTED_SPLIT": selected_split,
            }
        )
    return assignments, rows, statuses


def parse_all_entries(
    cif_dir: Path,
    exclusions: set[str],
    report_dir: Path,
    logger: legacy.RunLogger,
) -> tuple[dict[str, legacy.Entry], dict[str, Path]]:
    cif_files = legacy.index_cifs(cif_dir)
    missing_exclusions = exclusions - set(cif_files)
    if missing_exclusions:
        raise ValueError(f"Source exclusion IDs missing from CIF directory: {sorted(missing_exclusions)}")
    entries: dict[str, legacy.Entry] = {}
    inventory: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, (pdb_id, path) in enumerate(sorted(cif_files.items()), start=1):
        excluded = pdb_id in exclusions
        try:
            if excluded:
                digest = legacy.sha256_file(path)
            else:
                entry = legacy.parse_entry(path, pdb_id)
                entries[pdb_id] = entry
                digest = entry.sha256
            inventory.append(
                {
                    "PDB_ID": pdb_id,
                    "CIF_PATH": str(path),
                    "FILE_SIZE_BYTES": path.stat().st_size,
                    "SHA256": digest,
                    "EXCLUDED": excluded,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "PDB_ID": pdb_id,
                    "CIF_PATH": str(path),
                    "ERROR": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % 100 == 0 or index == len(cif_files):
            logger.log(f"Parsed {index}/{len(cif_files)} CIF files; errors={len(errors)}")
    legacy.write_tsv(
        report_dir / "source_inventory.tsv",
        legacy.REPORT_COLUMNS["source_inventory.tsv"],
        inventory,
    )
    if errors:
        legacy.write_tsv(
            report_dir / "parse_errors.tsv", ["PDB_ID", "CIF_PATH", "ERROR"], errors
        )
        raise ValueError(f"{len(errors)} CIF files failed validation; see parse_errors.tsv")
    return entries, cif_files


def run_pair_search(
    *,
    query_entities: list[legacy.Entity],
    target_entities: list[legacy.Entity],
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    temporary_dir: Path,
    args: argparse.Namespace,
    logger: legacy.RunLogger,
    exclude_same_pdb: bool,
) -> list[legacy.Hit]:
    query_map = legacy.write_fasta(query_fasta, query_entities)
    target_map = legacy.write_fasta(target_fasta, target_entities)
    if not query_entities or not target_entities:
        output_tsv.write_text("", encoding="utf-8")
        return []
    legacy.run_mmseqs(
        executable=args.mmseqs,
        query_fasta=query_fasta,
        target_fasta=target_fasta,
        output_tsv=output_tsv,
        temporary_dir=temporary_dir,
        min_seq_id=args.min_seq_id,
        min_cov=min(args.min_query_cov, args.min_target_cov),
        threads=args.threads,
        logger=logger,
    )
    mmseqs_hits = legacy.load_hits(
        output_tsv,
        query_map,
        target_map,
        args.min_seq_id,
        args.min_query_cov,
        args.min_target_cov,
        exclude_same_pdb=exclude_same_pdb,
    )
    fallback_hits = legacy.short_sequence_hits(
        query_entities,
        target_entities,
        args.min_seq_id,
        args.min_query_cov,
        args.min_target_cov,
        exclude_same_pdb=exclude_same_pdb,
    )
    hits = legacy.merge_hits(mmseqs_hits, fallback_hits)
    logger.log(
        f"Hits for {output_tsv.name}: MMseqs2={len(mmseqs_hits)}, "
        f"short-fallback={len(fallback_hits)}, merged={len(hits)}"
    )
    return hits


def percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (position - low) * (ordered[high] - ordered[low]))


def distribution_rows(
    expected: dict[str, set[str]],
    rank1: dict[str, Rank1Record],
    entries: dict[str, legacy.Entry],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        ids = sorted(expected[split])
        rmsd = [rank1[pdb_id].rmsd for pdb_id in ids]
        values = [float(value) for value in rmsd if value is not None]
        lengths = [entry_rna_length(entries[pdb_id]) for pdb_id in ids]
        rows.append(
            {
                "SPLIT": split,
                "PDB_COUNT": len(ids),
                "RMSD_MEAN": sum(values) / len(values) if values else None,
                "RMSD_Q25": percentile(values, 0.25),
                "RMSD_MEDIAN": percentile(values, 0.50),
                "RMSD_Q75": percentile(values, 0.75),
                "RMSD_P90": percentile(values, 0.90),
                "RMSD_MAX": max(values) if values else None,
                "RNA_LENGTH_MEDIAN": percentile(lengths, 0.50),
                "RNA_LENGTH_P90": percentile(lengths, 0.90),
            }
        )
    return rows


def kde(values: Sequence[float], points: int = 400) -> tuple[Any, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([]), np.asarray([])
    span = float(np.ptp(array))
    low = max(0.0, float(array.min()) - 0.05 * max(1.0, span))
    high = float(array.max()) + 0.05 * max(1.0, span)
    if high <= low:
        high = low + 1.0
    grid = np.linspace(low, high, points)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    bandwidth = 1.06 * std * array.size ** (-0.2) if std > 0 else 0.0
    bandwidth = max(bandwidth, (high - low) / 100.0, 1e-3)
    scaled = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).mean(axis=1)
    density /= bandwidth * math.sqrt(2.0 * math.pi)
    return grid, density


def plot_distributions(
    report_dir: Path,
    expected: dict[str, set[str]],
    rank1: dict[str, Rank1Record],
    entries: dict[str, legacy.Entry],
    train_rmsd_max: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires numpy and matplotlib. Install them in the server "
            "environment or use --skip-plots for development only."
        ) from exc

    figures = report_dir / "figures"
    figures.mkdir()
    colors = {"train": "#4C78A8", "val": "#F2A541", "test": "#E45756"}
    rmsd_groups = {
        split: [float(rank1[pdb_id].rmsd) for pdb_id in sorted(ids)]
        for split, ids in expected.items()
    }
    length_groups = {
        split: [float(entry_rna_length(entries[pdb_id])) for pdb_id in sorted(ids)]
        for split, ids in expected.items()
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for split in ("train", "val", "test"):
        grid, density = kde(rmsd_groups[split])
        if len(grid):
            ax.plot(grid, density, color=colors[split], linewidth=2, label=f"{split} (n={len(rmsd_groups[split])})")
            ax.fill_between(grid, density, color=colors[split], alpha=0.12)
    ax.axvline(train_rmsd_max, color="#333333", linestyle="--", linewidth=1.5, label=f"train cutoff = {train_rmsd_max:g} A")
    ax.set_xlabel("Strict rank-1 C3' RMSD (A)")
    ax.set_ylabel("Kernel density")
    ax.set_title("Final Data_V1 RMSD distributions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "rmsd_density.png", dpi=200)
    fig.savefig(figures / "rmsd_density.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for split in ("train", "val", "test"):
        transformed = np.log1p(rmsd_groups[split])
        grid, density = kde(transformed)
        if len(grid):
            ax.plot(grid, density, color=colors[split], linewidth=2, label=f"{split} (n={len(transformed)})")
            ax.fill_between(grid, density, color=colors[split], alpha=0.12)
    ax.set_xlabel("log1p(strict rank-1 C3' RMSD [A])")
    ax.set_ylabel("Kernel density")
    ax.set_title("Final Data_V1 RMSD distributions (log scale)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "rmsd_density_log1p.png", dpi=200)
    fig.savefig(figures / "rmsd_density_log1p.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for split in ("train", "val", "test"):
        values = np.sort(np.asarray(rmsd_groups[split], dtype=float))
        if values.size:
            y = np.arange(1, values.size + 1) / values.size
            ax.step(values, y, where="post", color=colors[split], linewidth=2, label=f"{split} (n={values.size})")
    ax.axvline(train_rmsd_max, color="#333333", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Strict rank-1 C3' RMSD (A)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("Final Data_V1 RMSD ECDF")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "rmsd_ecdf.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for split in ("train", "val", "test"):
        grid, density = kde(length_groups[split])
        if len(grid):
            ax.plot(grid, density, color=colors[split], linewidth=2, label=f"{split} (n={len(length_groups[split])})")
            ax.fill_between(grid, density, color=colors[split], alpha=0.12)
    ax.set_xlabel("RNA length (nt)")
    ax.set_ylabel("Kernel density")
    ax.set_title("Final Data_V1 RNA-length distributions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "rna_length_density.png", dpi=200)
    fig.savefig(figures / "rna_length_density.svg")
    plt.close(fig)


def run_pipeline(
    args: argparse.Namespace, report_dir: Path, logger: legacy.RunLogger
) -> dict[str, Any]:
    cif_dir = args.cif_dir.expanduser().resolve()
    exclusion_xlsx = args.exclusion_xlsx.expanduser().resolve()
    rank1_csv = args.rank1_csv.expanduser().resolve()
    rmsd_exclusion_list = args.rmsd_exclusion_list.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    for path, label in (
        (cif_dir, "CIF directory"),
        (exclusion_xlsx, "source exclusion XLSX"),
        (rank1_csv, "strict rank-1 CSV"),
        (rmsd_exclusion_list, "RMSD exclusion list"),
    ):
        expected = path.is_dir() if label == "CIF directory" else path.is_file()
        if not expected:
            raise FileNotFoundError(f"{label} not found: {path}")

    mmseqs_version = legacy.mmseqs_version(args.mmseqs)
    logger.log(f"MMseqs2 version: {mmseqs_version}")
    source_exclusions = legacy.load_exclusion_ids(exclusion_xlsx)
    frozen_rmsd_exclusions = read_id_list(rmsd_exclusion_list)
    rank1 = load_rank1_records(rank1_csv)
    entries, cif_files = parse_all_entries(cif_dir, source_exclusions, report_dir, logger)

    counts = {
        "source_cifs": len(cif_files),
        "source_exclusion_ids": len(source_exclusions),
        "included_pdbs": len(entries),
        "strict_rank1_targets": len(rank1),
        "frozen_rmsd_exclusions": len(frozen_rmsd_exclusions),
    }
    logger.log(f"Input counts: {counts}")
    if not args.skip_count_check:
        expected_counts = {
            "source_cifs": EXPECTED_SOURCE_CIFS,
            "source_exclusion_ids": EXPECTED_SOURCE_EXCLUSIONS,
            "included_pdbs": EXPECTED_INCLUDED_PDBS,
            "strict_rank1_targets": EXPECTED_RANK1_TARGETS,
            "frozen_rmsd_exclusions": EXPECTED_RMSD_EXCLUSIONS,
        }
        if counts != expected_counts:
            raise ValueError(f"Input-count audit failed: actual={counts}, expected={expected_counts}")
    unexpected_rank1 = set(rank1) - set(entries)
    if unexpected_rank1:
        raise ValueError(f"rank1_targets.csv contains PDBs outside the curated 2241: {sorted(unexpected_rank1)}")
    unexpected_frozen = frozen_rmsd_exclusions - set(rank1)
    if unexpected_frozen:
        raise ValueError(f"RMSD exclusion list contains PDBs absent from rank1 table: {sorted(unexpected_frozen)}")

    date_rows = []
    entity_rows = []
    for pdb_id, entry in sorted(entries.items()):
        release = entry.release
        date_rows.append(
            {
                "PDB_ID": pdb_id,
                "RELEASE_DATE": release.release_date.isoformat(),
                "SELECTED_SOURCE": release.selected_source,
                "REVISION_ORDINAL": release.revision_ordinal,
                "REVISION_DATE": release.revision_date,
                "LEGACY_ORIGINAL_DATE": release.legacy_original_date,
                "SOURCES_AGREE": release.sources_agree,
            }
        )
        for entity in entry.entities:
            entity_rows.append(
                {
                    "PDB_ID": pdb_id,
                    "ENTITY_ID": entity.entity_id,
                    "CHAIN_IDS": ",".join(entity.chain_ids),
                    "SEQUENCE": entity.sequence,
                    "SEARCH_SEQUENCE": entity.search_sequence,
                    "LENGTH": len(entity.search_sequence),
                    "AMBIGUOUS_BASES": entity.search_sequence.count("N"),
                    "RELEASE_DATE": entity.release_date.isoformat(),
                    "EXPERIMENT_METHOD": entity.experiment_method,
                    "RESOLUTION_ANGSTROM": entity.resolution,
                }
            )
    legacy.write_tsv(report_dir / "date_audit.tsv", legacy.REPORT_COLUMNS["date_audit.tsv"], date_rows)
    legacy.write_tsv(report_dir / "entity_metadata.tsv", legacy.REPORT_COLUMNS["entity_metadata.tsv"], entity_rows)

    assignments, selection_rows, selection_statuses = select_entries(
        entries,
        rank1,
        frozen_rmsd_exclusions,
        args.train_end,
        args.val_end,
        args.train_rmsd_max,
    )
    legacy.write_tsv(report_dir / "selection_audit.tsv", SELECTION_COLUMNS, selection_rows)
    selected_counts = Counter(assignments.values())
    logger.log(f"Selection statuses: {dict(selection_statuses)}")
    logger.log(f"Selected before test homology/de-redundancy: {dict(selected_counts)}")

    for pdb_id in assignments:
        entry = entries[pdb_id]
        if len(entry.entities) != 1 or len(entry.entities[0].chain_ids) != 1:
            raise ValueError(f"Single-chain invariant failed for selected PDB {pdb_id}")

    work_dir = report_dir / "mmseqs_work"
    work_dir.mkdir()
    reference_ids = {pdb_id for pdb_id, split in assignments.items() if split in {"train", "val"}}
    initial_test_ids = {pdb_id for pdb_id, split in assignments.items() if split == "test"}
    reference_entities = [entries[pdb_id].entities[0] for pdb_id in sorted(reference_ids)]
    test_entities = [entries[pdb_id].entities[0] for pdb_id in sorted(initial_test_ids)]

    reference_hits = run_pair_search(
        query_entities=test_entities,
        target_entities=reference_entities,
        query_fasta=work_dir / "test_entities.fasta",
        target_fasta=work_dir / "train_val_entities.fasta",
        output_tsv=work_dir / "test_vs_train_val.raw.tsv",
        temporary_dir=work_dir / "tmp_reference",
        args=args,
        logger=logger,
        exclude_same_pdb=False,
    )
    legacy.write_tsv(
        report_dir / "reference_redundancy_hits.tsv",
        HIT_COLUMNS,
        (legacy.hit_row(hit) for hit in reference_hits),
    )
    reference_masked_ids = {hit.query_pdb_id for hit in reference_hits}
    remaining_ids = initial_test_ids - reference_masked_ids
    remaining_entities = [entries[pdb_id].entities[0] for pdb_id in sorted(remaining_ids)]

    internal_hits = run_pair_search(
        query_entities=remaining_entities,
        target_entities=remaining_entities,
        query_fasta=work_dir / "remaining_test_entities.fasta",
        target_fasta=work_dir / "remaining_test_entities_target.fasta",
        output_tsv=work_dir / "remaining_test_internal.raw.tsv",
        temporary_dir=work_dir / "tmp_internal",
        args=args,
        logger=logger,
        exclude_same_pdb=True,
    )
    legacy.write_tsv(
        report_dir / "test_internal_hits.tsv",
        HIT_COLUMNS,
        (legacy.hit_row(hit) for hit in internal_hits),
    )
    remaining_keys = {legacy.entity_key(entity) for entity in remaining_entities}
    components = legacy.connected_components(remaining_keys, internal_hits)
    representative_for: dict[tuple[str, str], tuple[str, str]] = {}
    cluster_for: dict[tuple[str, str], str] = {}
    representatives: set[tuple[str, str]] = set()
    for index, component in enumerate(components, start=1):
        representative = min(component, key=lambda key: legacy.representative_key(key, entries))
        representatives.add(representative)
        cluster_id = f"TEST_ENTITY_CLUSTER_{index:04d}"
        for key in component:
            representative_for[key] = representative
            cluster_for[key] = cluster_id
    final_test_ids = {pdb_id for pdb_id, _ in representatives}
    internal_masked_ids = remaining_ids - final_test_ids

    best_reference: dict[tuple[str, str], legacy.Hit] = {}
    for hit in sorted(
        reference_hits,
        key=lambda item: (-item.identity, -item.query_coverage, -item.target_coverage, item.target_pdb_id),
    ):
        best_reference.setdefault(legacy.hit_query_key(hit), hit)

    test_rows: list[dict[str, Any]] = []
    mask_json: dict[str, Any] = {
        "schema_version": "1.0-single-chain",
        "selection_unit": "complete single-chain PDB entry",
        "pdbs": {},
    }
    for pdb_id in sorted(initial_test_ids):
        entity = entries[pdb_id].entities[0]
        key = legacy.entity_key(entity)
        match = best_reference.get(key)
        representative = representative_for.get(key)
        if pdb_id in reference_masked_ids:
            status = "MASK_REFERENCE_HOMOLOG"
            reason = "Single RNA chain is homologous to final train/val"
            evaluate = False
        elif pdb_id in internal_masked_ids:
            status = "MASK_INTERNAL_REDUNDANT"
            reason = "Non-representative member of a test homology component"
            evaluate = False
        else:
            status = "EVALUATE"
            reason = ""
            evaluate = True
        chain_id = entity.chain_ids[0]
        rep_pdb, rep_entity = representative if representative else ("", "")
        test_rows.append(
            {
                "PDB_ID": pdb_id,
                "CHAIN_ID": chain_id,
                "ENTITY_ID": entity.entity_id,
                "RELEASE_DATE": entity.release_date.isoformat(),
                "SEQUENCE_LENGTH": len(entity.search_sequence),
                "STRICT_RANK1_RMSD_ANGSTROM": rank1[pdb_id].rmsd,
                "EVALUATE": evaluate,
                "CHAIN_STATUS": status,
                "REASON": reason,
                "MATCH_PDB_ID": match.target_pdb_id if match else "",
                "MATCH_ENTITY_ID": match.target_entity_id if match else "",
                "IDENTITY": f"{match.identity:.6f}" if match else "",
                "QUERY_COVERAGE": f"{match.query_coverage:.6f}" if match else "",
                "TARGET_COVERAGE": f"{match.target_coverage:.6f}" if match else "",
                "ALIGNMENT_SOURCE": match.alignment_source if match else "",
                "INTERNAL_CLUSTER_ID": cluster_for.get(key, ""),
                "REPRESENTATIVE_PDB_ID": rep_pdb,
                "REPRESENTATIVE_ENTITY_ID": rep_entity,
            }
        )
        mask_json["pdbs"][pdb_id] = {
            "release_date": entity.release_date.isoformat(),
            "chain_id": chain_id,
            "entity_id": entity.entity_id,
            "strict_rank1_rmsd_angstrom": rank1[pdb_id].rmsd,
            "evaluate": evaluate,
            "status": status,
            "match_pdb_id": match.target_pdb_id if match else None,
            "internal_cluster_id": cluster_for.get(key) or None,
            "representative_pdb_id": rep_pdb or None,
        }
    test_columns = [
        "PDB_ID", "CHAIN_ID", "ENTITY_ID", "RELEASE_DATE", "SEQUENCE_LENGTH",
        "STRICT_RANK1_RMSD_ANGSTROM", "EVALUATE", "CHAIN_STATUS", "REASON",
        "MATCH_PDB_ID", "MATCH_ENTITY_ID", "IDENTITY", "QUERY_COVERAGE",
        "TARGET_COVERAGE", "ALIGNMENT_SOURCE", "INTERNAL_CLUSTER_ID",
        "REPRESENTATIVE_PDB_ID", "REPRESENTATIVE_ENTITY_ID",
    ]
    legacy.write_tsv(report_dir / "test_chain_evaluation.tsv", test_columns, test_rows)
    legacy.write_json(report_dir / "test_evaluation_mask.json", mask_json)

    expected = {
        "train": {pdb_id for pdb_id, split in assignments.items() if split == "train"},
        "val": {pdb_id for pdb_id, split in assignments.items() if split == "val"},
        "test": final_test_ids,
    }
    selection_by_id = {row["PDB_ID"]: row for row in selection_rows}
    manifest: list[dict[str, Any]] = []
    for pdb_id, entry in sorted(entries.items()):
        selection = selection_by_id[pdb_id]
        selected_split = assignments.get(pdb_id, "")
        if selected_split in {"train", "val"}:
            final_split = selected_split
            final_status = "KEPT"
            reason = ""
        elif selected_split == "test" and pdb_id in final_test_ids:
            final_split = "test"
            final_status = "KEPT"
            reason = ""
        elif selected_split == "test" and pdb_id in reference_masked_ids:
            final_split = ""
            final_status = "DROP_REFERENCE_HOMOLOG"
            reason = "Single RNA chain is homologous to final train/val"
        elif selected_split == "test":
            final_split = ""
            final_status = "DROP_INTERNAL_REDUNDANT"
            reason = "Non-representative member of a test homology component"
        else:
            final_split = ""
            final_status = selection["SELECTION_STATUS"]
            reason = selection["SELECTION_REASON"]
        chain_id = (
            entries[pdb_id].entities[0].chain_ids[0]
            if entry_chain_count(entries[pdb_id]) == 1
            else ""
        )
        manifest.append(
            {
                "PDB_ID": pdb_id,
                "RELEASE_DATE": entry.release.release_date.isoformat(),
                "INITIAL_SPLIT": selection["INITIAL_SPLIT"],
                "STRICT_RANK1_RMSD_ANGSTROM": selection["STRICT_RANK1_RMSD_ANGSTROM"],
                "SELECTION_STATUS": selection["SELECTION_STATUS"],
                "FINAL_SPLIT": final_split,
                "FINAL_STATUS": final_status,
                "EXCLUSION_REASON": reason,
                "RNA_CHAIN_ID": chain_id,
                "TARGET_DIRECTORY": str(data_dir / final_split / pdb_id.lower()) if final_split else "",
            }
        )
    legacy.write_tsv(report_dir / "final_manifest.tsv", MANIFEST_COLUMNS, manifest)

    actions = legacy.materialize_empty_directories(data_dir, expected, args.execute)
    legacy.write_tsv(report_dir / "mkdir_actions.tsv", legacy.REPORT_COLUMNS["mkdir_actions.tsv"], actions)
    stats_rows = distribution_rows(expected, rank1, entries)
    stats_columns = [
        "SPLIT", "PDB_COUNT", "RMSD_MEAN", "RMSD_Q25", "RMSD_MEDIAN",
        "RMSD_Q75", "RMSD_P90", "RMSD_MAX", "RNA_LENGTH_MEDIAN", "RNA_LENGTH_P90",
    ]
    legacy.write_tsv(report_dir / "distribution_summary.tsv", stats_columns, stats_rows)
    if not args.skip_plots:
        plot_distributions(report_dir, expected, rank1, entries, args.train_rmsd_max)

    final_counts = {split: len(ids) for split, ids in expected.items()}
    summary = {
        "status": "SUCCESS",
        "pipeline_version": PIPELINE_VERSION,
        "mode": "EXECUTE" if args.execute else "DRYRUN",
        "input_counts": counts,
        "date_boundaries": {
            "train": f"release_date <= {args.train_end.isoformat()}",
            "val": f"{(args.train_end + date.resolution).isoformat()} <= release_date <= {args.val_end.isoformat()}",
            "test": f"release_date > {args.val_end.isoformat()}",
        },
        "selection_rules": {
            "rna_chain_count": "exactly 1",
            "strict_rank1_rmsd_required": True,
            "train_rmsd_max_angstrom_inclusive": args.train_rmsd_max,
            "rmsd_cutoff_applies_to": "train only",
            "directory_case": "lowercase",
        },
        "selection_status_counts": dict(sorted(selection_statuses.items())),
        "selected_before_test_filtering": dict(sorted(selected_counts.items())),
        "homology_thresholds": {
            "minimum_sequence_identity": args.min_seq_id,
            "minimum_query_coverage": args.min_query_cov,
            "minimum_target_coverage": args.min_target_cov,
            "test_reference": "final train + val",
            "test_internal_redundancy": True,
        },
        "test_filtering": {
            "initial_test_pdbs": len(initial_test_ids),
            "reference_homologs_removed": len(reference_masked_ids),
            "remaining_before_internal_dedup": len(remaining_ids),
            "internal_redundant_pdbs_removed": len(internal_masked_ids),
            "final_test_pdbs": len(final_test_ids),
            "internal_clusters": len(components),
        },
        "final_directory_counts": final_counts,
        "distribution_summary": stats_rows,
        "mmseqs_version": mmseqs_version,
        "report_directory": str(report_dir),
        "data_directory": str(data_dir),
    }
    legacy.write_json(report_dir / "summary.json", summary)
    legacy.write_json(
        report_dir / "input_checksums.json",
        {
            "exclusion_xlsx": {"path": str(exclusion_xlsx), "sha256": legacy.sha256_file(exclusion_xlsx)},
            "rank1_csv": {"path": str(rank1_csv), "sha256": legacy.sha256_file(rank1_csv)},
            "rmsd_exclusion_list": {"path": str(rmsd_exclusion_list), "sha256": legacy.sha256_file(rmsd_exclusion_list)},
        },
    )
    logger.log(f"Final directory counts: {final_counts}")
    logger.log(f"Report directory: {report_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = make_report_dir(
        args.report_root.expanduser().resolve(), args.execute, args.train_rmsd_max
    )
    logger = legacy.RunLogger(report_dir / "pipeline.log")
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": legacy.sha256_file(Path(__file__).resolve()),
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: (
                value.isoformat()
                if isinstance(value, date)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
    }
    legacy.write_json(report_dir / "run_config.json", config)
    logger.log(f"Report directory: {report_dir}")
    logger.log("Mode: " + ("EXECUTE (create empty directories)" if args.execute else "DRYRUN (no Data_V1 writes)"))
    try:
        summary = run_pipeline(args, report_dir, logger)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        logger.log(f"FAILED: {type(exc).__name__}: {exc}")
        with (report_dir / "traceback.txt").open("w", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        legacy.write_json(
            report_dir / "failure.json",
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "report_directory": str(report_dir),
            },
        )
        print(f"FAILED; see {report_dir}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
