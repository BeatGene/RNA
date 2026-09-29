#!/usr/bin/env python3
"""Reconcile every V2 prediction CIF with main/high PT files and build logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
CIF_RE = re.compile(r"^.+_sample_(\d+)\.cif$", re.IGNORECASE)
SEED_RE = re.compile(r"^seed_(\d+)$", re.IGNORECASE)
PT_RE = re.compile(r"^sample_(\d+)\.pt$", re.IGNORECASE)
FIELDS = (
    "split", "pdb_id", "status", "prediction_cif_count", "main_pt_count",
    "high_rmsd_pt_count", "total_pt_count", "rmsd_gt30_logged_count",
    "missing_prediction_count", "missing_pt_count", "pt_issue_count",
    "unexpected_pt_count", "cause_counts", "all_200_accounted",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, records: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def pt_keys(root: Path, split: str, pdb_id: str) -> set[tuple[int, int]]:
    pdb_dir = root / split / pdb_id.lower()
    found: set[tuple[int, int]] = set()
    if not pdb_dir.is_dir():
        return found
    for seed_dir in pdb_dir.glob("seed_*"):
        match = SEED_RE.fullmatch(seed_dir.name)
        if not match or not seed_dir.is_dir():
            continue
        seed = int(match.group(1))
        for path in seed_dir.glob("sample_*.pt"):
            sample = PT_RE.fullmatch(path.name)
            if sample and path.is_file():
                found.add((seed, int(sample.group(1))))
    return found


def prediction_keys(root: Path, split: str, pdb_id: str) -> set[tuple[int, int]]:
    pdb_dir = root / split / pdb_id.lower()
    found: set[tuple[int, int]] = set()
    if not pdb_dir.is_dir():
        return found
    for seed_dir in pdb_dir.glob("seed_*"):
        match = SEED_RE.fullmatch(seed_dir.name)
        if not match or not seed_dir.is_dir():
            continue
        seed = int(match.group(1))
        for predictions_dir in seed_dir.rglob("predictions"):
            if not predictions_dir.is_dir():
                continue
            for path in predictions_dir.glob("*_sample_*.cif"):
                sample = CIF_RE.fullmatch(path.name)
                if sample and path.is_file() and not path.name.lower().endswith("_wounresol.cif"):
                    found.add((seed, int(sample.group(1))))
    return found


def run_rows(run_dirs: list[Path], name: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for run in run_dirs:
        output.extend(read_tsv(run / name))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--main-pt-root", required=True, type=Path)
    parser.add_argument("--high-pt-root", required=True, type=Path)
    parser.add_argument("--exclude-pdb-file", required=True, type=Path)
    parser.add_argument("--skip-pdb-file", required=True, type=Path)
    parser.add_argument("--new-pt-run", required=True, type=Path)
    parser.add_argument("--retained-pt-run", required=True, type=Path)
    parser.add_argument("--old-pt-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = [args.prediction_root, args.main_pt_root, args.high_pt_root]
    for root in roots:
        if not root.expanduser().is_dir():
            raise FileNotFoundError(root)
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def progress(stage: str, done: int, total: int, detail: str = "") -> None:
        fraction = done / total if total else 1.0
        filled = int(fraction * 20)
        elapsed = time.monotonic() - started
        eta = int(elapsed / done * (total - done)) if done else None
        line = (f"PROGRESS {stage} [{'#' * filled}{'-' * (20 - filled)}] "
                f"{fraction * 100:.1f}% {done}/{total} elapsed={int(elapsed)}s "
                f"eta_approx={eta if eta is not None else '?'}s {detail}")
        print(line, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    progress("load_pt_logs", 0, 1)
    selected = {
        row["PDB_ID"].upper(): row["FINAL_SPLIT"]
        for row in read_tsv(args.split_report.expanduser() / "final_manifest.tsv")
        if row["FINAL_STATUS"] == "KEPT"
    }
    if len(selected) != 1015:
        raise ValueError(f"expected 1015 V2 selected PDBs, got {len(selected)}")
    excluded = {row["pdb_id"].upper() for row in read_tsv(args.exclude_pdb_file.expanduser())}
    skipped = {
        row["PDB_ID"].upper(): row["REASON"]
        for row in read_tsv(args.skip_pdb_file.expanduser())
    }
    run_dirs = [args.new_pt_run.expanduser(), args.retained_pt_run.expanduser()]
    main_rows = run_rows(run_dirs, "manifest.tsv")
    filtered_rows = run_rows(run_dirs, "filtered_samples.tsv")
    issue_rows = run_rows(run_dirs, "issues.tsv")
    progress("load_pt_logs", 1, 1,
             f"main={len(main_rows)} high={len(filtered_rows)} issues={len(issue_rows)}")
    main_log: dict[tuple[str, str, int, int], dict[str, str]] = {}
    filtered_log: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for records, destination in ((main_rows, main_log), (filtered_rows, filtered_log)):
        for row in records:
            key = (row["split"], row["pdb_id"].upper(), int(row["seed"]), int(row["sample"]))
            if selected.get(key[1]) != key[0]:
                raise ValueError(f"PT log sample is not assigned to this V2 split: {key}")
            if key in destination:
                raise ValueError(f"duplicate PT log candidate: {key}")
            destination[key] = row
    if main_log.keys() & filtered_log.keys():
        raise ValueError("PT sample logged in both main and high-RMSD categories")
    sample_issues: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    pdb_issues: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in issue_rows:
        base = (row["split"], row["pdb_id"].upper())
        match = re.fullmatch(r"seed_(\d+)/sample_(\d+)", row["sample"])
        if match:
            sample_issues[(*base, int(match.group(1)), int(match.group(2)))].append(row["error"])
        else:
            pdb_issues[base].append(row["error"])

    expected_keys = {(seed, sample) for seed in range(300, 350) for sample in range(4)}
    high_by_pdb: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for split, pdb_id, seed, sample in filtered_log:
        high_by_pdb[(split, pdb_id)].add((seed, sample))
    sample_issues_by_pdb = Counter(key[:2] for key in sample_issues)
    pdb_records: list[dict] = []
    missing_records: list[dict] = []
    unexpected_records: list[dict] = []
    progress("audit_pdb", 0, len(selected))
    for index, (pdb_id, split) in enumerate(sorted(selected.items(), key=lambda item: (item[1], item[0])), 1):
        base = (split, pdb_id)
        pred = prediction_keys(args.prediction_root.expanduser(), split, pdb_id)
        main = pt_keys(args.main_pt_root.expanduser(), split, pdb_id)
        high = pt_keys(args.high_pt_root.expanduser(), split, pdb_id)
        logged_high = high_by_pdb[base]
        causes: Counter[str] = Counter()
        if pdb_id in skipped:
            status = "PRED_OOM_SKIPPED" if "OOM" in skipped[pdb_id] else "PREP_SKIPPED"
        elif pdb_id in excluded:
            status = "PT_POLICY_EXCLUDED"
        else:
            for seed, sample in sorted(expected_keys - pred):
                cause = "PRED_MISSING"
                causes[cause] += 1
                missing_records.append({"split": split, "pdb_id": pdb_id, "seed": seed,
                                        "sample": sample, "stage": "prediction", "reason": cause})
            for seed, sample in sorted(pred - main - high):
                key = (*base, seed, sample)
                if key in sample_issues:
                    cause = "PT_BUILD_ERROR"
                    detail = " | ".join(sample_issues[key])
                elif base in pdb_issues:
                    cause = "PT_PDB_ERROR"
                    detail = " | ".join(pdb_issues[base])
                elif key in filtered_log:
                    cause = "RMSD_GT30_PT_NOT_SAVED"
                    detail = "see filtered_samples.tsv"
                else:
                    cause = "PT_MISSING_UNEXPLAINED"
                    detail = "no matching PT issue or filtered row"
                causes[cause] += 1
                missing_records.append({"split": split, "pdb_id": pdb_id, "seed": seed,
                                        "sample": sample, "stage": "pt", "reason": cause,
                                        "detail": detail})
            for seed, sample in sorted((main | high) - pred):
                unexpected_records.append({"split": split, "pdb_id": pdb_id,
                                           "seed": seed, "sample": sample,
                                           "reason": "PT_WITHOUT_PREDICTION_CIF"})
            for seed, sample in sorted(main & high):
                unexpected_records.append({"split": split, "pdb_id": pdb_id,
                                           "seed": seed, "sample": sample,
                                           "reason": "PT_IN_BOTH_MAIN_AND_HIGH"})
            for seed, sample in sorted(high - logged_high):
                unexpected_records.append({"split": split, "pdb_id": pdb_id,
                                           "seed": seed, "sample": sample,
                                           "reason": "HIGH_PT_WITHOUT_RMSD_LOG"})
            if pred != expected_keys:
                status = "PRED_INCOMPLETE"
            elif causes["PT_BUILD_ERROR"] or causes["PT_PDB_ERROR"]:
                status = "PT_BUILD_ERROR"
            elif (len(main | high) == 200 and not ((main | high) - pred)
                  and not (main & high) and high == logged_high):
                status = "ALL_200_PT_ACCOUNTED"
            else:
                status = "PT_INCOMPLETE"
        pdb_records.append({
            "split": split, "pdb_id": pdb_id, "status": status,
            "prediction_cif_count": len(pred), "main_pt_count": len(main),
            "high_rmsd_pt_count": len(high), "total_pt_count": len(main | high),
            "rmsd_gt30_logged_count": len(logged_high),
            "missing_prediction_count": len(expected_keys - pred),
            "missing_pt_count": len(pred - main - high),
            "pt_issue_count": sample_issues_by_pdb[base] + len(pdb_issues.get(base, [])),
            "unexpected_pt_count": len((main | high) - pred) + len(main & high) + len(high - logged_high),
            "cause_counts": json.dumps(causes, ensure_ascii=False, sort_keys=True),
            "all_200_accounted": status == "ALL_200_PT_ACCOUNTED",
        })
        if index % 25 == 0 or index == len(selected):
            progress("audit_pdb", index, len(selected), f"current={split}/{pdb_id}")

    combined = output / "combined_pt_log"
    combined.mkdir()
    write_tsv(combined / "manifest.tsv", main_rows, tuple(main_rows[0]) if main_rows else ("split", "pdb_id", "seed", "sample"))
    write_tsv(combined / "filtered_samples.tsv", filtered_rows, tuple(filtered_rows[0]) if filtered_rows else ("split", "pdb_id", "seed", "sample"))
    write_tsv(output / "per_pdb.tsv", pdb_records, FIELDS)
    write_tsv(output / "missing_samples.tsv", missing_records,
              ("split", "pdb_id", "seed", "sample", "stage", "reason", "detail"))
    write_tsv(output / "unexpected_samples.tsv", unexpected_records,
              ("split", "pdb_id", "seed", "sample", "reason"))

    old_high_rows = read_tsv(args.old_pt_run.expanduser() / "filtered_samples.tsv")
    old_recovery: list[dict] = []
    recovery_cache: dict[tuple[str, str], tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {}
    progress("audit_old_gt30", 0, len(old_high_rows))
    for index, row in enumerate(old_high_rows, 1):
        split, pdb_id = row["split"], row["pdb_id"].upper()
        if selected.get(pdb_id) != split or pdb_id in excluded or pdb_id in skipped:
            continue
        seed, sample = int(row["seed"]), int(row["sample"])
        base = (split, pdb_id)
        if base not in recovery_cache:
            recovery_cache[base] = (
                pt_keys(args.main_pt_root.expanduser(), split, pdb_id),
                pt_keys(args.high_pt_root.expanduser(), split, pdb_id),
            )
        main, high = recovery_cache[base]
        if (seed, sample) in high:
            recovery = "RECOVERED_HIGH_PT"
        elif (seed, sample) in main:
            recovery = "NOW_MAIN_PT_CHECK_RMSD"
        else:
            recovery = "NOT_RECOVERED"
        old_recovery.append({"split": split, "pdb_id": pdb_id, "seed": seed,
                             "sample": sample, "old_rmsd": row["pre_refinement_aligned_rmsd"],
                             "recovery_status": recovery})
        if index % 500 == 0 or index == len(old_high_rows):
            progress("audit_old_gt30", index, len(old_high_rows),
                     f"relevant={len(old_recovery)}")
    progress("audit_old_gt30", len(old_high_rows), len(old_high_rows),
             f"relevant={len(old_recovery)}")
    write_tsv(output / "old_gt30_recovery.tsv", old_recovery,
              ("split", "pdb_id", "seed", "sample", "old_rmsd", "recovery_status"))
    summary = {
        "selected_pdb_count": len(selected),
        "pdb_status_counts": dict(Counter(row["status"] for row in pdb_records)),
        "prediction_cif_count": sum(row["prediction_cif_count"] for row in pdb_records),
        "main_pt_count": sum(row["main_pt_count"] for row in pdb_records),
        "high_rmsd_pt_count": sum(row["high_rmsd_pt_count"] for row in pdb_records),
        "missing_reason_counts": dict(Counter(row["reason"] for row in missing_records)),
        "old_gt30_recovery_counts": dict(Counter(row["recovery_status"] for row in old_recovery)),
        "combined_pt_log": str(combined),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
