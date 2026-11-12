#!/usr/bin/env python3
"""Audit V2 prep skips and V1 50x4 prediction reuse; optionally link verified PDBs.

The default mode is read-only apart from writing reports. --execute-links replaces
only empty V2 PDB directories with symlinks to verified complete V1 directories.
It never writes into, deletes, or moves any V1 prediction directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "val", "test")
EXPECTED_SEEDS = range(300, 350)
EXPECTED_SAMPLES = {0, 1, 2, 3}
PATTERNS = {
    "cif": re.compile(r"_sample_(\d+)\.cif$", re.I),
    "summary": re.compile(r"_summary_confidence_sample_(\d+)\.json$", re.I),
    "full": re.compile(r"_full_data_sample_(\d+)\.json$", re.I),
}


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def selected(path: Path) -> dict[str, str]:
    result = {}
    for row in read_rows(path / "final_manifest.tsv", "\t"):
        if row["FINAL_STATUS"] != "KEPT":
            continue
        pdb_id = row["PDB_ID"].strip().upper()
        split = row["FINAL_SPLIT"].strip().lower()
        if pdb_id in result or split not in SPLITS:
            raise ValueError(f"Invalid or duplicate selected PDB: {pdb_id}, {split}")
        result[pdb_id] = split
    return result


def read_ids(path: Path) -> set[str]:
    ids = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.split("#", 1)[0].split("\t", 1)[0].strip().upper()
        if value == "PDB_ID":
            continue
        if value:
            if value in ids:
                raise ValueError(f"Duplicate ID in {path}: {value}")
            ids.add(value)
    return ids


def old_prediction_rows(run_dir: Path) -> dict[str, dict[str, str]]:
    result = {}
    for split in SPLITS:
        files = sorted((run_dir / split / "rounds").glob("*/decoy_manifest.csv"))
        if not files:
            raise FileNotFoundError(f"No old prediction audit for {split} in {run_dir}")
        for path in files:
            for row in read_rows(path):
                pdb_id = row["PDB_ID"].strip().upper()
                result[pdb_id] = row
    return result


def old_log_complete(row: dict[str, str] | None) -> bool:
    return bool(
        row
        and row["OVERALL_STATUS"] == "COMPLETE"
        and row["COMPLETE_SEED_COUNT"] == "50"
        and row["VALID_DECOY_COUNT"] == "200"
        and row["EXPECTED_DECOY_COUNT"] == "200"
    )


def sample_ids(pred_dir: Path, kind: str) -> set[int] | None:
    pattern = PATTERNS[kind]
    suffix = "*.cif" if kind == "cif" else "*.json"
    found = []
    for path in pred_dir.glob(suffix):
        match = pattern.search(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            found.append(int(match.group(1)))
    return set(found) if len(found) == len(set(found)) else None


def check_old_files(pdb_dir: Path) -> tuple[bool, str]:
    if not pdb_dir.is_dir():
        return False, "old PDB directory missing"
    for seed in EXPECTED_SEEDS:
        pred_dir = pdb_dir / f"seed_{seed}" / "predictions"
        if not pred_dir.is_dir():
            return False, f"seed_{seed} predictions directory missing"
        for kind in PATTERNS:
            if sample_ids(pred_dir, kind) != EXPECTED_SAMPLES:
                return False, f"seed_{seed} {kind} samples not exactly 0..3"
    return True, "50 seeds x 4 CIF/summary/full-data files present"


def destination_status(destination: Path, source: Path) -> str:
    if destination.is_symlink():
        return "ALREADY_LINKED" if destination.resolve() == source.resolve() else "WRONG_SYMLINK"
    if destination.is_dir():
        return "READY_EMPTY" if not any(destination.iterdir()) else "DEST_NOT_EMPTY"
    return "DEST_MISSING" if not destination.exists() else "DEST_NOT_DIRECTORY"


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-split-report", type=Path, required=True)
    parser.add_argument("--new-split-report", type=Path, required=True)
    parser.add_argument("--prep-audit", type=Path, required=True)
    parser.add_argument("--abandoned-file", type=Path, required=True)
    parser.add_argument("--old-pred-run", type=Path, required=True)
    parser.add_argument("--old-data-root", type=Path, required=True)
    parser.add_argument("--new-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-links", action="store_true")
    parser.add_argument("--max-links", type=int, default=0,
                        help="with --execute-links, link only this many PDBs (0 means all)")
    args = parser.parse_args()
    if args.max_links < 0 or (args.max_links and not args.execute_links):
        parser.error("--max-links must be nonnegative and requires --execute-links")
    old = selected(args.old_split_report.expanduser())
    new = selected(args.new_split_report.expanduser())
    abandoned = read_ids(args.abandoned_file.expanduser())
    prep_rows = {r["PDB_ID"].strip().upper(): r for r in read_rows(args.prep_audit.expanduser())}
    if set(prep_rows) != set(new):
        raise ValueError("Prep audit PDB set differs from the selected V2 PDB set")
    missing_prep = {
        pdb_id for pdb_id, row in prep_rows.items()
        if row["PREP_STATUS"] not in {"COMPLETE", "COMPLETE_REBASABLE"}
    }
    selected_abandoned = abandoned & set(new)
    if missing_prep != selected_abandoned or len(abandoned) != 14 or len(missing_prep) != 10:
        raise ValueError(
            f"Prep skip mismatch: missing={sorted(missing_prep)}, "
            f"selected_abandoned={sorted(selected_abandoned)}"
        )
    if any(new.get(pdb_id) != old.get(pdb_id) for pdb_id in set(new) & set(old)):
        raise ValueError("V1 and V2 retained PDBs must remain in the same split")
    audits = old_prediction_rows(args.old_pred_run.expanduser())
    retained = set(new) & set(old)
    assert len(retained) == 964 and len(new) == 1015

    result_rows: list[dict[str, object]] = []
    old_root = args.old_data_root.expanduser().resolve()
    new_root = args.new_data_root.expanduser().resolve()
    if old_root == new_root:
        raise ValueError("Old and new data roots must differ")
    for pdb_id in sorted(new):
        split = new[pdb_id]
        source = old_root / split / pdb_id.lower()
        destination = new_root / split / pdb_id.lower()
        row: dict[str, object] = {
            "PDB_ID": pdb_id,
            "SPLIT": split,
            "PREP_STATUS": prep_rows[pdb_id]["PREP_STATUS"],
            "OLD_PRED_AUDIT_STATUS": "",
            "OLD_FILES_STATUS": "",
            "V2_DIRECTORY_STATUS": "",
            "ACTION": "",
            "DETAIL": "",
            "OLD_DIRECTORY": str(source) if pdb_id in retained else "",
            "V2_DIRECTORY": str(destination),
        }
        if pdb_id in missing_prep:
            row["ACTION"] = "SKIP_PREP_ABANDONED_LONG_CHAIN"
            row["DETAIL"] = "V2 split assignment retained; no prediction or PT task"
        elif pdb_id not in retained:
            row["ACTION"] = "NEW_PREDICTION_REQUIRED"
            row["DETAIL"] = "No V1 50x4 prediction run membership"
        else:
            log_ok = old_log_complete(audits.get(pdb_id))
            files_ok, file_detail = check_old_files(source) if log_ok else (False, "old prediction audit incomplete")
            dest_status = destination_status(destination, source)
            row["OLD_PRED_AUDIT_STATUS"] = "COMPLETE_50X4" if log_ok else "INCOMPLETE_OR_MISSING"
            row["OLD_FILES_STATUS"] = "COMPLETE_50X4" if files_ok else "INCOMPLETE"
            row["V2_DIRECTORY_STATUS"] = dest_status
            row["DETAIL"] = file_detail
            row["ACTION"] = (
                "REUSE_ALREADY_LINKED" if log_ok and files_ok and dest_status == "ALREADY_LINKED"
                else "REUSE_READY_TO_LINK" if log_ok and files_ok and dest_status == "READY_EMPTY"
                else "REVIEW_BEFORE_PREDICTION"
            )
        result_rows.append(row)
        if len(result_rows) % 100 == 0:
            print(f"Audited {len(result_rows)}/{len(new)} V2 PDBs", flush=True)

    counts = Counter(str(row["ACTION"]) for row in result_rows)
    blockers = [row["PDB_ID"] for row in result_rows if row["ACTION"] == "REVIEW_BEFORE_PREDICTION"]
    if counts["SKIP_PREP_ABANDONED_LONG_CHAIN"] != 10 or counts["NEW_PREDICTION_REQUIRED"] != 41:
        raise ValueError(f"Unexpected V2 operational counts: {dict(counts)}")
    if counts["REUSE_READY_TO_LINK"] + counts["REUSE_ALREADY_LINKED"] + len(blockers) != 964:
        raise ValueError(f"Unexpected V1 reuse counts: {dict(counts)}")
    if args.execute_links and blockers:
        raise ValueError(f"Refusing to link: retained PDBs need review: {blockers[:20]}")
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    linked_ids: list[str] = []
    if args.execute_links:
        ready_rows = [row for row in result_rows if row["ACTION"] == "REUSE_READY_TO_LINK"]
        if args.max_links:
            ready_rows = ready_rows[:args.max_links]
        for row in ready_rows:
            source = Path(str(row["OLD_DIRECTORY"]))
            destination = Path(str(row["V2_DIRECTORY"]))
            if destination_status(destination, source) != "READY_EMPTY":
                raise RuntimeError(f"Destination changed after audit: {destination}")
            destination.rmdir()  # Empty directory only; never recursively delete.
            try:
                destination.symlink_to(source, target_is_directory=True)
                primary_count = sum(
                    bool(PATTERNS["cif"].search(path.name))
                    for path in destination.rglob("*.cif")
                )
                if destination_status(destination, source) != "ALREADY_LINKED" or primary_count != 200:
                    raise RuntimeError(
                        f"Linked prediction traversal failed: {destination}, "
                        f"primary CIF count={primary_count}"
                    )
            except Exception:
                if destination.is_symlink():
                    destination.unlink()
                destination.mkdir(exist_ok=True)
                raise
            row["ACTION"] = "REUSE_LINKED"
            row["V2_DIRECTORY_STATUS"] = "ALREADY_LINKED"
            linked_ids.append(str(row["PDB_ID"]))
            if len(linked_ids) % 100 == 0:
                print(f"Linked {len(linked_ids)}/{len(ready_rows)} verified V1 PDBs", flush=True)
        counts = Counter(str(row["ACTION"]) for row in result_rows)

    fields = list(result_rows[0])
    write_tsv(output / "pdb_prediction_plan.tsv", result_rows, fields)
    for split in SPLITS:
        worklist = [
            row for row in result_rows
            if row["SPLIT"] == split and row["ACTION"] == "NEW_PREDICTION_REQUIRED"
        ]
        (output / f"predict_{split}_pdb_ids.txt").write_text(
            "".join(f"{row['PDB_ID']}\n" for row in worklist), encoding="utf-8"
        )
    (output / "predict_all_pdb_ids.txt").write_text(
        "".join(
            f"{row['PDB_ID']}\n" for row in result_rows
            if row["ACTION"] == "NEW_PREDICTION_REQUIRED"
        ),
        encoding="utf-8",
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "EXECUTE_LINKS" if args.execute_links else "DRYRUN",
        "max_links": args.max_links if args.execute_links else None,
        "linked_ids_this_run": linked_ids,
        "new_selected": len(new),
        "retained_from_v1": len(retained),
        "prep_abandoned_selected": len(missing_prep),
        "prep_abandoned_ids": sorted(missing_prep),
        "actions": dict(counts),
        "review_blockers": blockers,
        "old_data_root": str(old_root),
        "new_data_root": str(new_root),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
