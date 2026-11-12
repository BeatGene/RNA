#!/usr/bin/env python3
"""Plan or create V2 hard links to eligible V1 PT files without writing V1.

Run after the new V2 train PT build. The two ID lists also drive the separate
new-prediction and retained-PDB PT build passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "val", "test")
SAMPLE_RE = re.compile(r"sample_([0-3])\.pt\Z", re.IGNORECASE)
SEED_RE = re.compile(r"seed_(3\d\d)\Z", re.IGNORECASE)


def rows(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def selected(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows(path / "final_manifest.tsv"):
        if row["FINAL_STATUS"] != "KEPT":
            continue
        pdb_id, split = row["PDB_ID"].upper(), row["FINAL_SPLIT"]
        if pdb_id in result or split not in SPLITS:
            raise ValueError(f"invalid selected PDB: {pdb_id} {split}")
        result[pdb_id] = split
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-split-report", required=True, type=Path)
    parser.add_argument("--new-split-report", required=True, type=Path)
    parser.add_argument("--old-pt-root", required=True, type=Path)
    parser.add_argument("--old-pt-run", required=True, type=Path)
    parser.add_argument("--new-pt-root", required=True, type=Path)
    parser.add_argument("--exclude-pdb-file", required=True, type=Path)
    parser.add_argument("--skip-pdb-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-hardlinks", action="store_true")
    args = parser.parse_args()
    old_pt = args.old_pt_root.expanduser().resolve(strict=True)
    old_run = json.loads((args.old_pt_run.expanduser() / "summary.json").read_text())
    old_signature = (
        old_run.get("mode"), old_run.get("schema_version"),
        old_run.get("generator_version"), old_run.get("mapping_policy"),
        old_run.get("max_pre_refinement_rmsd"),
        old_run.get("successful_samples"), old_run.get("rmsd_filtered_samples"),
    )
    if old_signature != (
        "write", 2, "2.4-v1-filter-policy",
        "complete-canonical-sequence-ccd-parent-residue-grouped-v2",
        30.0, 182232, 3168,
    ) or Path(old_run.get("output_root", "")).resolve() != old_pt:
        raise ValueError(f"unexpected V1 PT run; refusing reuse: {old_signature}")
    new_pt_arg = args.new_pt_root.expanduser()
    if new_pt_arg.is_symlink():
        raise ValueError("new PT root must not be a symlink")
    new_pt = new_pt_arg.resolve()
    if old_pt == new_pt or old_pt in new_pt.parents or new_pt in old_pt.parents:
        raise ValueError("V1 and V2 PT roots must be separate")
    old = selected(args.old_split_report.expanduser())
    new = selected(args.new_split_report.expanduser())
    if any(old[pdb_id] != new[pdb_id] for pdb_id in old.keys() & new.keys()):
        raise ValueError("a retained PDB changed split")
    excluded = {row["pdb_id"].upper() for row in rows(args.exclude_pdb_file.expanduser())}
    skipped = {row["PDB_ID"].upper() for row in rows(args.skip_pdb_file.expanduser())}
    if excluded & skipped:
        raise ValueError(f"PT exclude/skip overlap: {sorted(excluded & skipped)}")
    retained = old.keys() & new.keys()
    new_ids = new.keys() - old.keys()
    new_train = sorted(pdb_id for pdb_id in new_ids if pdb_id not in excluded | skipped)
    retained_eligible = sorted(pdb_id for pdb_id in retained if pdb_id not in excluded | skipped)
    expected = (len(old), len(new), len(retained), len(new_train), len(retained_eligible))
    if expected != (988, 1015, 964, 40, 908):
        raise ValueError(f"unexpected V2 PT population: {expected}")
    if any(new[pdb_id] != "train" for pdb_id in new_train):
        raise ValueError("new eligible PT targets should all be train PDBs")
    if skipped & new.keys() != {"3J28", "3J29", "3J2A", "3J2B", "3J2D", "3J2E", "3J2F", "3J2G", "3J2H", "7ZFW", "9R82"}:
        raise ValueError("V2 upstream-skip membership changed")

    report = args.output_dir.expanduser()
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage_started = {"scan_old_pt": started}

    def progress(stage: str, done: int, total: int, detail: str = "") -> None:
        if done == 0:
            stage_started[stage] = time.monotonic()
        fraction = done / total if total else 1.0
        filled = int(20 * fraction)
        elapsed = time.monotonic() - stage_started[stage]
        eta = int(elapsed / done * (total - done)) if done else None
        line = (f"PROGRESS {stage} [{'#' * filled}{'-' * (20 - filled)}] "
                f"{100 * fraction:.1f}% {done}/{total} elapsed={int(elapsed)}s "
                f"eta_approx={eta if eta is not None else '?'}s {detail}")
        print(line, flush=True)
        with (report / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    planned: list[tuple[Path, Path]] = []
    per_pdb: list[dict[str, object]] = []
    blockers: list[str] = []
    already_linked = 0
    source_total = 0
    progress("scan_old_pt", 0, len(retained_eligible))
    for index, pdb_id in enumerate(retained_eligible, 1):
        split = new[pdb_id]
        source_dir = old_pt / split / pdb_id.lower()
        destination_dir = new_pt / split / pdb_id.lower()
        if destination_dir.is_symlink():
            blockers.append(f"V2 PDB PT directory is a symlink: {destination_dir}")
            if index % 25 == 0 or index == len(retained_eligible):
                progress("scan_old_pt", index, len(retained_eligible), f"files={source_total}")
            continue
        files = sorted(source_dir.glob("seed_*/sample_*.pt")) if source_dir.is_dir() else []
        for source in files:
            if (source.is_symlink() or not source.is_file()
                    or not SEED_RE.fullmatch(source.parent.name)
                    or not SAMPLE_RE.fullmatch(source.name)):
                blockers.append(f"invalid V1 PT path: {source}")
                continue
            destination = destination_dir / source.parent.name / source.name
            source_total += 1
            if destination.exists() or destination.is_symlink():
                if destination.is_file() and destination.samefile(source):
                    already_linked += 1
                else:
                    blockers.append(f"V2 PT destination already differs: {destination}")
            else:
                planned.append((source, destination))
        per_pdb.append({"split": split, "pdb_id": pdb_id, "old_main_pt_count": len(files)})
        if index % 25 == 0 or index == len(retained_eligible):
            progress("scan_old_pt", index, len(retained_eligible), f"files={source_total}")

    (report / "new_train_pdb_ids.txt").write_text("\n".join(new_train) + "\n", encoding="utf-8")
    (report / "retained_eligible_pdb_ids.txt").write_text("\n".join(retained_eligible) + "\n", encoding="utf-8")
    with (report / "retained_pt_counts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "pdb_id", "old_main_pt_count"), delimiter="\t")
        writer.writeheader()
        writer.writerows(per_pdb)
    if blockers:
        (report / "blockers.txt").write_text("\n".join(blockers) + "\n", encoding="utf-8")
    if args.execute_hardlinks and blockers:
        raise RuntimeError(f"{len(blockers)} blockers; see {report / 'blockers.txt'}")
    created = 0
    if args.execute_hardlinks:
        new_pt.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            (new_pt / split).mkdir(exist_ok=True)
        progress("link_pt", 0, len(planned))
        for source, destination in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            created += 1
            if created % 1000 == 0 or created == len(planned):
                progress("link_pt", created, len(planned))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "EXECUTE_HARDLINKS" if args.execute_hardlinks else "DRYRUN",
        "old_selected": len(old), "new_selected": len(new),
        "retained_selected": len(retained),
        "new_train_eligible": len(new_train),
        "retained_eligible": len(retained_eligible),
        "selected_upstream_skipped": len(skipped & new.keys()),
        "selected_pt_policy_excluded": len(excluded & new.keys()),
        "old_main_pt_files": source_total,
        "already_hardlinked": already_linked,
        "planned_hardlinks": len(planned),
        "created_hardlinks": created,
        "blockers": len(blockers),
        "old_pt_root": str(old_pt), "new_pt_root": str(new_pt),
        "old_pt_run": str(args.old_pt_run.expanduser()),
        "report_directory": str(report),
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
