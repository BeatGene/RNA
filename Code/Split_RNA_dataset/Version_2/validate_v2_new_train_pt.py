#!/usr/bin/env python3
"""Accept only the six audited, full-PDB V2 new-train PT failures."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_STATUS = {
    "1QZA": "SPARSE_NATIVE_COORDINATES",
    "1QZB": "SPARSE_NATIVE_COORDINATES",
    "2Z9Q": "SPARSE_NATIVE_COORDINATES",
    "5K8H": "SPARSE_NATIVE_COORDINATES",
    "2JJA": "UNRESOLVED_MODIFIED_RESIDUES",
    "2XC6": "UNRESOLVED_MODIFIED_RESIDUES",
}
EXPECTED_KEYS = {(seed, sample) for seed in range(300, 350) for sample in range(4)}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def issue_matches(status: str, error: str) -> bool:
    if status == "SPARSE_NATIVE_COORDINATES":
        match = re.fullmatch(
            r"ValueError: observed native atom fraction ([0-9.]+) is below 0\.500",
            error,
        )
        return match is not None and float(match.group(1)) < 0.5
    return error.startswith((
        "ValueError: no exact residue-grouped predicted/RNA-FM sequence match;",
        "ValueError: unsupported predicted RNA residues;",
    )) and ("ZHP" in error or "ZTH" in error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-pt-run", required=True, type=Path)
    parser.add_argument("--new-pdb-ids", required=True, type=Path)
    parser.add_argument("--unusable-file", required=True, type=Path)
    args = parser.parse_args()
    run = args.new_pt_run.expanduser()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    ids = {
        line.strip().upper()
        for line in args.new_pdb_ids.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    unusable_rows = read_tsv(args.unusable_file.expanduser())
    unusable = {row["PDB_ID"].upper(): row for row in unusable_rows}
    if (len(ids), len(unusable), set(unusable)) != (40, 6, set(EXPECTED_STATUS)):
        raise ValueError("new train or unusable PDB membership differs from audited population")
    for pdb_id, row in unusable.items():
        if pdb_id not in ids or row["STATUS"] != EXPECTED_STATUS[pdb_id] or not row["REASON"]:
            raise ValueError(f"invalid unusable PT policy row: {row}")
    signature = (
        summary["mode"], summary["discovered_samples"], summary["failed_samples"],
        summary["issues"], summary["successful_samples"],
        summary["high_rmsd_saved_samples"], summary["rmsd_filtered_samples"],
    )
    if signature != ("write", 8000, 1200, 1200, 5620, 1180, 1180):
        raise ValueError(f"new-train PT run has unexpected accounting: {signature}")

    issues = defaultdict(dict)
    for row in read_tsv(run / "issues.tsv"):
        pdb_id = row["pdb_id"].upper()
        match = re.fullmatch(r"seed_(\d+)/sample_(\d+)", row["sample"])
        if row["split"] != "train" or pdb_id not in unusable or match is None:
            raise ValueError(f"unexpected PT issue: {row}")
        key = (int(match.group(1)), int(match.group(2)))
        if key not in EXPECTED_KEYS or key in issues[pdb_id]:
            raise ValueError(f"duplicate or invalid PT issue: {row}")
        if not issue_matches(unusable[pdb_id]["STATUS"], row["error"]):
            raise ValueError(f"PT issue does not match audited cause: {row}")
        issues[pdb_id][key] = row["error"]
    if any(set(issues[pdb_id]) != EXPECTED_KEYS for pdb_id in unusable):
        raise ValueError("each unusable PDB must have exactly 200 audited sample issues")

    accounted = defaultdict(set)
    status_counts: Counter[str] = Counter()
    for filename, category in (("manifest.tsv", "normal"),
                               ("filtered_samples.tsv", "high")):
        for row in read_tsv(run / filename):
            pdb_id = row["pdb_id"].upper()
            key = (int(row["seed"]), int(row["sample"]))
            if row["split"] != "train" or pdb_id not in ids - set(unusable):
                raise ValueError(f"unexpected PT output population: {row}")
            if key not in EXPECTED_KEYS or key in accounted[pdb_id]:
                raise ValueError(f"duplicate or invalid PT output: {row}")
            output_key = "output" if category == "normal" else "high_rmsd_output"
            if not Path(row[output_key]).is_file():
                raise FileNotFoundError(row[output_key])
            if category == "high" and row["high_rmsd_status"] not in {"CREATED", "SKIPPED"}:
                raise ValueError(f"high-RMSD PT was not saved: {row}")
            accounted[pdb_id].add(key)
            status_counts[category] += 1
    if any(accounted[pdb_id] != EXPECTED_KEYS for pdb_id in ids - set(unusable)):
        raise ValueError("each usable new PDB must have exactly 200 saved PT files")
    if status_counts != {"normal": 5620, "high": 1180}:
        raise ValueError(f"saved PT counts do not match run summary: {status_counts}")

    result = {
        "validated_new_pdbs": len(ids),
        "complete_new_pdbs": len(ids - set(unusable)),
        "unusable_new_pdbs": len(unusable),
        "unusable_ids": sorted(unusable),
        "normal_pt": status_counts["normal"],
        "high_rmsd_pt": status_counts["high"],
        "documented_missing_pt": sum(len(rows) for rows in issues.values()),
        "new_pt_run": str(run),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
