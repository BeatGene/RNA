#!/usr/bin/env python3
"""Correct policy-excluded rows in an existing V2 lifecycle audit, without rescanning PT files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_pdb_lifecycle import FIELDS, policy_file_exclusions, tsv_rows


def correct(source: Path, policy_file: Path, output: Path) -> dict:
    source_rows = tsv_rows(source / "pdb_lifecycle.tsv")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    policy = policy_file_exclusions(policy_file)
    if len(source_rows) != source_summary["source_pdbs"]:
        raise ValueError("lifecycle row count differs from its summary")
    if set(source_rows[0]) != set(FIELDS):
        raise ValueError("unexpected lifecycle columns")
    if len({row["PDB_ID"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate lifecycle PDB_ID")
    if len(policy) != 60:
        raise ValueError(f"expected 60 V2 policy entries, found {len(policy)}")
    source_by_id = {row["PDB_ID"].upper(): row for row in source_rows}
    if {key[1] for key in policy} - set(source_by_id):
        raise ValueError("PT policy contains PDBs absent from lifecycle audit")
    selected_policy = {
        key: reason for key, reason in policy.items()
        if source_by_id[key[1]]["FINAL_SPLIT"]
    }
    if len(selected_policy) != 56:
        raise ValueError(f"expected 56 selected V2 PT policy exclusions, found {len(selected_policy)}")
    for key in set(policy) - set(selected_policy):
        if source_by_id[key[1]]["NEXT_STAGE"] != "NOT_ASSIGNED":
            raise ValueError(f"unselected PT policy PDB has unexpected stage: {key}")
    matched = set()
    for row in source_rows:
        key = (row["FINAL_SPLIT"].lower(), row["PDB_ID"].upper())
        if key not in selected_policy:
            if row["PT_POLICY_EXCLUDED"] == "True":
                raise ValueError(f"unexpected policy exclusion in source audit: {key}")
            continue
        if row["NEXT_STAGE"] != "PT_PARTIAL_OR_FAILED":
            raise ValueError(f"policy PDB has unexpected original stage: {key}: {row['NEXT_STAGE']}")
        if row["PT_COUNT"] != "0" or row["PT_RMSD_GT30_SAVED_COUNT"] != "0":
            raise ValueError(f"policy PDB already has PT files: {key}")
        if row["UPSTREAM_SKIP"] == "True" or row["PT_UNUSABLE_STATUS"]:
            raise ValueError(f"policy PDB overlaps another exception: {key}")
        row["PT_POLICY_EXCLUDED"] = "True"
        row["PT_POLICY_REASON"] = selected_policy[key]
        row["NEXT_STAGE"] = "PT_EXCLUDED_BY_POLICY"
        row["EVIDENCE_NOTE"] = selected_policy[key]
        matched.add(key)
    if matched != set(selected_policy):
        raise ValueError(f"policy PDBs absent or in wrong split: {sorted(set(selected_policy) - matched)}")

    original_counts = Counter(row["NEXT_STAGE"] for row in tsv_rows(source / "pdb_lifecycle.tsv"))
    if dict(original_counts) != source_summary["stage_counts"]:
        raise ValueError("original lifecycle stage counts differ from its summary")
    corrected_counts = Counter(row["NEXT_STAGE"] for row in source_rows)
    if corrected_counts.get("PT_PARTIAL_OR_FAILED", 0):
        raise ValueError("unexpected PT_PARTIAL_OR_FAILED rows remain")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pdb_lifecycle.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(source_rows)
    summary = dict(source_summary)
    summary["stage_counts"] = dict(corrected_counts)
    summary["policy_correction"] = {
        "source_lifecycle_dir": str(source.resolve()),
        "policy_file": str(policy_file.resolve()),
        "corrected_rows": len(matched),
        "method": "reclassified original lifecycle rows using authoritative PT exclusion TSV; PT files were not rescanned",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--policy-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = correct(args.source_dir.expanduser(), args.policy_file.expanduser(), args.output_dir.expanduser())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
