#!/usr/bin/env python3
"""Create a Protenix stage2 task manifest from a completed split report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"missing header: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--master-manifest", type=Path,
                        default=Path.home() / "Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv")
    parser.add_argument("--simple-json-dir", type=Path,
                        default=Path.home() / "Json_data/Simple_json")
    parser.add_argument("--complex-json-dir", type=Path,
                        default=Path.home() / "Json_data/Complex_json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    fields, master_rows = read_csv(args.master_manifest.expanduser(), ",")
    if not {"PDB_ID", "CURRENT_TARGET"}.issubset(fields):
        raise ValueError("master manifest needs PDB_ID and CURRENT_TARGET")
    _, split_rows = read_csv(args.split_report.expanduser() / "final_manifest.tsv", "\t")
    chosen = {}
    for row in split_rows:
        if row["FINAL_STATUS"].strip().upper() != "KEPT":
            continue
        pdb_id = row["PDB_ID"].strip().upper()
        if not pdb_id or pdb_id in chosen:
            raise ValueError(f"empty or duplicate PDB ID in split report: {pdb_id}")
        chosen[pdb_id] = row["FINAL_SPLIT"].strip().lower()
    if not chosen:
        raise ValueError("split report contains no KEPT PDBs")
    if any(split not in {"train", "val", "test"} for split in chosen.values()):
        raise ValueError("KEPT row has invalid FINAL_SPLIT")
    current = {}
    for row in master_rows:
        pdb_id = row["PDB_ID"].strip().upper()
        if row["CURRENT_TARGET"].strip().lower() in {"1", "true", "t", "yes", "y"}:
            if pdb_id in current:
                raise ValueError(f"duplicate current PDB in master manifest: {pdb_id}")
            current[pdb_id] = row
    missing = sorted(set(chosen) - set(current))
    if missing:
        raise ValueError(f"new split PDBs missing from master manifest: {missing[:20]}")
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "protenix_tasks.csv"
    with task_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(current[pdb_id] for pdb_id in sorted(chosen))
    simple = args.simple_json_dir.expanduser()
    complex_dir = args.complex_json_dir.expanduser()
    input_rows = []
    for pdb_id, split in sorted(chosen.items()):
        name = pdb_id.lower()
        input_rows.append({
            "PDB_ID": pdb_id,
            "FINAL_SPLIT": split,
            "RAW_JSON_PRESENT": (simple / f"{name}.json").is_file(),
            "UPDATED_JSON_PRESENT": (simple / f"{name}-final-updated.json").is_file(),
            "PREP_DIR_PRESENT": (complex_dir / f"prep_output_{name}").is_dir(),
        })
    with (output / "input_presence.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)
    summary = {
        "split_report": str(args.split_report.expanduser().resolve()),
        "task_manifest": str(task_path.resolve()),
        "selected_counts": dict(Counter(chosen.values())),
        "missing_raw_json": sum(not row["RAW_JSON_PRESENT"] for row in input_rows),
        "missing_updated_json": sum(not row["UPDATED_JSON_PRESENT"] for row in input_rows),
        "missing_prep_dir": sum(not row["PREP_DIR_PRESENT"] for row in input_rows),
        "note": "Presence is only a preflight; use stage2_decoy_pipeline.py audit for content validity.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
