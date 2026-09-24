#!/usr/bin/env python3
"""Read-only PDB lifecycle audit from a split report and optional server outputs.

The split manifest is the source of truth. Directory presence is reported as
evidence, never used to silently change a split assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
FIELDS = (
    "PDB_ID", "SOURCE_EXCLUDED", "RELEASE_DATE", "INITIAL_SPLIT", "FINAL_SPLIT",
    "FINAL_STATUS", "EXCLUSION_REASON", "RNA_LENGTH", "RNA_CHAIN_COUNT",
    "STRICT_RANK1_RMSD_ANGSTROM", "PREP_DIR_PRESENT", "PRED_CIF_COUNT",
    "PRED_AUDIT_STATUS", "PRED_EXPECTED_COUNT", "PRED_VALID_COUNT",
    "PREP_AUDIT_STATUS", "PREP_AUDIT_REASON",
    "PT_COUNT", "PT_ISSUE_COUNT", "PT_RMSD_GT30_COUNT", "PT_RMSD_GT30_SAVED_COUNT", "PT_RMSD_GT30_SOURCE",
    "PT_POLICY_EXCLUDED", "PT_POLICY_REASON", "PT_EXPECTED_COUNT", "PT_ACCOUNTED_COUNT",
    "NEXT_STAGE", "EVIDENCE_NOTE",
)


def tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def keyed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = row["PDB_ID"].strip().upper()
        if key in result:
            raise ValueError(f"duplicate PDB_ID: {key}")
        result[key] = row
    return result


def index_counts(root: Path | None, pattern: str, suffix: str) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if root is None:
        return counts
    if not root.is_dir():
        raise FileNotFoundError(root)
    for split in SPLITS:
        folder = root / split
        if not folder.is_dir():
            continue
        for pdb_dir in folder.iterdir():
            if not pdb_dir.is_dir():
                continue
            for path in pdb_dir.rglob(pattern):
                if path.is_file() and path.suffix == suffix:
                    if suffix == ".cif" and (
                        path.parent.name != "predictions"
                        or path.name.lower().endswith("_wounresol.cif")
                    ):
                        continue
                    counts[(split, pdb_dir.name.upper())] += 1
    return counts


def filtered_counts(log_dirs: list[Path]) -> Counter[tuple[str, str]]:
    # Multiple reruns can record the same candidate; count each once.
    seen: set[tuple[str, str, str, str]] = set()
    for directory in log_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("filtered_samples.tsv"):
            for row in tsv_rows(path):
                if row.get("reason") != "pre_refinement_aligned_rmsd_above_limit":
                    continue
                if float(row["pre_refinement_aligned_rmsd"]) <= 30:
                    continue
                seen.add((row["split"].lower(), row["pdb_id"].upper(), row["seed"], row["sample"]))
    counts: Counter[tuple[str, str]] = Counter()
    for split, pdb_id, _, _ in seen:
        counts[(split, pdb_id)] += 1
    return counts


def issue_counts(log_dirs: list[Path]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    seen_paths: set[Path] = set()
    for directory in log_dirs:
        for path in directory.rglob("issues.tsv"):
            if path.resolve() in seen_paths:
                continue
            seen_paths.add(path.resolve())
            for row in tsv_rows(path):
                if row.get("pdb_id"):
                    counts[(row["split"].lower(), row["pdb_id"].upper())] += 1
    return counts


def policy_exclusions(log_dirs: list[Path]) -> dict[tuple[str, str], str]:
    excluded: dict[tuple[str, str], str] = {}
    for directory in log_dirs:
        for path in directory.rglob("excluded_pdbs.tsv"):
            for row in tsv_rows(path):
                key = (row["split"].lower(), row["pdb_id"].upper())
                reason = row.get("reason", "")
                if key in excluded and excluded[key] != reason:
                    raise ValueError(f"conflicting PT exclusion reasons for {key}")
                excluded[key] = reason
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--pt-root", type=Path)
    parser.add_argument("--high-pt-root", type=Path)
    parser.add_argument("--prep-root", type=Path)
    parser.add_argument("--pred-audit", action="append", default=[], type=Path,
                        help="stage2 decoy_manifest.csv; pass once per split")
    parser.add_argument("--pt-log-dir", action="append", default=[], type=Path)
    args = parser.parse_args()
    report = args.split_report.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    inventory = keyed(tsv_rows(report / "source_inventory.tsv"))
    manifest = keyed(tsv_rows(report / "final_manifest.tsv"))
    selection = keyed(tsv_rows(report / "selection_audit.tsv"))
    if len(inventory) != 2246 or len(manifest) != 2241:
        raise ValueError(f"unexpected source/split rows: {len(inventory)}/{len(manifest)}")
    if set(manifest) - set(inventory) or set(selection) != set(manifest):
        raise ValueError("inventory, manifest and selection IDs disagree")

    preds = index_counts(args.prediction_root, "*_sample_*.cif", ".cif")
    pts = index_counts(args.pt_root, "*.pt", ".pt")
    high_pts = index_counts(args.high_pt_root, "*.pt", ".pt")
    high = filtered_counts(args.pt_log_dir)
    issues = issue_counts(args.pt_log_dir)
    pt_exclusions = policy_exclusions(args.pt_log_dir)
    pred_audit = {}
    for audit_path in args.pred_audit:
        with audit_path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["PDB_ID"].strip().upper()
                if key in pred_audit:
                    raise ValueError(f"duplicate PDB_ID in pred audit: {key}")
                pred_audit[key] = row
    prep_ids = None
    if args.prep_root is not None:
        prep_root = args.prep_root.expanduser().resolve()
        if not prep_root.is_dir():
            raise FileNotFoundError(prep_root)
        prep_ids = {
            path.name[len("prep_output_"):].upper()
            for path in prep_root.glob("prep_output_*") if path.is_dir()
        }
    rows = []
    stages = Counter()
    for pdb_id in sorted(inventory):
        source = inventory[pdb_id]
        assigned = manifest.get(pdb_id, {})
        choice = selection.get(pdb_id, {})
        split = assigned.get("FINAL_SPLIT", "").lower()
        pred_count = preds[(split, pdb_id)] if split else 0
        pt_count = pts[(split, pdb_id)] if split else 0
        high_pt_count = high_pts[(split, pdb_id)] if split else 0
        high_count = high[(split, pdb_id)] if split else 0
        issue_count = issues[(split, pdb_id)] if split else 0
        pt_policy_reason = pt_exclusions.get((split, pdb_id), "") if split else ""
        pred_row = pred_audit.get(pdb_id, {})
        audit_status = pred_row.get("OVERALL_STATUS", "")
        prep_status = pred_row.get("PREP_STATUS", "")
        pred_valid = pred_row.get("VALID_DECOY_COUNT", "")
        pred_expected = pred_row.get("EXPECTED_DECOY_COUNT", "")
        effective_pred_count = pred_count if args.prediction_root is not None else int(pred_valid or 0)
        pt_accounted_count = pt_count + high_pt_count
        prep_present = pdb_id in prep_ids if prep_ids is not None else None
        if not assigned:
            stage = "SOURCE_EXCLUDED"
            note = "listed in source exclusion XLSX (not in experimental pure RNA set)"
        elif not split:
            stage = "NOT_ASSIGNED"
            note = "see FINAL_STATUS and EXCLUSION_REASON"
        elif args.prediction_root is None and args.pt_root is None and prep_ids is None and not pred_audit:
            stage = "NOT_AUDITED"
            note = "server stage paths were not provided"
        elif audit_status == "NEED_PREP":
            stage = "PREP_PENDING_OR_FAILED"
            note = pred_row.get("PREP_REASON", "")
        elif audit_status == "NEED_PRED":
            stage = "PRED_PENDING_OR_FAILED"
            note = "see pred audit and seed manifest"
        elif args.prediction_root is not None and audit_status == "COMPLETE" and pred_count == 0:
            stage = "PRED_AUDIT_MISMATCH"
            note = "prediction audit says complete but no CIF was found under prediction root"
        elif audit_status == "NEED_JSON":
            stage = "JSON_PENDING_OR_FAILED"
            note = pred_row.get("RAW_JSON_REASON", "")
        elif pt_policy_reason:
            stage = "PT_EXCLUDED_BY_POLICY"
            note = pt_policy_reason
        elif args.pt_root is None:
            stage = "PT_NOT_AUDITED" if effective_pred_count else "STAGE_UNKNOWN"
            note = "supply PT root and PT log directory" if effective_pred_count else "supply prediction and PT stage evidence"
        elif effective_pred_count and effective_pred_count > pt_count + high_count:
            stage = "PT_PARTIAL_OR_FAILED"
            note = "prediction count exceeds main PT plus high-RMSD records; inspect PT issues"
        elif high_count > high_pt_count:
            stage = "PT_GT30_NOT_SAVED"
            note = "old PT policy filtered RMSD >30 A samples without saving them"
        elif effective_pred_count and pt_accounted_count == effective_pred_count:
            stage = "PT_COMPLETE_BY_COUNT"
            note = "all prediction samples have a PT file; inspect PT issues for candidate-level errors"
        elif pt_count or high_pt_count:
            stage = "PT_PRESENT_COUNT_UNVERIFIED"
            note = "PT files present; prediction count unavailable or inconsistent"
        elif effective_pred_count:
            stage = "PT_PENDING_OR_FAILED"
            note = "prediction CIF exists; inspect PT issues log"
        elif prep_present is False:
            stage = "PREP_PENDING_OR_FAILED"
            note = "prep directory missing; inspect prep logs"
        elif prep_present is True:
            stage = "PRED_PENDING_OR_FAILED"
            note = "prep directory exists; inspect pred logs"
        else:
            stage = "STAGE_UNKNOWN"
            note = "supply prep/pred/PT server paths"
        stages[stage] += 1
        rows.append({
            "PDB_ID": pdb_id,
            "SOURCE_EXCLUDED": source["EXCLUDED"],
            "RELEASE_DATE": assigned.get("RELEASE_DATE", ""),
            "INITIAL_SPLIT": assigned.get("INITIAL_SPLIT", ""),
            "FINAL_SPLIT": split,
            "FINAL_STATUS": assigned.get("FINAL_STATUS", "SOURCE_EXCLUDED"),
            "EXCLUSION_REASON": assigned.get("EXCLUSION_REASON", note if not assigned else ""),
            "RNA_LENGTH": choice.get("RNA_LENGTH", ""),
            "RNA_CHAIN_COUNT": choice.get("RNA_CHAIN_COUNT", ""),
            "STRICT_RANK1_RMSD_ANGSTROM": assigned.get("STRICT_RANK1_RMSD_ANGSTROM", ""),
            "PREP_DIR_PRESENT": "" if prep_present is None else str(prep_present),
            "PRED_CIF_COUNT": pred_count if args.prediction_root is not None and split else "",
            "PRED_AUDIT_STATUS": audit_status if split else "",
            "PRED_EXPECTED_COUNT": pred_expected if split else "",
            "PRED_VALID_COUNT": pred_valid if split else "",
            "PREP_AUDIT_STATUS": prep_status if split else "",
            "PREP_AUDIT_REASON": pred_row.get("PREP_REASON", "") if split else "",
            "PT_COUNT": pt_count if args.pt_root is not None and split else "",
            "PT_ISSUE_COUNT": issue_count if args.pt_log_dir and split else "",
            "PT_RMSD_GT30_COUNT": high_count if args.pt_log_dir and split else "",
            "PT_RMSD_GT30_SAVED_COUNT": high_pt_count if args.high_pt_root is not None and split else "",
            "PT_RMSD_GT30_SOURCE": (
                "filtered log and separate PT directory" if args.high_pt_root is not None
                else "filtered log only; old PT was not saved"
            ) if high_count or high_pt_count else "",
            "PT_POLICY_EXCLUDED": bool(pt_policy_reason) if split else "",
            "PT_POLICY_REASON": pt_policy_reason,
            "PT_EXPECTED_COUNT": effective_pred_count if split and (args.prediction_root is not None or pred_valid) else "",
            "PT_ACCOUNTED_COUNT": pt_accounted_count if split and args.pt_root is not None else "",
            "NEXT_STAGE": stage,
            "EVIDENCE_NOTE": note,
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pdb_lifecycle.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split_report": str(report), "source_pdbs": len(inventory),
        "assigned_counts": dict(Counter(row["FINAL_SPLIT"] for row in rows if row["FINAL_SPLIT"])),
        "stage_counts": dict(stages),
        "limitations": "Counts locate the next stage; use run logs for failure cause. Run the audit with matching split, prediction and PT versions. Old filtered RMSD >30 samples were not saved.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
