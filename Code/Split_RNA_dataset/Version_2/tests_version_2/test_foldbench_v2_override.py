import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Code.Split_RNA_dataset.Version_2.audit_pdb_lifecycle import FIELDS


VERSION = Path(__file__).resolve().parents[1]
WORKSPACE = VERSION.parents[2]
SPLIT_REPORT = WORKSPACE / "Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"
SCRIPT = VERSION / "apply_v2_foldbench_test_override.py"
PROMOTE = ("7SXP", "7WIA", "7WII", "7ZJ4", "8HB8")
HOLDOUT = ("7WI9", "7WIB", "7WIE", "7WIF", "7ZJ5")


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class FoldbenchV2OverrideTests(unittest.TestCase):
    def test_dry_run_execute_and_resume_preserve_source_and_hardlink_high_test(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            pred, pt, high = (base / name for name in ("Data_V2", "Data_PT_V2", "Data_PT_V2_RMSD_GT30"))
            for root in (pred, pt, high):
                for split in ("train", "val", "test"):
                    (root / split).mkdir(parents=True)
            manifest = read_tsv(SPLIT_REPORT / "final_manifest.tsv")
            assigned = {row["PDB_ID"]: row["FINAL_SPLIT"] for row in manifest if row["FINAL_SPLIT"]}
            for pdb_id, split in assigned.items():
                (pred / split / pdb_id.lower()).mkdir()
            for pdb_id in PROMOTE + HOLDOUT:
                for seed in range(300, 350):
                    for sample in range(4):
                        name = pdb_id.lower()
                        cif_dir = pred / "val" / name / f"seed_{seed}" / "predictions"
                        cif_dir.mkdir(parents=True, exist_ok=True)
                        (cif_dir / f"{name}_seed_{seed}_sample_{sample}.cif").write_bytes(b"CIF")
                        root = high if pdb_id == "7ZJ4" else pt
                        pt_dir = root / "val" / name / f"seed_{seed}"
                        pt_dir.mkdir(parents=True, exist_ok=True)
                        (pt_dir / f"sample_{sample}.pt").write_bytes(b"PT")
            audit = base / "audit"
            audit_rows = []
            for pdb_id, split in assigned.items():
                audited = pdb_id in PROMOTE + HOLDOUT
                audit_rows.append({
                    "pdb_id": pdb_id, "split": split,
                    "status": "ALL_200_PT_ACCOUNTED" if audited else "UNUSED_IN_TEST_FIXTURE",
                    "prediction_cif_count": "200" if audited else "0",
                    "total_pt_count": "200" if audited else "0",
                })
            write_tsv(audit / "per_pdb.tsv", list(audit_rows[0]), audit_rows)
            # All ten affected PDBs must be among exactly 942 complete rows.
            complete = set(PROMOTE + HOLDOUT) | set(
                pdb_id for pdb_id in sorted(assigned) if pdb_id not in PROMOTE + HOLDOUT
            )
            complete = set(PROMOTE + HOLDOUT) | set(sorted(complete - set(PROMOTE + HOLDOUT))[:932])
            lifecycle = []
            for source in read_tsv(SPLIT_REPORT / "source_inventory.tsv"):
                pdb_id = source["PDB_ID"]
                row = dict.fromkeys(FIELDS, "")
                split = assigned.get(pdb_id, "")
                row.update({
                    "PDB_ID": pdb_id, "FINAL_SPLIT": split,
                    "FINAL_STATUS": "KEPT" if split else "NOT_ASSIGNED",
                    "NEXT_STAGE": "PT_COMPLETE_BY_COUNT" if pdb_id in complete else
                                  "STAGE_UNKNOWN" if split else "NOT_ASSIGNED",
                    "PT_COUNT": "0" if pdb_id == "7ZJ4" else "200" if pdb_id in complete else "",
                    "PT_RMSD_GT30_SAVED_COUNT": "200" if pdb_id == "7ZJ4" else "0",
                    "PT_ACCOUNTED_COUNT": "200" if pdb_id in complete else "",
                })
                lifecycle.append(row)
            write_tsv(audit / "lifecycle_policy_corrected/pdb_lifecycle.tsv", list(FIELDS), lifecycle)
            report = base / "report"
            command = [
                sys.executable, str(SCRIPT), "--report-dir", str(report),
                "--split-report", str(SPLIT_REPORT), "--pt-audit", str(audit),
                "--prediction-root", str(pred), "--pt-root", str(pt),
                "--high-pt-root", str(high),
            ]
            subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertFalse((pred / "test/7zj4").exists())
            execute = [sys.executable, str(SCRIPT), "--report-dir", str(report), "--execute"]
            subprocess.run(execute, capture_output=True, text=True, check=True)
            subprocess.run(execute, capture_output=True, text=True, check=True)
            summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["amended_selected_counts"], {"train": 823, "val": 108, "test": 79})
            self.assertEqual(summary["lifecycle"]["stage_counts"]["PT_COMPLETE_BY_COUNT"], 937)
            mask = json.loads((report / "test_evaluation_mask.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(row["evaluate"] for row in mask["pdbs"].values()), 79)
            self.assertEqual(mask["pdbs"]["7WII"]["internal_cluster_id"],
                             mask["pdbs"]["7WIA"]["internal_cluster_id"])
            self.assertTrue((pred / "test/7wia").is_dir())
            self.assertTrue((pred / "holdout_foldbench_val_homology/7wi9").is_dir())
            self.assertFalse((pred / "val/7wi9").exists())
            self.assertTrue(os.path.samefile(
                pt / "test/7zj4/seed_300/sample_0.pt",
                high / "test/7zj4/seed_300/sample_0.pt",
            ))
            self.assertEqual(len(list((pt / "test/7zj4").rglob("*.pt"))), 200)
            self.assertEqual(len(read_tsv(SPLIT_REPORT / "final_manifest.tsv")), 2241)


if __name__ == "__main__":
    unittest.main()
