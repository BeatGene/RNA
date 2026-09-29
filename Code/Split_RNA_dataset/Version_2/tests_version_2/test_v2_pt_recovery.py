import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


VERSION = Path(__file__).resolve().parents[1]
WORKSPACE = VERSION.parents[2]


def tsv(path: Path, fields: list[str], records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


class V2PtRecoveryTests(unittest.TestCase):
    def test_hardlink_plan_reuses_only_selected_eligible_old_pt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_pt = root / "Data_PT_V1"
            retained = old_pt / "train/17ra/seed_300/sample_0.pt"
            removed = old_pt / "test/10zu/seed_300/sample_0.pt"
            for path in (retained, removed):
                path.parent.mkdir(parents=True)
                path.write_bytes(path.name.encode())
            old_run = root / "old_run"
            old_run.mkdir()
            (old_run / "summary.json").write_text(json.dumps({
                "mode": "write", "schema_version": 2,
                "generator_version": "2.4-v1-filter-policy",
                "mapping_policy": "complete-canonical-sequence-ccd-parent-residue-grouped-v2",
                "max_pre_refinement_rmsd": 30.0,
                "successful_samples": 182232, "rmsd_filtered_samples": 3168,
                "output_root": str(old_pt),
            }), encoding="utf-8")
            new_pt = root / "Data_PT_V2"
            command = [sys.executable, str(VERSION / "plan_v2_pt_reuse.py"),
                       "--old-split-report", str(WORKSPACE / "Code/New_Data_pipeline_reports/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE"),
                       "--new-split-report", str(WORKSPACE / "Code/New_Data_pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"),
                       "--old-pt-root", str(old_pt), "--old-pt-run", str(old_run),
                       "--new-pt-root", str(new_pt),
                       "--exclude-pdb-file", str(WORKSPACE / "Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv"),
                       "--skip-pdb-file", str(VERSION / "pt_upstream_skipped_v2.tsv")]
            subprocess.run(command + ["--output-dir", str(root / "plan")], check=True,
                           stdout=subprocess.DEVNULL)
            self.assertFalse(new_pt.exists())
            result = subprocess.run(command + ["--output-dir", str(root / "execute"),
                                               "--execute-hardlinks"], check=True,
                                    capture_output=True, text=True)
            summary = json.loads((root / "execute/summary.json").read_text(encoding="utf-8"))
            self.assertIn("PROGRESS link_pt", result.stdout)
            self.assertEqual(summary["created_hardlinks"], 1)
            self.assertTrue((new_pt / "train/17ra/seed_300/sample_0.pt").samefile(retained))
            self.assertFalse((new_pt / "test/10zu").exists())
            self.assertEqual(retained.read_bytes(), b"sample_0.pt")

    def test_audit_separates_saved_high_rmsd_from_missing_pt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            ids = [f"X{i:04d}" for i in range(1015)]
            tsv(split / "final_manifest.tsv", ["PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"],
                [{"PDB_ID": name, "FINAL_SPLIT": "train", "FINAL_STATUS": "KEPT"} for name in ids])
            skip = root / "skip.tsv"
            tsv(skip, ["PDB_ID", "REASON"],
                [{"PDB_ID": name, "REASON": "PREP_SKIP"} for name in ids[1:]])
            exclude = root / "exclude.tsv"
            tsv(exclude, ["pdb_id", "split", "reason"], [])
            pred, main_pt, high_pt = (root / name for name in ("pred", "main", "high"))
            for directory in (pred, main_pt, high_pt):
                directory.mkdir()
            main_rows, high_rows = [], []
            for seed in range(300, 350):
                for sample in range(4):
                    folder = pred / f"train/x0000/seed_{seed}/predictions"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"x0000_sample_{sample}.cif").write_text("cif", encoding="utf-8")
                    key = {"split": "train", "pdb_id": "X0000", "seed": seed,
                           "sample": sample, "pre_refinement_aligned_rmsd": "31"}
                    if seed == 349 and sample == 3:
                        high_rows.append(key)
                        target = high_pt / f"train/x0000/seed_{seed}/sample_{sample}.pt"
                    else:
                        main_rows.append(key)
                        target = main_pt / f"train/x0000/seed_{seed}/sample_{sample}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"pt")
            runs = [root / "new_run", root / "old_run"]
            for directory in runs:
                directory.mkdir()
            fields = list(main_rows[0])
            tsv(runs[0] / "manifest.tsv", fields, main_rows)
            tsv(runs[0] / "filtered_samples.tsv", fields, high_rows)
            tsv(runs[0] / "issues.tsv", ["split", "pdb_id", "sample", "error"], [])
            for name in ("manifest.tsv", "filtered_samples.tsv"):
                tsv(runs[1] / name, fields, [])
            tsv(runs[1] / "issues.tsv", ["split", "pdb_id", "sample", "error"], [])
            prior = root / "prior"
            prior.mkdir()
            tsv(prior / "filtered_samples.tsv", fields, high_rows)
            command = [sys.executable, str(VERSION / "audit_v2_pt_samples.py"),
                       "--split-report", str(split), "--prediction-root", str(pred),
                       "--main-pt-root", str(main_pt), "--high-pt-root", str(high_pt),
                       "--exclude-pdb-file", str(exclude), "--skip-pdb-file", str(skip),
                       "--new-pt-run", str(runs[0]), "--retained-pt-run", str(runs[1]),
                       "--old-pt-run", str(prior)]
            subprocess.run(command + ["--output-dir", str(root / "report1")], check=True,
                           stdout=subprocess.DEVNULL)
            with (root / "report1/per_pdb.tsv").open(encoding="utf-8") as handle:
                target = next(row for row in csv.DictReader(handle, delimiter="\t") if row["pdb_id"] == "X0000")
            self.assertEqual(target["status"], "ALL_200_PT_ACCOUNTED")
            self.assertEqual(target["high_rmsd_pt_count"], "1")
            (high_pt / "train/x0000/seed_349/sample_3.pt").unlink()
            subprocess.run(command + ["--output-dir", str(root / "report2")], check=True,
                           stdout=subprocess.DEVNULL)
            with (root / "report2/missing_samples.tsv").open(encoding="utf-8") as handle:
                missing = [row for row in csv.DictReader(handle, delimiter="\t") if row["pdb_id"] == "X0000"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["reason"], "RMSD_GT30_PT_NOT_SAVED")


if __name__ == "__main__":
    unittest.main()
