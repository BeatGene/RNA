import csv
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Code.Split_RNA_dataset.Version_2 import validate_v2_new_train_pt as validator


VERSION = Path(__file__).resolve().parents[1]
WORKSPACE = VERSION.parents[2]


def tsv(path: Path, fields: list[str], records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


class V2PtRecoveryTests(unittest.TestCase):
    def test_lifecycle_policy_correction_reclassifies_only_configured_rows(self):
        from Code.Split_RNA_dataset.Version_2.audit_pdb_lifecycle import FIELDS

        policy_file = WORKSPACE / "Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv"
        with policy_file.open(encoding="utf-8-sig", newline="") as handle:
            policy = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(policy), 60)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            rows = []
            unselected = {"8YAM", "8YAN", "9CSQ", "9CSR"}
            for item in policy:
                row = dict.fromkeys(FIELDS, "")
                row.update({
                    "PDB_ID": item["pdb_id"],
                    "FINAL_SPLIT": "" if item["pdb_id"] in unselected else item["split"],
                    "NEXT_STAGE": "NOT_ASSIGNED" if item["pdb_id"] in unselected else "PT_PARTIAL_OR_FAILED", "PT_COUNT": "0",
                    "PT_RMSD_GT30_SAVED_COUNT": "0", "UPSTREAM_SKIP": "False",
                    "PT_POLICY_EXCLUDED": "False",
                })
                rows.append(row)
            tsv(source / "pdb_lifecycle.tsv", list(FIELDS), rows)
            (source / "summary.json").write_text(json.dumps({
                "source_pdbs": 60, "stage_counts": {"PT_PARTIAL_OR_FAILED": 56, "NOT_ASSIGNED": 4}
            }), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(VERSION / "correct_v2_lifecycle_policy.py"),
                "--source-dir", str(source), "--policy-file", str(policy_file),
                "--output-dir", str(root / "corrected"),
            ], capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["stage_counts"], {"PT_EXCLUDED_BY_POLICY": 56, "NOT_ASSIGNED": 4})
            with (root / "corrected/pdb_lifecycle.tsv").open(encoding="utf-8") as handle:
                corrected = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(sum(row["PT_POLICY_EXCLUDED"] == "True" for row in corrected), 56)

    def test_lifecycle_keeps_six_unusable_pt_targets_in_train(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "lifecycle"
            result = subprocess.run([
                sys.executable, str(VERSION / "audit_pdb_lifecycle.py"),
                "--split-report", str(WORKSPACE / "Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"),
                "--pt-unusable-file", str(VERSION / "pt_unusable_new_train_v2.tsv"),
                "--output-dir", str(output),
            ], capture_output=True, text=True, check=True)
            self.assertIn("pt_unusable_assigned", result.stdout)
            with (output / "pdb_lifecycle.tsv").open(encoding="utf-8") as handle:
                rows = {row["PDB_ID"]: row for row in csv.DictReader(handle, delimiter="\t")}
            self.assertEqual(len(rows), 2246)
            for pdb_id, status in validator.EXPECTED_STATUS.items():
                self.assertEqual(rows[pdb_id]["FINAL_SPLIT"], "train")
                self.assertEqual(rows[pdb_id]["NEXT_STAGE"], "PT_UNUSABLE_" + status)

    def test_new_train_validator_accepts_only_six_audited_failure_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "new_run"
            run.mkdir()
            good_ids = [f"X{i:03d}" for i in range(34)]
            bad_ids = sorted(validator.EXPECTED_STATUS)
            ids = root / "new_ids.txt"
            ids.write_text("\n".join(good_ids + bad_ids) + "\n", encoding="utf-8")
            policy = root / "unusable.tsv"
            tsv(policy, ["PDB_ID", "STATUS", "REASON"], [
                {"PDB_ID": pdb_id, "STATUS": validator.EXPECTED_STATUS[pdb_id],
                 "REASON": "audited cause"} for pdb_id in bad_ids
            ])
            (run / "summary.json").write_text(json.dumps({
                "mode": "write", "discovered_samples": 8000,
                "failed_samples": 1200, "issues": 1200,
                "successful_samples": 5620,
                "high_rmsd_saved_samples": 1180,
                "rmsd_filtered_samples": 1180,
            }), encoding="utf-8")
            keys = sorted(validator.EXPECTED_KEYS)
            issue_rows = []
            for pdb_id in bad_ids:
                error = (
                    "ValueError: observed native atom fraction 0.047 is below 0.500"
                    if validator.EXPECTED_STATUS[pdb_id] == "SPARSE_NATIVE_COORDINATES"
                    else "ValueError: unsupported predicted RNA residues; ZTH mapped_symbol X"
                )
                issue_rows.extend({"split": "train", "pdb_id": pdb_id,
                                   "sample": f"seed_{seed}/sample_{sample}", "error": error}
                                  for seed, sample in keys)
            tsv(run / "issues.tsv", ["split", "pdb_id", "sample", "error"], issue_rows)
            normal_rows, high_rows = [], []
            for index, (pdb_id, (seed, sample)) in enumerate(
                (pdb_id, key) for pdb_id in good_ids for key in keys
            ):
                output = str(root / "virtual" / pdb_id / f"seed_{seed}/sample_{sample}.pt")
                row = {"split": "train", "pdb_id": pdb_id, "seed": seed,
                       "sample": sample, "output": output,
                       "high_rmsd_output": output, "high_rmsd_status": "CREATED"}
                (normal_rows if index < 5620 else high_rows).append(row)
            fields = list(normal_rows[0])
            tsv(run / "manifest.tsv", fields, normal_rows)
            tsv(run / "filtered_samples.tsv", fields, high_rows)
            command = ["validator", "--new-pt-run", str(run),
                       "--new-pdb-ids", str(ids), "--unusable-file", str(policy)]
            real_is_file = Path.is_file

            def virtual_pt_is_file(path):
                return path.suffix == ".pt" or real_is_file(path)

            with patch.object(sys, "argv", command), patch.object(Path, "is_file", virtual_pt_is_file):
                with contextlib.redirect_stdout(io.StringIO()):
                    validator.main()
                issue_rows[0]["error"] = "ValueError: unrelated failure"
                tsv(run / "issues.tsv", ["split", "pdb_id", "sample", "error"], issue_rows)
                with self.assertRaisesRegex(ValueError, "does not match audited cause"):
                    validator.main()

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
                       "--old-split-report", str(WORKSPACE / "Code/pipeline_reports/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE"),
                       "--new-split-report", str(WORKSPACE / "Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"),
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
            ids[1] = "1QZA"
            tsv(split / "final_manifest.tsv", ["PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"],
                [{"PDB_ID": name, "FINAL_SPLIT": "train", "FINAL_STATUS": "KEPT"} for name in ids])
            skip = root / "skip.tsv"
            tsv(skip, ["PDB_ID", "REASON"],
                [{"PDB_ID": name, "REASON": "PREP_SKIP"} for name in ids[2:]])
            unusable = root / "unusable.tsv"
            tsv(unusable, ["PDB_ID", "STATUS", "REASON"], [
                {"PDB_ID": "1QZA", "STATUS": "SPARSE_NATIVE_COORDINATES",
                 "REASON": "native has one atom per residue"}
            ])
            exclude = root / "exclude.tsv"
            tsv(exclude, ["pdb_id", "split", "reason"], [])
            pred, main_pt, high_pt = (root / name for name in ("pred", "main", "high"))
            for directory in (pred, main_pt, high_pt):
                directory.mkdir()
            main_rows, high_rows = [], []
            issue_rows = []
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
                    unusable_pred = pred / f"train/1qza/seed_{seed}/predictions"
                    unusable_pred.mkdir(parents=True, exist_ok=True)
                    (unusable_pred / f"1qza_sample_{sample}.cif").write_text("cif", encoding="utf-8")
                    issue_rows.append({"split": "train", "pdb_id": "1QZA",
                                       "sample": f"seed_{seed}/sample_{sample}",
                                       "error": "ValueError: observed native atom fraction 0.047 is below 0.500"})
            runs = [root / "new_run", root / "old_run"]
            for directory in runs:
                directory.mkdir()
            fields = list(main_rows[0])
            tsv(runs[0] / "manifest.tsv", fields, main_rows)
            tsv(runs[0] / "filtered_samples.tsv", fields, high_rows)
            tsv(runs[0] / "issues.tsv", ["split", "pdb_id", "sample", "error"], issue_rows)
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
                       "--unusable-pdb-file", str(unusable),
                       "--new-pt-run", str(runs[0]), "--retained-pt-run", str(runs[1]),
                       "--old-pt-run", str(prior)]
            subprocess.run(command + ["--output-dir", str(root / "report1")], check=True,
                           stdout=subprocess.DEVNULL)
            with (root / "report1/per_pdb.tsv").open(encoding="utf-8") as handle:
                target = next(row for row in csv.DictReader(handle, delimiter="\t") if row["pdb_id"] == "X0000")
            self.assertEqual(target["status"], "ALL_200_PT_ACCOUNTED")
            self.assertEqual(target["high_rmsd_pt_count"], "1")
            with (root / "report1/per_pdb.tsv").open(encoding="utf-8") as handle:
                unusable_row = next(row for row in csv.DictReader(handle, delimiter="\t")
                                    if row["pdb_id"] == "1QZA")
            self.assertEqual(unusable_row["status"], "PT_UNUSABLE_SPARSE_NATIVE_COORDINATES")
            self.assertEqual(unusable_row["missing_pt_count"], "200")
            (high_pt / "train/x0000/seed_349/sample_3.pt").unlink()
            subprocess.run(command + ["--output-dir", str(root / "report2")], check=True,
                           stdout=subprocess.DEVNULL)
            with (root / "report2/missing_samples.tsv").open(encoding="utf-8") as handle:
                missing = [row for row in csv.DictReader(handle, delimiter="\t") if row["pdb_id"] == "X0000"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["reason"], "RMSD_GT30_PT_NOT_SAVED")


if __name__ == "__main__":
    unittest.main()
