import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".." / "predict_protenix" / "Version_3" / "data_v1_50x4_confidence"
    / "prepare_data_v1_confidence_run.py"
).resolve()


class PredictionWorklistTests(unittest.TestCase):
    def test_selected_worklist_skips_unavailable_prep_but_checks_full_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "Data_V2"
            simple = root / "simple"
            complex_dir = root / "complex"
            run = root / "run"
            simple.mkdir()
            complex_dir.mkdir()
            for split, ids in {"train": ("1aaa", "1aab"), "val": ("1aac",), "test": ("1aad",)}.items():
                for pdb_id in ids:
                    (data / split / pdb_id).mkdir(parents=True)
            for pdb_id in ("1aaa", "1aad"):
                (simple / f"{pdb_id}-final-updated.json").write_text("{}", encoding="utf-8")
                (complex_dir / f"prep_output_{pdb_id}").mkdir()
            master = root / "master.csv"
            with master.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["PDB_ID", "CURRENT_TARGET"])
                writer.writeheader()
                writer.writerows({"PDB_ID": pdb_id, "CURRENT_TARGET": "true"} for pdb_id in ("1AAA", "1AAB", "1AAC", "1AAD"))
            split_manifest = root / "final_manifest.tsv"
            with split_manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"], delimiter="\t")
                writer.writeheader()
                writer.writerows(
                    {"PDB_ID": pdb_id, "FINAL_SPLIT": split, "FINAL_STATUS": "KEPT"}
                    for pdb_id, split in (("1AAA", "train"), ("1AAB", "train"), ("1AAC", "val"), ("1AAD", "test"))
                )
            targets = root / "targets.txt"
            targets.write_text("1AAA\n1AAD\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "-B", str(SCRIPT), "--master-manifest", str(master),
                    "--split-manifest", str(split_manifest), "--data-root", str(data),
                    "--simple-json-dir", str(simple), "--complex-json-dir", str(complex_dir),
                    "--run-dir", str(run), "--seeds", "300,301", "--samples", "4",
                    "--allow-variable-split-counts", "--target-ids-file", str(targets),
                ], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((run / "selection_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["full_split_target_count"], 4)
            self.assertEqual(summary["total_target_count"], 2)
            self.assertEqual((summary["train_count"], summary["val_count"], summary["test_count"]), (1, 0, 1))
            with (run / "train_50x4_confidence_manifest.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([row["PDB_ID"] for row in csv.DictReader(stream)], ["1AAA"])


if __name__ == "__main__":
    unittest.main()
