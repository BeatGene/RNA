import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "data_v1_50x4_confidence"
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_data_v1_confidence_run as prepare


class PrepareDataV1ConfidenceRunTests(unittest.TestCase):
    def test_builds_all_three_audited_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master = root / "master.csv"
            split_manifest = root / "split.tsv"
            data_root = root / "Data_V1"
            simple = root / "Simple_json"
            complex_dir = root / "Complex_json"
            run_dir = root / "run"
            simple.mkdir()
            complex_dir.mkdir()

            assignments: list[tuple[str, str]] = []
            serial = 0
            for split, count in prepare.EXPECTED_COUNTS.items():
                (data_root / split).mkdir(parents=True)
                for _ in range(count):
                    identifier = f"P{serial:04d}"
                    serial += 1
                    assignments.append((identifier, split))
                    (data_root / split / identifier.lower()).mkdir()
                    (simple / f"{identifier.lower()}-final-updated.json").write_text(
                        "[]", encoding="utf-8"
                    )
                    (complex_dir / f"prep_output_{identifier.lower()}").mkdir()

            with master.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["PDB_ID", "CURRENT_TARGET", "FILE_PATH"]
                )
                writer.writeheader()
                for identifier, _ in assignments:
                    writer.writerow(
                        {
                            "PDB_ID": identifier,
                            "CURRENT_TARGET": "True",
                            "FILE_PATH": f"/pdb/{identifier.lower()}.cif",
                        }
                    )
            with split_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    delimiter="\t",
                    fieldnames=["PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"],
                )
                writer.writeheader()
                for identifier, split in assignments:
                    writer.writerow(
                        {
                            "PDB_ID": identifier,
                            "FINAL_SPLIT": split,
                            "FINAL_STATUS": "KEPT",
                        }
                    )

            argv = [
                "prepare_data_v1_confidence_run.py",
                "--master-manifest",
                str(master),
                "--split-manifest",
                str(split_manifest),
                "--data-root",
                str(data_root),
                "--simple-json-dir",
                str(simple),
                "--complex-json-dir",
                str(complex_dir),
                "--run-dir",
                str(run_dir),
                "--seeds",
                ",".join(map(str, range(300, 350))),
                "--samples",
                "4",
            ]
            with mock.patch.object(sys, "argv", argv):
                prepare.main()

            summary = json.loads(
                (run_dir / "selection_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["total_target_count"], 988)
            self.assertEqual(summary["expected_total_decoys"], 197600)
            self.assertEqual(summary["expected_total_full_data_json"], 197600)
            for split, count in prepare.EXPECTED_COUNTS.items():
                with (run_dir / f"{split}_50x4_confidence_manifest.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)


if __name__ == "__main__":
    unittest.main()
