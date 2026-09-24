import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "audit_v2_prediction_reuse.py"
SPEC = importlib.util.spec_from_file_location("audit_v2_prediction_reuse", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ReuseAuditTests(unittest.TestCase):
    def test_abandoned_table_reads_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            table = Path(temp) / "abandoned.tsv"
            table.write_text("PDB_ID\tREASON\n3J28\tLONG\n7ZFW\tLONG\n", encoding="utf-8")
            self.assertEqual(MODULE.read_ids(table), {"3J28", "7ZFW"})

    def test_complete_files_and_missing_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            pdb = Path(temp) / "data" / "train" / "1abc"
            with patch.object(MODULE, "EXPECTED_SEEDS", range(300, 302)):
                for seed in (300, 301):
                    pred = pdb / f"seed_{seed}" / "predictions"
                    pred.mkdir(parents=True)
                    for sample in range(4):
                        for name in (
                            f"1abc_sample_{sample}.cif",
                            f"1abc_summary_confidence_sample_{sample}.json",
                            f"1abc_full_data_sample_{sample}.json",
                        ):
                            (pred / name).write_text("valid", encoding="utf-8")
                self.assertTrue(MODULE.check_old_files(pdb)[0])
                (pdb / "seed_301" / "predictions" / "1abc_full_data_sample_3.json").unlink()
                complete, reason = MODULE.check_old_files(pdb)
                self.assertFalse(complete)
                self.assertIn("seed_301 full", reason)

    def test_destination_must_be_empty_and_unlinked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "old"
            source.mkdir()
            destination = root / "new"
            destination.mkdir()
            self.assertEqual(MODULE.destination_status(destination, source), "READY_EMPTY")
            (destination / "unexpected.txt").write_text("x", encoding="utf-8")
            self.assertEqual(MODULE.destination_status(destination, source), "DEST_NOT_EMPTY")

    def test_root_directory_symlink_exposes_prediction_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "old"
            pred = source / "seed_300" / "predictions"
            pred.mkdir(parents=True)
            (pred / "1abc_sample_0.cif").write_text("valid", encoding="utf-8")
            destination = root / "new"
            try:
                destination.symlink_to(source, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlink unavailable here: {exc}")
            self.assertEqual(MODULE.destination_status(destination, source), "ALREADY_LINKED")
            self.assertEqual(len(list(destination.rglob("*_sample_*.cif"))), 1)


if __name__ == "__main__":
    unittest.main()
