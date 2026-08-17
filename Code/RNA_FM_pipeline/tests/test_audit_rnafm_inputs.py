from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "audit_rnafm_inputs.py"
SPEC = importlib.util.spec_from_file_location("audit_rnafm_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_json(path: Path, pdb_id: str, rna_entries: list[tuple[str, int]]) -> None:
    sequences = [
        {"rnaSequence": {"sequence": sequence, "count": count}}
        for sequence, count in rna_entries
    ]
    path.write_text(
        json.dumps([{"name": pdb_id.lower(), "sequences": sequences}]),
        encoding="utf-8",
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["PDB_ID", "ENTITY_ID", "CHAIN_ID", "SEQUENCE_CANONICAL"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "PDB_ID",
                "INITIAL_SPLIT",
                "FINAL_SPLIT",
                "FINAL_STATUS",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


class AuditRnaFmInputsTests(unittest.TestCase):
    def build_valid_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        json_dir = root / "Simple_json"
        split_root = root / "Data"
        report_root = root / "reports"
        manifest = root / "rna_chain_sequences.csv"
        json_dir.mkdir()
        for split in ("train", "val", "test"):
            (split_root / split).mkdir(parents=True)

        write_json(json_dir / "1abc.json", "1abc", [("ACG", 2)])
        write_json(json_dir / "2def.json", "2def", [("UUG", 1), ("CACA", 1)])
        write_json(json_dir / "3ok2.json", "3ok2", [("A", 1)])
        # A prep output is deliberately present and must be ignored as an input.
        write_json(
            json_dir / "1abc-final-updated.json", "1abc", [("ACG", 2)]
        )

        (split_root / "train" / "1ABC").mkdir()
        (split_root / "val" / "2def").mkdir()
        write_manifest(
            manifest,
            [
                {"PDB_ID": "1ABC", "ENTITY_ID": "1", "CHAIN_ID": "A", "SEQUENCE_CANONICAL": "ACG"},
                {"PDB_ID": "1ABC", "ENTITY_ID": "1", "CHAIN_ID": "B", "SEQUENCE_CANONICAL": "ACG"},
                {"PDB_ID": "2DEF", "ENTITY_ID": "2", "CHAIN_ID": "X", "SEQUENCE_CANONICAL": "UUG"},
                {"PDB_ID": "2DEF", "ENTITY_ID": "3", "CHAIN_ID": "Y", "SEQUENCE_CANONICAL": "CACA"},
                {"PDB_ID": "3OK2", "ENTITY_ID": "4", "CHAIN_ID": "A", "SEQUENCE_CANONICAL": "A"},
            ],
        )
        return json_dir, manifest, split_root, report_root

    def test_valid_fixture_passes_and_writes_chain_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir, manifest, split_root, report_root = self.build_valid_fixture(root)
            code = AUDIT.main(
                [
                    "--json-dir", str(json_dir),
                    "--chain-manifest", str(manifest),
                    "--split-root", str(split_root),
                    "--report-root", str(report_root),
                    "--report-name", "AUDIT_TEST",
                    "--exclude", "3OK2",
                    "--expected-source-count", "3",
                    "--expected-target-count", "2",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads(
                (report_root / "AUDIT_TEST" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["counts"]["expanded_json_rna_chains"], 4)
            self.assertEqual(summary["counts"]["mapped_original_chains"], 4)
            self.assertEqual(summary["counts"]["final_updated_files_seen"], 1)

            with (report_root / "AUDIT_TEST" / "chain_audit.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["ORIGINAL_CHAIN_ID"] for row in rows], ["A", "B", "X", "Y"])
            self.assertEqual(
                [row["EXPECTED_PROTENIX_CHAIN_ID"] for row in rows],
                ["A", "B", "A", "B"],
            )
            self.assertEqual(
                rows[0]["MAPPING_STATUS"], "IDENTICAL_SEQUENCE_ORDER_ASSUMED"
            )

    def test_sequence_mismatch_returns_needs_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir, manifest, split_root, report_root = self.build_valid_fixture(root)
            rows = [
                {"PDB_ID": "1ABC", "ENTITY_ID": "1", "CHAIN_ID": "A", "SEQUENCE_CANONICAL": "ACG"},
                {"PDB_ID": "1ABC", "ENTITY_ID": "1", "CHAIN_ID": "B", "SEQUENCE_CANONICAL": "AAA"},
                {"PDB_ID": "2DEF", "ENTITY_ID": "2", "CHAIN_ID": "X", "SEQUENCE_CANONICAL": "UUG"},
                {"PDB_ID": "2DEF", "ENTITY_ID": "3", "CHAIN_ID": "Y", "SEQUENCE_CANONICAL": "CACA"},
                {"PDB_ID": "3OK2", "ENTITY_ID": "4", "CHAIN_ID": "A", "SEQUENCE_CANONICAL": "A"},
            ]
            write_manifest(manifest, rows)
            code = AUDIT.main(
                [
                    "--json-dir", str(json_dir),
                    "--chain-manifest", str(manifest),
                    "--split-root", str(split_root),
                    "--report-root", str(report_root),
                    "--report-name", "AUDIT_BAD",
                    "--exclude", "3OK2",
                    "--expected-source-count", "3",
                    "--expected-target-count", "2",
                ]
            )
            self.assertEqual(code, 1)
            summary = json.loads(
                (report_root / "AUDIT_BAD" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "NEEDS_REVIEW")
            self.assertEqual(
                summary["counts"]["sequence_multiset_mismatch_pdbs"], 1
            )

    def test_expected_dropped_pdb_is_not_a_missing_split_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir, manifest, split_root, report_root = self.build_valid_fixture(root)
            (split_root / "val" / "2def").rmdir()
            split_manifest = (
                report_root
                / "DATA_SPLIT_2241_CHAINMASK_20260807T114307Z_EXECUTE"
                / "final_manifest.tsv"
            )
            write_split_manifest(
                split_manifest,
                [
                    {
                        "PDB_ID": "1ABC",
                        "INITIAL_SPLIT": "train",
                        "FINAL_SPLIT": "train",
                        "FINAL_STATUS": "KEPT",
                    },
                    {
                        "PDB_ID": "2DEF",
                        "INITIAL_SPLIT": "test",
                        "FINAL_SPLIT": "",
                        "FINAL_STATUS": "DROP_NO_EVALUABLE_CHAINS",
                    },
                ],
            )
            code = AUDIT.main(
                [
                    "--json-dir", str(json_dir),
                    "--chain-manifest", str(manifest),
                    "--split-root", str(split_root),
                    "--report-root", str(report_root),
                    "--report-name", "AUDIT_DROPPED",
                    "--exclude", "3OK2",
                    "--expected-source-count", "3",
                    "--expected-target-count", "2",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads(
                (report_root / "AUDIT_DROPPED" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["counts"]["expected_dropped_pdbs"], 1)
            self.assertEqual(summary["counts"]["split_counts"], {"train": 1})

    def test_non_acgu_and_overlength_are_warnings_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir, manifest, split_root, report_root = self.build_valid_fixture(root)
            write_json(json_dir / "2def.json", "2def", [("NUG", 1), ("CACA", 1)])
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["PDB_ID"] == "2DEF" and row["CHAIN_ID"] == "X":
                    row["SEQUENCE_CANONICAL"] = "NUG"
            write_manifest(manifest, rows)
            code = AUDIT.main(
                [
                    "--json-dir", str(json_dir),
                    "--chain-manifest", str(manifest),
                    "--split-root", str(split_root),
                    "--report-root", str(report_root),
                    "--report-name", "AUDIT_WARNINGS",
                    "--exclude", "3OK2",
                    "--expected-source-count", "3",
                    "--expected-target-count", "2",
                    "--max-residues", "3",
                ]
            )
            self.assertEqual(code, 0)
            summary = json.loads(
                (report_root / "AUDIT_WARNINGS" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["counts"]["non_acgu_chains"], 1)
            self.assertGreater(summary["counts"]["warning_issues"], 0)
            self.assertEqual(summary["counts"]["error_issues"], 0)


if __name__ == "__main__":
    unittest.main()
