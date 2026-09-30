from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook


MODULE_PATH = Path(__file__).resolve().parents[1] / "split_rna_dataset.py"
SPEC = importlib.util.spec_from_file_location("split_rna_dataset", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


VALID_CIF = """\
data_1ABC
_entry.id 1ABC
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 2024-06-01
2 'Structure model' 1 1 2025-01-01
loop_
_database_PDB_rev.num
_database_PDB_rev.date_original
1 2024-06-01
_exptl.method 'X-RAY DIFFRACTION'
_refine.ls_d_res_high 2.4
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_strand_id
1 polyribonucleotide ACGU ACGU A,B
#
"""


class SplitPipelineTests(unittest.TestCase):
    def test_parse_entry_uses_initial_revision_date_and_one_entity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1abc.cif"
            path.write_text(VALID_CIF, encoding="utf-8")
            entry = pipeline.parse_entry(path, "1ABC")
        self.assertEqual(entry.release.release_date, date(2024, 6, 1))
        self.assertEqual(entry.release.revision_ordinal, "1")
        self.assertTrue(entry.release.sources_agree)
        self.assertEqual(len(entry.entities), 1)
        self.assertEqual(entry.entities[0].chain_ids, ("A", "B"))
        self.assertEqual(entry.entities[0].search_sequence, "ACGT")

    def test_conflicting_date_sources_are_rejected(self):
        text = VALID_CIF.replace("1 2024-06-01\n_exptl", "1 2024-06-02\n_exptl")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1abc.cif"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting initial release dates"):
                pipeline.parse_entry(path, "1ABC")

    def test_chronological_boundaries(self):
        train_end = date(2023, 12, 31)
        val_end = date(2024, 12, 31)
        self.assertEqual(pipeline.initial_split(date(2023, 12, 31), train_end, val_end), "train")
        self.assertEqual(pipeline.initial_split(date(2024, 1, 1), train_end, val_end), "val")
        self.assertEqual(pipeline.initial_split(date(2024, 12, 31), train_end, val_end), "val")
        self.assertEqual(pipeline.initial_split(date(2025, 1, 1), train_end, val_end), "test")

    def test_hit_filter_requires_both_coverages(self):
        query = pipeline.Entity("9AAA", "1", ("A",), "ACGU", "ACGT", date(2025, 1, 1), "NMR", None)
        target = pipeline.Entity("1AAA", "2", ("B",), "ACGU", "ACGT", date(2020, 1, 1), "NMR", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hits.tsv"
            path.write_text(
                "9AAA__ENTITY_1\t1AAA__ENTITY_2\t0.90\t0.90\t0.79\t4\t1e-5\t20\n"
                "9AAA__ENTITY_1\t1AAA__ENTITY_2\t0.85\t0.80\t0.80\t4\t1e-4\t18\n",
                encoding="utf-8",
            )
            hits = pipeline.load_hits(
                path,
                {"9AAA__ENTITY_1": query},
                {"1AAA__ENTITY_2": target},
                0.8,
                0.8,
                0.8,
                exclude_same_pdb=False,
            )
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].identity, 0.85)

    def test_connected_components_and_quality_representative(self):
        def fake_entry(pdb_id: str, resolution: float | None):
            audit = pipeline.DateAudit(date(2025, 1, 1), "x", "1", "2025-01-01", "", None)
            return pipeline.Entry(pdb_id, Path(pdb_id), "hash", audit, (), resolution)

        hit = pipeline.Hit("9AAA", "1", "9AAB", "1", 0.9, 0.9, 0.9, 10, "0", "1")
        components = pipeline.connected_components({"9AAA", "9AAB", "9AAC"}, [hit])
        self.assertEqual({frozenset(item) for item in components}, {frozenset({"9AAA", "9AAB"}), frozenset({"9AAC"})})
        entries = {"9AAA": fake_entry("9AAA", 3.0), "9AAB": fake_entry("9AAB", 2.0)}
        representative = min({"9AAA", "9AAB"}, key=lambda pdb_id: pipeline.representative_key(entries[pdb_id]))
        self.assertEqual(representative, "9AAB")

    def test_short_sequence_fallback_applies_identity_threshold(self):
        query = pipeline.Entity("9AAA", "1", ("A",), "ACGU", "ACGT", date(2025, 1, 1), "NMR", None)
        target_hit = pipeline.Entity("1AAA", "1", ("A",), "ACGA", "ACGA", date(2020, 1, 1), "NMR", None)
        target_miss = pipeline.Entity("1AAB", "1", ("A",), "AAAA", "AAAA", date(2020, 1, 1), "NMR", None)
        hits = pipeline.short_sequence_hits(
            [query], [target_hit, target_miss], 0.75, 0.8, 0.8, exclude_same_pdb=False
        )
        self.assertEqual([(hit.target_pdb_id, hit.alignment_source) for hit in hits], [("1AAA", "SHORT_GLOBAL_FALLBACK")])

    def test_materialization_dry_run_does_not_create_data_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "Data"
            expected = {"train": {"1ABC"}, "val": {"2ABC"}, "test": {"3ABC"}}
            actions = pipeline.materialize_empty_directories(data_dir, expected, execute=False)
            self.assertFalse(data_dir.exists())
            self.assertEqual({row["ACTION"] for row in actions}, {"WOULD_CREATE"})

    def test_execute_creates_only_empty_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "Data"
            expected = {"train": {"1ABC"}, "val": {"2ABC"}, "test": {"3ABC"}}
            pipeline.materialize_empty_directories(data_dir, expected, execute=True)
            self.assertTrue((data_dir / "train" / "1abc").is_dir())
            self.assertEqual(list((data_dir / "train" / "1abc").iterdir()), [])

    def test_dry_run_pipeline_writes_reports_but_not_data_directories(self):
        def make_cif(pdb_id: str, release_date: str, sequence: str) -> str:
            return (
                VALID_CIF.replace("1ABC", pdb_id)
                .replace("2024-06-01", release_date)
                .replace("ACGU ACGU", f"{sequence} {sequence}")
            )

        def fake_mmseqs(**kwargs):
            kwargs["output_tsv"].write_text("", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cif_dir = root / "pdb_data"
            report_dir = root / "report"
            data_dir = root / "Data"
            cif_dir.mkdir()
            report_dir.mkdir()
            for pdb_id, released, sequence in (
                ("1AAA", "2023-01-01", "ACGU"),
                ("2AAA", "2024-01-01", "CCCC"),
                ("3AAA", "2025-01-01", "ACGU"),
                ("4AAA", "2025-02-01", "GGGG"),
            ):
                (cif_dir / f"{pdb_id.lower()}.cif").write_text(
                    make_cif(pdb_id, released, sequence), encoding="utf-8"
                )
            (cif_dir / "5aaa.cif").write_text("excluded", encoding="utf-8")

            exclusion_xlsx = root / "exclude.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["PDB_ID"])
            sheet.append(["5AAA"])
            workbook.save(exclusion_xlsx)
            workbook.close()

            args = SimpleNamespace(
                cif_dir=cif_dir,
                exclusion_xlsx=exclusion_xlsx,
                data_dir=data_dir,
                train_end=date(2023, 12, 31),
                val_end=date(2024, 12, 31),
                min_seq_id=0.8,
                min_query_cov=0.8,
                min_target_cov=0.8,
                threads=1,
                mmseqs="mmseqs",
                execute=False,
                skip_count_check=True,
            )
            logger = pipeline.RunLogger(report_dir / "pipeline.log")
            with patch.object(pipeline, "mmseqs_version", return_value="test"), patch.object(
                pipeline, "run_mmseqs", side_effect=fake_mmseqs
            ):
                summary = pipeline.run_pipeline(args, report_dir, logger)

            self.assertEqual(summary["split_before_dedup"], {"test": 2, "train": 1, "val": 1})
            self.assertEqual(summary["test_removed_against_train_val"], 1)
            self.assertEqual(summary["final_directory_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertFalse(data_dir.exists())
            self.assertTrue((report_dir / "final_manifest.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
