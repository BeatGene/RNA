from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "split_rna_dataset_v1.py"
SPEC = importlib.util.spec_from_file_location("split_rna_dataset_v1", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)
legacy = pipeline.legacy


def make_entry(
    pdb_id: str,
    released: date,
    *,
    chain_ids: tuple[str, ...] = ("A",),
    sequence: str = "ACGU",
) -> legacy.Entry:
    audit = legacy.DateAudit(
        release_date=released,
        selected_source="test",
        revision_ordinal="1",
        revision_date=released.isoformat(),
        legacy_original_date="",
        sources_agree=None,
    )
    entity = legacy.Entity(
        pdb_id=pdb_id,
        entity_id="1",
        chain_ids=chain_ids,
        sequence=sequence,
        search_sequence=sequence.replace("U", "T"),
        release_date=released,
        experiment_method="NMR",
        resolution=None,
    )
    return legacy.Entry(pdb_id, Path(f"{pdb_id}.cif"), "hash", audit, (entity,), None)


def make_rank1(
    pdb_id: str,
    released: date,
    rmsd: float | None,
    *,
    chain_count: int = 1,
    eval_status: str = "SUCCESS",
) -> pipeline.Rank1Record:
    return pipeline.Rank1Record(
        pdb_id=pdb_id,
        release_date=released,
        chain_count=chain_count,
        rmsd=rmsd,
        eval_status=eval_status,
        seed="42",
        sample="0",
        ranking_score="0.5",
    )


class DataV1PipelineTests(unittest.TestCase):
    def test_defaults_match_agreed_contract(self):
        args = pipeline.parse_args([])
        self.assertEqual(args.train_end, date(2021, 9, 30))
        self.assertEqual(args.val_end, date(2023, 12, 31))
        self.assertEqual(args.train_rmsd_max, 15.0)
        self.assertEqual(args.data_dir, Path("~/Data_V1").expanduser())
        self.assertFalse(args.execute)

    def test_rmsd_cutoff_applies_only_to_train(self):
        entries = {
            "1AAA": make_entry("1AAA", date(2021, 9, 30)),
            "1AAB": make_entry("1AAB", date(2021, 9, 29)),
            "2AAA": make_entry("2AAA", date(2022, 1, 1)),
            "3AAA": make_entry("3AAA", date(2024, 1, 1)),
        }
        rank1 = {
            "1AAA": make_rank1("1AAA", date(2021, 9, 30), 15.0),
            "1AAB": make_rank1("1AAB", date(2021, 9, 29), 15.001),
            "2AAA": make_rank1("2AAA", date(2022, 1, 1), 50.0),
            "3AAA": make_rank1("3AAA", date(2024, 1, 1), 55.0),
        }
        assignments, rows, statuses = pipeline.select_entries(
            entries,
            rank1,
            set(),
            date(2021, 9, 30),
            date(2023, 12, 31),
            15.0,
        )
        self.assertEqual(assignments, {"1AAA": "train", "2AAA": "val", "3AAA": "test"})
        self.assertEqual(statuses["EXCLUDE_TRAIN_RMSD_CUTOFF"], 1)
        by_id = {row["PDB_ID"]: row for row in rows}
        self.assertEqual(by_id["1AAB"]["SELECTION_STATUS"], "EXCLUDE_TRAIN_RMSD_CUTOFF")
        self.assertEqual(by_id["2AAA"]["SELECTION_STATUS"], "SELECTED")
        self.assertEqual(by_id["3AAA"]["SELECTION_STATUS"], "SELECTED")

    def test_single_chain_and_metric_requirements(self):
        released = date(2020, 1, 1)
        entries = {
            "1AAA": make_entry("1AAA", released, chain_ids=("A", "B")),
            "1AAB": make_entry("1AAB", released),
            "1AAC": make_entry("1AAC", released),
            "1AAD": make_entry("1AAD", released),
        }
        rank1 = {
            "1AAA": make_rank1("1AAA", released, 1.0, chain_count=2),
            "1AAB": make_rank1("1AAB", released, None, eval_status="FAILED"),
            "1AAC": make_rank1("1AAC", released, 1.0),
        }
        assignments, rows, _ = pipeline.select_entries(
            entries,
            rank1,
            {"1AAC"},
            date(2021, 9, 30),
            date(2023, 12, 31),
            15.0,
        )
        self.assertEqual(assignments, {})
        by_id = {row["PDB_ID"]: row["SELECTION_STATUS"] for row in rows}
        self.assertEqual(by_id["1AAA"], "EXCLUDE_MULTI_RNA_CHAIN")
        self.assertEqual(by_id["1AAB"], "EXCLUDE_INVALID_RMSD")
        self.assertEqual(by_id["1AAC"], "EXCLUDE_FROZEN_RMSD")
        self.assertEqual(by_id["1AAD"], "EXCLUDE_NO_STRICT_RANK1")

    def test_rank1_metadata_mismatch_is_fatal(self):
        entries = {"1AAA": make_entry("1AAA", date(2020, 1, 1))}
        rank1 = {"1AAA": make_rank1("1AAA", date(2020, 1, 2), 1.0)}
        with self.assertRaisesRegex(ValueError, "Release-date mismatch"):
            pipeline.select_entries(
                entries,
                rank1,
                set(),
                date(2021, 9, 30),
                date(2023, 12, 31),
                15.0,
            )

    def test_distribution_figures_are_generated(self):
        entries = {
            "1AAA": make_entry("1AAA", date(2020, 1, 1), sequence="ACGU"),
            "2AAA": make_entry("2AAA", date(2022, 1, 1), sequence="ACGUACGU"),
            "3AAA": make_entry("3AAA", date(2024, 1, 1), sequence="ACGUACGUACGU"),
        }
        rank1 = {
            "1AAA": make_rank1("1AAA", date(2020, 1, 1), 2.0),
            "2AAA": make_rank1("2AAA", date(2022, 1, 1), 8.0),
            "3AAA": make_rank1("3AAA", date(2024, 1, 1), 20.0),
        }
        expected = {"train": {"1AAA"}, "val": {"2AAA"}, "test": {"3AAA"}}
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            pipeline.plot_distributions(report_dir, expected, rank1, entries, 15.0)
            figures = report_dir / "figures"
            for name in (
                "rmsd_density.png",
                "rmsd_density_log1p.png",
                "rmsd_ecdf.png",
                "rna_length_density.png",
            ):
                self.assertTrue((figures / name).is_file(), name)

    def test_load_rank1_rejects_duplicate_pdb(self):
        header = "pdb_id,seed,sample,ranking_score,eval_status,rmsd,release_date,chain_count\n"
        row = "1AAA,42,0,0.5,SUCCESS,1.2,2020-01-01,1\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank1.csv"
            path.write_text(header + row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate strict rank-1"):
                pipeline.load_rank1_records(path)

    def test_end_to_end_dry_run_filters_train_and_test(self):
        def cif_text(
            pdb_id: str,
            released: str,
            sequence: str,
            chains: str = "A",
        ) -> str:
            return f"""data_{pdb_id}
_entry.id {pdb_id}
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 {released}
_exptl.method 'SOLUTION NMR'
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_strand_id
1 polyribonucleotide {sequence} {sequence} {chains}
#
"""

        def fake_mmseqs(**kwargs):
            output = kwargs["output_tsv"]
            if output.name == "test_vs_train_val.raw.tsv":
                output.write_text(
                    "3AAA__ENTITY_1\t1AAA__ENTITY_1\t0.90\t0.90\t0.90\t20\t0\t20\n",
                    encoding="utf-8",
                )
            else:
                output.write_text("", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cif_dir = root / "pdb_data"
            report_dir = root / "report"
            data_dir = root / "Data_V1"
            cif_dir.mkdir()
            report_dir.mkdir()
            specs = (
                ("1AAA", "2021-09-30", "ACGUACGUACGUACGUACGU", "A", 14.0, 1),
                ("1AAB", "2021-09-29", "CCCCCCCCCCCCCCCCCCCC", "A", 16.0, 1),
                ("2AAA", "2022-01-01", "GGGGGGGGGGGGGGGGGGGG", "A", 50.0, 1),
                ("3AAA", "2024-01-01", "UUUUUUUUUUUUUUUUUUUU", "A", 40.0, 1),
                ("4AAA", "2025-01-01", "AGAGAGAGAGAGAGAGAGAG", "A", 45.0, 1),
                ("6AAA", "2020-01-01", "CACACACACACACACACACA", "A,B", 2.0, 2),
            )
            for pdb_id, released, sequence, chains, _, _ in specs:
                (cif_dir / f"{pdb_id.lower()}.cif").write_text(
                    cif_text(pdb_id, released, sequence, chains), encoding="utf-8"
                )
            (cif_dir / "5aaa.cif").write_text("excluded", encoding="utf-8")

            exclusion_xlsx = root / "exclude.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["PDB_ID"])
            sheet.append(["5AAA"])
            workbook.save(exclusion_xlsx)
            workbook.close()

            rank1_csv = root / "rank1.csv"
            with rank1_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "pdb_id", "seed", "sample", "ranking_score", "eval_status",
                        "rmsd", "release_date", "chain_count",
                    ]
                )
                for pdb_id, released, _, _, rmsd, chain_count in specs:
                    writer.writerow(
                        [pdb_id, 42, 0, 0.5, "SUCCESS", rmsd, released, chain_count]
                    )
            rmsd_exclusions = root / "rmsd_exclusions.txt"
            rmsd_exclusions.write_text("", encoding="utf-8")

            args = SimpleNamespace(
                cif_dir=cif_dir,
                exclusion_xlsx=exclusion_xlsx,
                rank1_csv=rank1_csv,
                rmsd_exclusion_list=rmsd_exclusions,
                data_dir=data_dir,
                train_end=date(2021, 9, 30),
                val_end=date(2023, 12, 31),
                train_rmsd_max=15.0,
                min_seq_id=0.8,
                min_query_cov=0.8,
                min_target_cov=0.8,
                threads=1,
                mmseqs="mmseqs",
                execute=False,
                skip_count_check=True,
                skip_plots=True,
            )
            logger = legacy.RunLogger(report_dir / "pipeline.log")
            with patch.object(legacy, "mmseqs_version", return_value="test"), patch.object(
                legacy, "run_mmseqs", side_effect=fake_mmseqs
            ):
                summary = pipeline.run_pipeline(args, report_dir, logger)

            self.assertEqual(
                summary["selected_before_test_filtering"],
                {"test": 2, "train": 1, "val": 1},
            )
            self.assertEqual(
                summary["selection_status_counts"],
                {
                    "EXCLUDE_MULTI_RNA_CHAIN": 1,
                    "EXCLUDE_TRAIN_RMSD_CUTOFF": 1,
                    "SELECTED": 4,
                },
            )
            self.assertEqual(summary["test_filtering"]["reference_homologs_removed"], 1)
            self.assertEqual(summary["final_directory_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertFalse(data_dir.exists())
            self.assertTrue((report_dir / "selection_audit.tsv").is_file())
            self.assertTrue((report_dir / "final_manifest.tsv").is_file())
            self.assertTrue((report_dir / "distribution_summary.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
