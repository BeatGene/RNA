import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import stage2_decoy_pipeline as pipeline


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, msa_path: str = "") -> None:
    rna = {"sequence": "ACGU", "count": 1}
    if msa_path:
        rna["unpairedMsaPath"] = msa_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [{"name": path.name.split("-")[0], "sequences": [{"rnaSequence": rna}]}]
        ),
        encoding="utf-8",
    )


def write_cif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "data_test\n#\n"
        "_atom_site.group_PDB ATOM\n"
        "_atom_site.Cartn_x 1.0\n"
        + ("# padding\n" * 20),
        encoding="utf-8",
    )


class Stage2AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.simple = self.root / "simple"
        self.complex = self.root / "complex"
        self.pred = self.root / "foldbench_predictions"
        self.report = self.root / "reports"
        self.cif_dir = self.root / "cif"
        self.manifest = self.root / "pdb_cif_manifest.csv"
        self.chains = self.root / "rna_chain_sequences.csv"
        write_csv(
            self.manifest,
            [
                {
                    "PDB_ID": "1ABC",
                    "CURRENT_TARGET": "True",
                    "LEGACY_1979": "True",
                    "FILE_PATH": "",
                },
                {
                    "PDB_ID": "2DEF",
                    "CURRENT_TARGET": "True",
                    "LEGACY_1979": "False",
                    "FILE_PATH": "",
                },
                {
                    "PDB_ID": "3GHI",
                    "CURRENT_TARGET": "False",
                    "LEGACY_1979": "True",
                    "FILE_PATH": "",
                },
            ],
        )
        write_csv(
            self.chains,
            [
                {
                    "PDB_ID": "1ABC",
                    "ENTITY_ID": "1",
                    "CHAIN_ID": "X",
                    "SEQUENCE_CANONICAL": "ACGU",
                },
                {
                    "PDB_ID": "2DEF",
                    "ENTITY_ID": "1",
                    "CHAIN_ID": "Y",
                    "SEQUENCE_CANONICAL": "ACGU",
                },
            ],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            manifest=self.manifest,
            chain_manifest=self.chains,
            cif_dir=self.cif_dir,
            simple_json_dir=self.simple,
            complex_json_dir=self.complex,
            pred_output_dir=self.pred,
            report_dir=self.report,
            seeds=[42, 43],
            samples=2,
            cif_validation="quick",
            need_atom_confidence=False,
            protenix="protenix",
        )

    def make_complete_target_with_stale_msa(self) -> None:
        write_json(self.simple / "1abc.json")
        write_json(
            self.simple / "1abc-final-updated.json",
            "/old/server/prep_output_1abc/1abc/rna_msa/0/rna_msa.a3m",
        )
        a3m = (
            self.complex
            / "prep_output_1abc"
            / "1abc"
            / "rna_msa"
            / "0"
            / "rna_msa.a3m"
        )
        a3m.parent.mkdir(parents=True)
        a3m.write_text(">query\nACGU\n", encoding="utf-8")
        for seed in (42, 43):
            pred = (
                self.pred
                / f"pred_output_1abc_seed_{seed}"
                / "1abc"
                / f"seed_{seed}"
                / "predictions"
            )
            for rank in range(2):
                write_cif(pred / f"1abc_sample_{rank}.cif")
                write_cif(pred / f"1abc_sample_{rank}_wounresol.cif")
                (pred / f"1abc_summary_confidence_sample_{rank}.json").write_text(
                    json.dumps({"ranking_score": 0.5}), encoding="utf-8"
                )

    def test_audit_counts_primary_cifs_and_rebases_msa(self) -> None:
        self.make_complete_target_with_stale_msa()
        rows, seed_rows, chain_rows, summary = pipeline.build_audit(self.args())
        by_id = {row["PDB_ID"]: row for row in rows}
        self.assertEqual(by_id["1ABC"]["PREP_STATUS"], "COMPLETE_REBASABLE")
        self.assertEqual(by_id["1ABC"]["VALID_DECOY_COUNT"], 4)
        self.assertEqual(by_id["1ABC"]["OVERALL_STATUS"], "COMPLETE")
        self.assertEqual(by_id["2DEF"]["OVERALL_STATUS"], "NEED_JSON")
        self.assertEqual(summary["overall_complete"], 1)
        self.assertEqual(summary["valid_decoy_count"], 4)
        self.assertTrue(all(row["UNRESOLVED_VARIANT_COUNT"] == 2 for row in seed_rows[:2]))
        mapping = next(row for row in chain_rows if row["PDB_ID"] == "1ABC")
        self.assertEqual(mapping["PROTENIX_CHAIN_ID"], "A")
        self.assertEqual(mapping["ORIGINAL_CHAIN_ID"], "X")

    def test_runtime_json_does_not_overwrite_original(self) -> None:
        self.make_complete_target_with_stale_msa()
        _, updated_index = pipeline.index_json_files(self.simple)
        prep_index, _ = pipeline.index_output_dirs(self.complex)
        source = pipeline.choose_indexed_path(updated_index["1ABC"])
        prep_dir = pipeline.choose_indexed_path(prep_index["1ABC"])
        prep = pipeline.inspect_prep("1ABC", source, prep_dir)
        before = source.read_text(encoding="utf-8")
        runtime = pipeline.runtime_json("1ABC", prep, self.report)
        self.assertEqual(source.read_text(encoding="utf-8"), before)
        payload = json.loads(runtime.read_text(encoding="utf-8"))
        msa = payload[0]["sequences"][0]["rnaSequence"]["unpairedMsaPath"]
        self.assertTrue(Path(msa).is_file())

    def test_batch_json_rebases_msa_and_removes_json_seeds(self) -> None:
        self.make_complete_target_with_stale_msa()
        source = self.simple / "1abc-final-updated.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload[0]["modelSeeds"] = [999]
        source.write_text(json.dumps(payload), encoding="utf-8")
        _, updated_index = pipeline.index_json_files(self.simple)
        prep_index, _ = pipeline.index_output_dirs(self.complex)
        prep = pipeline.inspect_prep(
            "1ABC",
            pipeline.choose_indexed_path(updated_index["1ABC"]),
            pipeline.choose_indexed_path(prep_index["1ABC"]),
        )
        output = self.report / "batch.json"
        target = pipeline.Target("1ABC", "", True)
        pipeline.write_batch_json(output, [(target, prep)])
        batch = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(batch[0]["name"], "1abc")
        self.assertNotIn("modelSeeds", batch[0])
        msa = batch[0]["sequences"][0]["rnaSequence"]["unpairedMsaPath"]
        self.assertTrue(Path(msa).is_file())

    def test_indexes_resident_worker_layout(self) -> None:
        pred = self.pred / "1abc" / "seed_42" / "predictions"
        pred.mkdir(parents=True)
        _, index = pipeline.index_output_dirs(self.pred)
        self.assertIn(("1ABC", 42), index)
        self.assertEqual(pipeline.choose_indexed_path(index[("1ABC", 42)]).name, "seed_42")

    def test_load_pdb_id_file_normalizes_and_ignores_comments(self) -> None:
        path = self.root / "quarantine.txt"
        path.write_text("# known failures\n1abc\n 2def # gpu oom\n", encoding="utf-8")
        self.assertEqual(pipeline.load_pdb_id_file(path), {"1ABC", "2DEF"})

    def test_writes_readable_reports(self) -> None:
        self.make_complete_target_with_stale_msa()
        summary = pipeline.audit_and_write(self.args())
        self.assertFalse(summary["all_complete"])
        self.assertTrue((self.report / "decoy_report.xlsx").is_file())
        self.assertTrue((self.report / "decoy_manifest.csv").is_file())
        self.assertTrue((self.report / "decoy_seed_manifest.csv").is_file())
        self.assertTrue((self.report / "chain_id_mapping.csv").is_file())

    def test_cli_defaults_to_foldbench_budget(self) -> None:
        args = pipeline.build_parser().parse_args(["audit"])
        self.assertEqual(args.seeds, [42, 66, 101, 2024, 8888])
        self.assertEqual(args.samples, 5)
        self.assertEqual(args.pred_output_dir.name, "Foldbench_predictions")

    def test_missing_ranking_score_makes_seed_incomplete(self) -> None:
        self.make_complete_target_with_stale_msa()
        confidence = next(self.pred.rglob("*summary_confidence_sample_0.json"))
        confidence.write_text("{}", encoding="utf-8")
        info = pipeline.inspect_seed(confidence.parents[3], 2, "quick")
        self.assertEqual(info.status, "INCOMPLETE")
        self.assertIn("ranking_score", info.reason)

    def test_full_confidence_requires_all_five_fields(self) -> None:
        self.make_complete_target_with_stale_msa()
        seed_dir = next(self.pred.rglob("seed_42"))
        pred = seed_dir / "predictions"
        payload = {key: list(range(20)) for key in pipeline.FULL_DATA_KEYS}
        for rank in range(2):
            (pred / f"1abc_full_data_sample_{rank}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        complete = pipeline.inspect_seed(seed_dir, 2, "quick", True)
        self.assertEqual(complete.status, "COMPLETE")
        self.assertEqual(complete.full_data_count, 2)

        (pred / "1abc_full_data_sample_0.json").write_text(
            json.dumps({"atom_plddt": [0], "padding": "x" * 256}), encoding="utf-8"
        )
        incomplete = pipeline.inspect_seed(seed_dir, 2, "quick", True)
        self.assertEqual(incomplete.status, "INCOMPLETE")
        self.assertIn("缺少字段", incomplete.reason)


if __name__ == "__main__":
    unittest.main()
