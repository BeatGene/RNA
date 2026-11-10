import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import cached_rna_prep as cached
import stage2_decoy_pipeline as pipeline


def write_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["PDB_ID", "CURRENT_TARGET", "LEGACY_1979", "FILE_PATH"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"PDB_ID": "1ABC", "CURRENT_TARGET": "True", "LEGACY_1979": "", "FILE_PATH": ""},
                {"PDB_ID": "2DEF", "CURRENT_TARGET": "True", "LEGACY_1979": "", "FILE_PATH": ""},
            ]
        )


def write_raw(path: Path, name: str = "task") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [{"name": name, "sequences": [{"rnaSequence": {"sequence": "ACGU", "count": 1}}]}]
        ),
        encoding="utf-8",
    )


class CachedRnaPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.simple = self.root / "simple"
        self.complex = self.root / "complex"
        self.report = self.root / "report"
        self.official = self.root / "rna_msa"
        self.manifest = self.root / "manifest.csv"
        write_manifest(self.manifest)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            complex_json_dir=self.complex,
            report_dir=self.report,
        )

    def test_official_mapping_and_a3m_validation(self) -> None:
        source_id = "1abc_1"
        a3m = self.official / "msas" / source_id / f"{source_id}_all.a3m"
        a3m.parent.mkdir(parents=True)
        a3m.write_text(">query\nACGU\n>hit\nACGU\n", encoding="utf-8")
        (self.official / "rna_sequence_to_pdb_chains.json").write_text(
            json.dumps({"ACGU": [source_id]}), encoding="utf-8"
        )
        mapping = cached.load_official_mapping(self.official)
        source = cached.official_source(self.official, mapping, "ACGT")
        self.assertIsNotNone(source)
        self.assertEqual(source.source_type, "official")
        self.assertTrue(cached.valid_a3m(source.path, "ACGU"))

    def test_two_pdbs_get_two_jsons_but_share_one_msa(self) -> None:
        shared = self.root / "cache" / "rna_msa.a3m"
        shared.parent.mkdir(parents=True)
        shared.write_text(">query\nACGU\n", encoding="utf-8")
        source = cached.MsaSource("ACGU", shared, "official", "shared_1")
        targets = pipeline.load_targets(self.manifest)
        for target in targets:
            raw = self.simple / f"{target.pdb_id.lower()}.json"
            write_raw(raw, target.pdb_id.lower())
            ok, rows, _ = cached.materialize_target(
                target, raw, {"ACGU": source}, self.args()
            )
            self.assertTrue(ok)
            self.assertEqual(len(rows), 1)
        updated = sorted(self.simple.glob("*-final-updated.json"))
        self.assertEqual(len(updated), 2)
        paths = []
        for path in updated:
            payload = json.loads(path.read_text(encoding="utf-8"))
            paths.append(payload[0]["sequences"][0]["rnaSequence"]["unpairedMsaPath"])
        self.assertEqual(len(set(paths)), 1)

    def test_existing_complete_prep_builds_sequence_index(self) -> None:
        write_raw(self.simple / "1abc.json", "1abc")
        msa = self.complex / "prep_output_1abc" / "1abc" / "rna_msa" / "0" / "rna_msa.a3m"
        msa.parent.mkdir(parents=True)
        msa.write_text(">query\nACGU\n", encoding="utf-8")
        updated = self.simple / "1abc-final-updated.json"
        updated.write_text(
            json.dumps(
                [{"name": "1abc", "sequences": [{"rnaSequence": {
                    "sequence": "ACGU", "count": 1, "unpairedMsaPath": str(msa)
                }}]}]
            ),
            encoding="utf-8",
        )
        index = cached.build_existing_index(
            pipeline.load_targets(self.manifest), self.simple, self.complex
        )
        self.assertIn("ACGU", index)
        self.assertEqual(index["ACGU"].source_type, "existing")

    def test_database_specific_cache_index_is_verified(self) -> None:
        cache = self.root / "cache"
        database_set_id = "legacy"
        digest = cached.sequence_digest("ACGU")
        a3m = cache / database_set_id / "entries" / digest / "rna_msa.a3m"
        a3m.parent.mkdir(parents=True)
        a3m.write_text(">query\nACGU\n", encoding="utf-8")
        result = cached.build_cache_index({"ACGU", "GGGG"}, cache, database_set_id)
        self.assertEqual(set(result), {"ACGU"})
        self.assertEqual(result["ACGU"].source_type, "cache")


if __name__ == "__main__":
    unittest.main()
