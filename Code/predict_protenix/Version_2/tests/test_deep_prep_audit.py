import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from Code.predict_protenix import deep_prep_audit as audit


def write_json(path: Path, pdb_id: str, sequence: str, msa_path: str = "") -> None:
    rna = {"sequence": sequence, "count": 1}
    if msa_path:
        rna["unpairedMsaPath"] = msa_path
    path.write_text(
        json.dumps(
            [
                {
                    "name": pdb_id.lower(),
                    "sequences": [{"rnaSequence": rna}],
                }
            ]
        ),
        encoding="utf-8",
    )


class DeepPrepAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.simple = self.root / "Simple_json"
        self.complex = self.root / "Complex_json"
        self.simple.mkdir()
        self.complex.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_case(
        self,
        pdb_id: str = "1ABC",
        json_sequence: str = "ACGU",
        query_sequence: str = "ACGU",
    ):
        raw = self.simple / f"{pdb_id.lower()}.json"
        final = self.simple / f"{pdb_id.lower()}-final-updated.json"
        stale = (
            f"/old/server/prep_output_{pdb_id.lower()}/{pdb_id.lower()}/"
            "rna_msa/0/rna_msa.a3m"
        )
        write_json(raw, pdb_id, json_sequence)
        write_json(final, pdb_id, json_sequence, stale)

        prep = self.complex / f"prep_output_{pdb_id.lower()}"
        msa = prep / pdb_id.lower() / "rna_msa" / "0" / "rna_msa.a3m"
        msa.parent.mkdir(parents=True)
        msa.write_text(
            f">query\n{query_sequence}\n>hit\n{query_sequence}\n",
            encoding="utf-8",
        )
        target = audit.Target(pdb_id, True, True)
        raw_index, final_index = audit.index_json_files(self.simple)
        prep_index = audit.index_prep_dirs(self.complex)
        return target, raw_index, final_index, prep_index

    def test_valid_rebased_prep_passes(self) -> None:
        target, raw_index, final_index, prep_index = self.make_case()
        pdb_row, rna_rows = audit.audit_target(
            target,
            raw_index,
            final_index,
            prep_index,
            {"1ABC": Counter({"ACGU": 1})},
        )
        self.assertEqual(pdb_row["STATUS"], "PASS")
        self.assertTrue(pdb_row["RAW_FINAL_RNA_ENTRIES_MATCH"])
        self.assertTrue(pdb_row["RAW_JSON_MANIFEST_SEQUENCE_MATCH"])
        self.assertEqual(rna_rows[0]["STATUS"], "PASS")
        self.assertEqual(rna_rows[0]["MSA_RESOLUTION"], "REBASED_INDEX")
        self.assertTrue(rna_rows[0]["A3M_QUERY_MATCH"])

    def test_wrong_a3m_query_fails(self) -> None:
        target, raw_index, final_index, prep_index = self.make_case(
            query_sequence="AAAA"
        )
        pdb_row, rna_rows = audit.audit_target(
            target,
            raw_index,
            final_index,
            prep_index,
            {"1ABC": Counter({"ACGU": 1})},
        )
        self.assertEqual(pdb_row["STATUS"], "FAIL")
        self.assertEqual(rna_rows[0]["STATUS"], "FAIL")
        self.assertFalse(rna_rows[0]["A3M_QUERY_MATCH"])
        self.assertIn("query", rna_rows[0]["MESSAGE"])

    def test_wrong_chain_manifest_sequence_fails(self) -> None:
        target, raw_index, final_index, prep_index = self.make_case()
        pdb_row, _ = audit.audit_target(
            target,
            raw_index,
            final_index,
            prep_index,
            {"1ABC": Counter({"GGGG": 1})},
        )
        self.assertEqual(pdb_row["STATUS"], "FAIL")
        self.assertFalse(pdb_row["RAW_JSON_MANIFEST_SEQUENCE_MATCH"])
        self.assertIn("RNA链清单", pdb_row["MESSAGE"])


if __name__ == "__main__":
    unittest.main()
