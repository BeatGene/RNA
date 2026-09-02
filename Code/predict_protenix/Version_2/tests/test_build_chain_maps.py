import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from openpyxl import Workbook

from build_chain_maps import (
    OUTPUT_COLUMNS,
    STATUS_PASS,
    STATUS_REVIEW,
    extract_coordinate_rna_chains,
    global_sequence_identity,
    main,
    make_chain_rows,
    ChainRecord,
)


SEEDS = (42, 66, 101, 2024, 8888)


def cif_text(label_chain: str, auth_chain: str, sequence: str = "ACGU") -> str:
    atom_rows = []
    for index, base in enumerate(sequence, start=1):
        atom_rows.append(
            f"ATOM {index} C \"C4'\" {base} {label_chain} 1 {index} "
            f"{index}.0 0.0 0.0 1.00 10.0 {index} {auth_chain}"
        )
    return f"""data_test
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
1 polyribonucleotide {sequence} {sequence}
loop_
_struct_asym.id
_struct_asym.entity_id
{label_chain} 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
{chr(10).join(atom_rows)}
"""


def protenix_cif_text(label_chain: str, sequence: str = "ACGU") -> str:
    atom_rows = []
    poly_seq_rows = []
    for index, base in enumerate(sequence, start=1):
        poly_seq_rows.append(f"1 {index} {base} n")
        atom_rows.append(
            f"ATOM {index} C \"C4'\" {base} {label_chain} 1 {index} "
            f"{index}.0 0.0 0.0 1.00 10.0 {index} {label_chain}"
        )
    return f"""data_predicted
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_strand_id
1 polyribonucleotide {label_chain}
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
{chr(10).join(poly_seq_rows)}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
{chr(10).join(atom_rows)}
"""


def write_targets(path: Path, pdb_ids: list[str]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Targets"
    worksheet.append(["target_id", "PDB_id"])
    for pdb_id in pdb_ids:
        worksheet.append([pdb_id, pdb_id])
    path.parent.mkdir(parents=True)
    workbook.save(path)


class BuildChainMapsTests(unittest.TestCase):
    def test_extracts_label_and_coordinate_auth_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1abc.cif"
            path.write_text(cif_text("X", "A"), encoding="utf-8")
            chains, issues = extract_coordinate_rna_chains(path)
            self.assertEqual(issues, [])
            self.assertEqual(len(chains), 1)
            self.assertEqual(chains[0].label_asym_id, "X")
            self.assertEqual(chains[0].auth_asym_id, "A")
            self.assertEqual(chains[0].sequence, "ACGU")

    def test_duplicate_sequences_require_review(self):
        predicted = [
            ChainRecord("A", "A", "ACGU"),
            ChainRecord("B", "B", "ACGU"),
        ]
        gt = [
            ChainRecord("X", "C", "ACGU"),
            ChainRecord("Y", "D", "ACGU"),
        ]
        rows, issues = make_chain_rows(
            ["ACGU", "ACGU"],
            predicted,
            gt,
            prediction_validation_ok=True,
            inherited_issues=[],
        )
        self.assertTrue(all(row["status"] == STATUS_REVIEW for row in rows))
        self.assertTrue(all(row["exact_match"] == "True" for row in rows))
        self.assertTrue(any("identical sequence occurs 2 times" in item for item in issues))

    def test_global_sequence_identity_includes_gap_columns(self):
        self.assertAlmostEqual(global_sequence_identity("ACGU", "ACU"), 0.75)
        self.assertAlmostEqual(global_sequence_identity("AAAA", "CCCC"), 0.0)
        self.assertAlmostEqual(global_sequence_identity("ACGU", "ACGU"), 1.0)

    def test_distinct_multichain_unique_mapping_passes(self):
        rows, issues = make_chain_rows(
            ["ACGU", "GGCA"],
            [ChainRecord("A", "A", "ACGU"), ChainRecord("B", "B", "GGCA")],
            [ChainRecord("X", "C", "GGCA"), ChainRecord("Y", "D", "ACGU")],
            prediction_validation_ok=True,
            inherited_issues=[],
        )
        self.assertTrue(all(row["status"] == STATUS_PASS for row in rows))
        self.assertEqual(rows[0]["gt_label_asym_id"], "Y")
        self.assertEqual(rows[1]["gt_label_asym_id"], "X")
        self.assertEqual(issues, [])

    def test_single_nonexact_chain_is_retained_for_review(self):
        rows, issues = make_chain_rows(
            ["ACGU"],
            [ChainRecord("A", "A", "ACGU")],
            [ChainRecord("X", "G", "ACGA")],
            prediction_validation_ok=True,
            inherited_issues=[],
        )
        self.assertEqual(rows[0]["gt_label_asym_id"], "X")
        self.assertEqual(rows[0]["gt_sequence"], "ACGA")
        self.assertEqual(rows[0]["identity"], "0.750000")
        self.assertEqual(rows[0]["exact_match"], "False")
        self.assertEqual(rows[0]["status"], STATUS_REVIEW)
        self.assertTrue(any("non-exact sequence" in item for item in issues))

    def test_reads_protenix_style_without_struct_asym_or_canonical_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.cif"
            path.write_text(protenix_cif_text("A"), encoding="utf-8")
            chains, issues = extract_coordinate_rna_chains(path)
            self.assertEqual(issues, [])
            self.assertEqual(chains, [ChainRecord("A", "A", "ACGU")])

    def test_end_to_end_single_chain_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "mapping" / "xlsx" / "targets.xlsx"
            json_dir = root / "Simple_json"
            gt_dir = root / "pdb_data"
            pred_dir = root / "Foldbench_predictions"
            output_dir = root / "mapping" / "tsv"
            write_targets(targets, ["1ABC"])
            json_dir.mkdir()
            gt_dir.mkdir()
            (json_dir / "1abc.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "1abc",
                            "sequences": [
                                {"rnaSequence": {"sequence": "ACGU", "count": 1}}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (gt_dir / "1abc.cif").write_text(cif_text("X", "G"), encoding="utf-8")
            for seed in SEEDS:
                predictions = (
                    pred_dir
                    / f"pred_output_1abc_seed_{seed}"
                    / "1abc"
                    / f"seed_{seed}"
                    / "predictions"
                )
                predictions.mkdir(parents=True)
                for sample in range(5):
                    (predictions / f"1abc_sample_{sample}.cif").write_text(
                        protenix_cif_text("A"), encoding="utf-8"
                    )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--targets-xlsx",
                        str(targets),
                        "--simple-json-dir",
                        str(json_dir),
                        "--gt-cif-dir",
                        str(gt_dir),
                        "--pred-dir",
                        str(pred_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            output = output_dir / "1ABC" / "chain_map.tsv"
            self.assertTrue(output.is_file())
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), OUTPUT_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pred_chain_id"], "A")
            self.assertEqual(rows[0]["gt_label_asym_id"], "X")
            self.assertEqual(rows[0]["gt_auth_asym_id"], "G")
            self.assertEqual(rows[0]["identity"], "1.000000")
            self.assertEqual(rows[0]["exact_match"], "True")
            self.assertEqual(rows[0]["status"], STATUS_PASS)
            self.assertIn("PDB directories written: 1", stdout.getvalue())
            self.assertIn("PASS targets: 1", stdout.getvalue())
            self.assertIn("需要人工审核 targets: 0", stdout.getvalue())

    def test_available_subset_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "mapping" / "xlsx" / "targets.xlsx"
            json_dir = root / "Simple_json"
            gt_dir = root / "pdb_data"
            pred_dir = root / "Foldbench_predictions"
            output_dir = root / "mapping" / "tsv"
            write_targets(targets, ["9R7W"])
            json_dir.mkdir()
            gt_dir.mkdir()
            (json_dir / "9r7w.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "9r7w",
                            "sequences": [
                                {"rnaSequence": {"sequence": "ACGU", "count": 1}}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (gt_dir / "9r7w.cif").write_text(cif_text("X", "R"), encoding="utf-8")
            predictions = (
                pred_dir
                / "pred_output_9r7w_seed_42"
                / "9r7w"
                / "seed_42"
                / "predictions"
            )
            predictions.mkdir(parents=True)
            for sample in range(5):
                (predictions / f"9r7w_sample_{sample}.cif").write_text(
                    protenix_cif_text("A"), encoding="utf-8"
                )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--targets-xlsx",
                        str(targets),
                        "--simple-json-dir",
                        str(json_dir),
                        "--gt-cif-dir",
                        str(gt_dir),
                        "--pred-dir",
                        str(pred_dir),
                        "--output-dir",
                        str(output_dir),
                        "--workers",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            with (output_dir / "9R7W" / "chain_map.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["status"], STATUS_PASS)
            log = stdout.getvalue()
            self.assertIn("missing prediction seeds=[66, 101, 2024, 8888]", log)
            self.assertIn("PASS targets: 1", log)


if __name__ == "__main__":
    unittest.main()
