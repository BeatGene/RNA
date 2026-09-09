import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import gemmi


SCRIPT = Path(__file__).with_name("build_refinement_pt.py")
SPEC = importlib.util.spec_from_file_location("build_refinement_pt", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_cif(path: Path, rows: list[tuple], sequence: str = "AC", quote_style: str = "single") -> None:
    def quoted(atom):
        if quote_style == "double":
            return '"' + atom + '"'
        if quote_style == "minimal":
            return gemmi.cif.quote(atom)
        return "'" + atom + "'"

    atom_lines = "\n".join(
        f"ATOM {index + 1} {element} {quoted(atom)} {comp} {chain} 1 {residue} "
        f"{residue} {chain} ? {x:.4f} {y:.4f} {z:.4f} 1.0 0.0 1"
        for index, (element, atom, comp, chain, residue, x, y, z) in enumerate(rows)
    )
    entity_rows = "\n".join(f"1 {index + 1} {symbol}" for index, symbol in enumerate(sequence))
    text = f"""data_test
loop_
_entity_poly.entity_id
_entity_poly.type
1 polyribonucleotide
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
{entity_rows}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
{atom_lines}
"""
    path.write_text(text, encoding="utf-8")


class BuildRefinementPtTest(unittest.TestCase):
    def test_alignment_and_missing_native_atom_mask(self):
        predicted = [
            ("P", "P", "A", "A", 1, 0.0, 0.0, 0.0),
            ("C", "C4'", "A", "A", 1, 1.0, 0.0, 0.0),
            ("C", "C1'", "A", "A", 1, 0.0, 1.0, 0.0),
            ("N", "N9", "A", "A", 1, 0.0, 0.0, 1.0),
            ("P", "P", "C", "A", 2, 3.0, 0.0, 0.0),
            ("C", "C4'", "C", "A", 2, 4.0, 0.0, 0.0),
            ("C", "C1'", "C", "A", 2, 3.0, 1.0, 0.0),
            ("N", "N1", "C", "A", 2, 3.0, 0.0, 1.0),
        ]
        # Native is translated; omit the last atom to emulate unresolved data.
        native = [row[:-3] + (row[-3] + 10, row[-2] - 4, row[-1] + 2) for row in predicted[:-1]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_cif, native_cif = root / "x_sample_0.cif", root / "x.cif"
            confidence_json, rnafm_pt = root / "x_full_data_sample_0.json", root / "fm.pt"
            write_cif(pred_cif, predicted)
            write_cif(native_cif, native)
            confidence_json.write_text(json.dumps({
                "atom_to_token_idx": [0] * 4 + [1] * 4,
                "atom_plddt": [0.9] * 8,
                "token_pair_pae": [[0.0, 1.0], [1.2, 0.0]],
                "token_pair_pde": [[0.0, 0.5], [0.5, 0.0]],
                "contact_probs": [[1.0, 0.7], [0.7, 1.0]],
            }), encoding="utf-8")
            torch.save({
                "residue_embedding": torch.randn(2, 640),
                "sequences": ["AC"],
                "chain_offsets": torch.tensor([0, 2]),
                "original_chain_ids": ["A"],
                "expected_protenix_chain_ids": ["A"],
                "model_name": "rna_fm_t12",
            }, rnafm_pt)
            sample = module.build_sample(pred_cif, confidence_json, native_cif, rnafm_pt, "1abc", 300, 0)

        self.assertEqual(sample["schema_version"], 2)
        self.assertEqual(tuple(sample["pos"].shape), (8, 3))
        self.assertEqual(int(sample["target_mask"].sum()), 7)
        self.assertFalse(bool(sample["target_mask"][-1]))
        torch.testing.assert_close(sample["pos"][-1], sample["pos_pred"][-1])
        torch.testing.assert_close(sample["pos"][sample["target_mask"]], sample["pos_pred"][sample["target_mask"]], atol=2e-4, rtol=0)
        self.assertEqual(tuple(sample["rnafm_embedding"].shape), (2, 640))
        self.assertEqual(tuple(sample["token_pair_pae"].shape), (2, 2))
        self.assertTrue(bool((sample["atom_name_id"] != 0).all()))
        self.assertEqual(sample["geometry_bond_index"].shape[1], 2)
        self.assertEqual(sample["edge_index"].shape[1], 6)

    def build_fixture(self, predicted, native, sequence, pred_style="single", native_style="double"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, target, confidence, fm = [root / n for n in ("pred.cif", "native.cif", "conf.json", "fm.pt")]
            write_cif(pred, predicted, sequence, pred_style)
            write_cif(target, native, sequence, native_style)
            count = len(sequence)
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [row[4] - 1 for row in predicted],
                "atom_plddt": [0.9] * len(predicted),
                "token_pair_pae": torch.zeros(count, count).tolist(),
                "token_pair_pde": torch.zeros(count, count).tolist(),
                "contact_probs": torch.eye(count).tolist(),
            }), encoding="utf-8")
            torch.save({"residue_embedding": torch.zeros(count, 640),
                        "sequences": [sequence], "chain_offsets": [0, count]}, fm)
            return module.build_sample(pred, confidence, target, fm, "test", 0, 0)

    def test_different_legal_cif_quotes_preserve_atom_identity_and_graph(self):
        rows = [("C", "C1'", "A", "A", 1, 0., 0., 0.),
                ("N", "N9", "A", "A", 1, 1.47, 0., 0.),
                ("C", "C4", "A", "A", 1, 1.47, 1.37, 0.)]
        for style in ("single", "double", "minimal"):
            with self.subTest(style=style):
                sample = self.build_fixture(rows, rows, "A", pred_style=style)
                self.assertTrue(bool(sample["target_mask"].all()))
                self.assertEqual(sample["atom_name_id"].tolist(),
                                 [module.ATOM_NAME_TO_ID[row[1]] for row in rows])
                self.assertEqual(sample["geometry_bond_index"].T.tolist(), [[0, 1], [1, 2]])
                self.assertIn([0, 2], sample["clash_exclusion_index"].T.tolist())

    def test_cif_string_columns_decode_quotes_and_preserve_nulls(self):
        block = gemmi.cif.read_string("data_x\nloop_\n_chem_comp.id\n_chem_comp.mon_nstd_parent_comp_id\n'PSU' 'U'\n'A' ?\n'G' .\n").sole_block()
        self.assertEqual(module._comp_parent_map(block), {"PSU": "U", "A": "?", "G": "."})

    def test_whole_missing_residue_uses_complete_entity_sequence(self):
        rows = []
        for residue, symbol in enumerate("ACG", start=1):
            for name, xyz in [("P", (0., 0., 0.)), ("C4'", (1., 0., 0.)),
                              ("C1'", (0., 1., 0.)), ("O4'", (0., 0., 1.))]:
                rows.append((name[0], name, symbol, "A", residue,
                             xyz[0] + residue * 5., xyz[1], xyz[2]))
        # Native still declares ACG in entity_poly_seq, but C has no atom rows.
        native = [r[:-3] + (r[-3]+8., r[-2]-3., r[-1]+2.) for r in rows if r[4] != 2]
        sample = self.build_fixture(rows, native, "ACG")
        self.assertEqual(sample["target_mask"].tolist(), [True]*4 + [False]*4 + [True]*4)
        self.assertEqual(sample["native_sequence_identity"], 1.)
        self.assertEqual(sample["native_sequence_coverage"], 1.)
        torch.testing.assert_close(sample["pos"], sample["pos_pred"], atol=2e-5, rtol=0)

    def test_purine_ring_closure_and_induced_exclusions(self):
        expected_base_ring = {frozenset(pair) for pair in (
            ("N9", "C8"), ("C8", "N7"), ("N7", "C5"), ("C5", "C4"), ("C4", "N9"),
            ("C5", "C6"), ("C6", "N1"), ("N1", "C2"), ("C2", "N3"), ("N3", "C4"))}
        names = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3", "C1'"]
        lookup = {(0, name): i for i, name in enumerate(names)}
        for symbol in ("A", "G"):
            with self.subTest(symbol=symbol):
                edges, _, bonds, lengths, exclusions = module.build_graph(symbol, lookup)
                actual = {frozenset((names[i], names[j])) for i, j in bonds.T.tolist()}
                self.assertEqual(actual, expected_base_ring | {frozenset(("C1'", "N9"))})
                self.assertEqual(edges.shape[1], 2 * len(actual))
                self.assertTrue(bool((lengths > 0).all()))
                excluded = {frozenset((names[i], names[j])) for i, j in exclusions.T.tolist()}
                self.assertTrue({frozenset(("N9", "C4")), frozenset(("C1'", "C4")),
                                 frozenset(("N9", "N3"))} <= excluded)

    def test_global_alignment_handles_missing_native_residue(self):
        mapping, identity, coverage = module.global_align("ACGU", "ACU")
        self.assertEqual(mapping, {0: 0, 1: 1, 3: 2})
        self.assertEqual(identity, 1.0)
        self.assertEqual(coverage, 0.75)


if __name__ == "__main__":
    unittest.main()
