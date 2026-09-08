import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("build_refinement_pt.py")
SPEC = importlib.util.spec_from_file_location("build_refinement_pt", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_cif(path: Path, rows: list[tuple], sequence: str = "AC") -> None:
    atom_lines = "\n".join(
        f"ATOM {index + 1} {element} '{atom}' {comp} {chain} 1 {residue} "
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

    def test_global_alignment_handles_missing_native_residue(self):
        mapping, identity, coverage = module.global_align("ACGU", "ACU")
        self.assertEqual(mapping, {0: 0, 1: 1, 3: 2})
        self.assertEqual(identity, 1.0)
        self.assertEqual(coverage, 0.75)


if __name__ == "__main__":
    unittest.main()
