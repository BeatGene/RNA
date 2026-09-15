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


def write_cif(
    path: Path,
    rows: list[tuple],
    sequence: str | list[str] = "AC",
    quote_style: str = "single",
    component_parents: dict[str, str] | None = None,
    component_one_letters: dict[str, str] | None = None,
) -> None:
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
    component_ids = sorted(
        set(component_parents or {}) | set(component_one_letters or {})
    )
    component_section = ""
    if component_ids:
        component_rows = "\n".join(
            f"{component} {(component_parents or {}).get(component, '?')} "
            f"{(component_one_letters or {}).get(component, '?')}"
            for component in component_ids
        )
        component_section = f"""loop_
_chem_comp.id
_chem_comp.mon_nstd_parent_comp_id
_chem_comp.one_letter_code
{component_rows}
"""
    text = f"""data_test
{component_section}loop_
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

    def build_fixture(self, predicted, native, sequence, pred_style="single", native_style="double", native_sequence=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, target, confidence, fm = [root / n for n in ("pred.cif", "native.cif", "conf.json", "fm.pt")]
            write_cif(pred, predicted, sequence, pred_style)
            write_cif(target, native, sequence if native_sequence is None else native_sequence, native_style)
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

    def chain(self, sequence, label="A", auth="A", source="entity_poly_seq"):
        return module.NativeChain(label, auth, sequence, {}, source,
                                  tuple(str(i + 1) for i in range(len(sequence))))

    def test_strict_mapping_rejects_mismatch_even_at_high_identity(self):
        with self.assertRaisesRegex(ValueError, "differs") as caught:
            module.choose_native_chain([self.chain("ACGUUCGUAC")], "ACGUACGUAC")
        self.assertIn("prediction_sequence='ACGUACGUAC'", str(caught.exception))
        self.assertIn("'sequence': 'ACGUUCGUAC'", str(caught.exception))

    def test_strict_mapping_rejects_ambiguous_repeat_deletion(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            module.choose_native_chain([self.chain("ACAAA")], "ACAAAA")

    def test_strict_mapping_rejects_observed_only_even_when_identical(self):
        with self.assertRaisesRegex(ValueError, "observed-only"):
            module.choose_native_chain([self.chain("ACGU", source="observed_atoms_only")], "ACGU")

    def test_strict_chain_selection_never_breaks_ties_silently(self):
        chains = [self.chain("ACGU"), self.chain("ACGU", label="B", auth="B")]
        with self.assertRaisesRegex(ValueError, "ambiguous native chain"):
            module.choose_native_chain(chains, "ACGU")
        chosen, mapping, identity, coverage = module.choose_native_chain(chains, "ACGU", "B")
        self.assertEqual(chosen.label_chain, "B")
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual((identity, coverage), (1., 1.))
        with self.assertRaisesRegex(ValueError, "not found"):
            module.choose_native_chain(chains, "ACGU", "C")

    def test_conflicting_and_incomplete_entity_positions_are_rejected(self):
        for rows, message in (("1 1 A\n1 1 G\n", "ambiguous"), ("1 1 A\n1 3 G\n", "incomplete")):
            block = gemmi.cif.read_string("data_x\nloop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n" + rows).sole_block()
            with self.assertRaisesRegex(ValueError, message):
                module._entity_sequences(block, {})

    def test_repeats_with_complete_sequence_and_missing_coordinates_are_valid(self):
        rows = []
        for residue in range(1, 5):
            for name, xyz in [("P", (0., 0., 0.)), ("C4'", (1., 0., 0.)), ("C1'", (0., 1., 0.))]:
                rows.append((name[0], name, "A", "A", residue, xyz[0] + 5*residue, xyz[1], xyz[2]))
        sample = self.build_fixture(rows, [r for r in rows if r[4] != 2], "AAAA")
        self.assertEqual(sample["target_mask"].tolist(), [True]*3 + [False]*3 + [True]*6)
        self.assertEqual(sample["prediction_to_native_residue_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(sample["native_label_seq_ids"], ["1", "2", "3", "4"])
        self.assertEqual(sample["native_atom_site_row"].tolist(), [0, 1, 2, -1, -1, -1, 3, 4, 5, 6, 7, 8])
        self.assertEqual(sample["predicted_atom_site_row"].tolist(), list(range(12)))

    def test_builder_itself_rejects_sequence_mismatch(self):
        predicted = [("P", "P", "A", "A", 1, 0., 0., 0.),
                     ("C", "C4'", "A", "A", 1, 1., 0., 0.),
                     ("C", "C1'", "A", "A", 1, 0., 1., 0.)]
        native = [(e, a, "G", chain, r, x, y, z) for e, a, _, chain, r, x, y, z in predicted]
        with self.assertRaisesRegex(ValueError, "differs"):
            self.build_fixture(predicted, native, "A", native_sequence="G")

    def test_resume_rejects_old_mapping_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pt"
            torch.save({"schema_version": 2, "generator_version": "2.1-cif-decoding-purine-bonds"}, path)
            with self.assertRaisesRegex(ValueError, "current strict mapping"):
                module.load_resumable_sample(path)
            payload = {"schema_version": 2, "generator_version": module.GENERATOR_VERSION,
                       "mapping_policy": module.MAPPING_POLICY}
            torch.save(payload, path)
            self.assertEqual(module.load_resumable_sample(path), payload)

    def test_builder_rejects_atom_plddt_length_mismatch(self):
        rows = [("P", "P", "A", "A", 1, 0., 0., 0.),
                ("C", "C4'", "A", "A", 1, 1., 0., 0.),
                ("C", "C1'", "A", "A", 1, 0., 1., 0.)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, native = root / "pred.cif", root / "native.cif"
            confidence, fm = root / "conf.json", root / "fm.pt"
            write_cif(pred, rows, "A")
            write_cif(native, rows, "A")
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [0, 0, 0],
                "atom_plddt": [0.9, 0.9],
                "token_pair_pae": [[0.0]],
                "token_pair_pde": [[0.0]],
                "contact_probs": [[1.0]],
            }), encoding="utf-8")
            torch.save({"residue_embedding": torch.zeros(1, 640),
                        "sequences": ["A"], "chain_offsets": [0, 1]}, fm)
            with self.assertRaisesRegex(ValueError, "atom_plddt.*different lengths"):
                module.build_sample(pred, confidence, native, fm, "test", 0, 0)

    def test_adjacent_modified_residues_are_grouped_and_pair_features_are_meaned(self):
        residue_atoms = [
            ("P", "P", (0.0, 0.0, 0.0)),
            ("C", "C4'", (1.0, 0.0, 0.0)),
            ("C", "C1'", (0.0, 1.0, 0.0)),
            ("N", "N1", (0.0, 0.0, 1.0)),
            ("C", "C5M", (1.0, 1.0, 1.0)),
        ]
        predicted = []
        for residue, component in ((1, "PSU"), (2, "5MC")):
            for element, atom_name, xyz in residue_atoms:
                predicted.append((
                    element, atom_name, component, "A", residue,
                    xyz[0] + 5.0 * residue, xyz[1], xyz[2],
                ))
        native = [
            row[:-3] + (row[-3] + 10.0, row[-2] - 4.0, row[-1] + 2.0)
            for row in predicted
        ]
        pair = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        parents = {"PSU": "U", "5MC": "C"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, target = root / "pred.cif", root / "native.cif"
            confidence, fm = root / "conf.json", root / "fm.pt"
            # The prediction CIF intentionally has no _chem_comp metadata;
            # native metadata supplies the same CCD parent mapping available in
            # downloaded PDB CIFs.
            write_cif(pred, predicted, ["PSU", "5MC"])
            write_cif(
                target, native, ["PSU", "5MC"],
                component_parents=parents,
            )
            confidence.write_text(json.dumps({
                "atom_to_token_idx": list(range(10)),
                "atom_plddt": [0.9] * 10,
                "token_pair_pae": pair.tolist(),
                "token_pair_pde": (pair / 10.0).tolist(),
                "contact_probs": (pair / 99.0).tolist(),
            }), encoding="utf-8")
            torch.save({
                "residue_embedding": torch.zeros(2, 640),
                "sequences": ["UC"], "chain_offsets": [0, 2],
            }, fm)
            sample = module.build_sample(
                pred, confidence, target, fm, "mods", 0, 0
            )

        self.assertEqual(sample["sequence"], "UC")
        self.assertEqual(tuple(sample["pos"].shape), (8, 3))
        self.assertEqual(tuple(sample["pos_pred"].shape), (8, 3))
        self.assertEqual(sample["excluded_modification_atom_count"], 2)
        self.assertEqual(sample["modified_residue_mask"].tolist(), [True, True])
        self.assertEqual(
            sample["protenix_residue_token_offsets"].tolist(), [0, 5, 10]
        )
        expected = torch.tensor([
            [pair[:5, :5].mean(), pair[:5, 5:].mean()],
            [pair[5:, :5].mean(), pair[5:, 5:].mean()],
        ], dtype=torch.float16)
        torch.testing.assert_close(sample["token_pair_pae"], expected)

    def test_native_only_modified_atoms_never_create_coordinate_nodes(self):
        predicted = [
            ("P", "P", "U", "A", 1, 0.0, 0.0, 0.0),
            ("C", "C4'", "U", "A", 1, 1.0, 0.0, 0.0),
            ("C", "C1'", "U", "A", 1, 0.0, 1.0, 0.0),
            ("N", "N1", "U", "A", 1, 0.0, 0.0, 1.0),
        ]
        native = [
            (element, atom, "PSU", chain, residue, x + 8.0, y - 3.0, z + 2.0)
            for element, atom, _, chain, residue, x, y, z in predicted
        ] + [("C", "C5M", "PSU", "A", 1, 9.0, -2.0, 3.0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, target = root / "pred.cif", root / "native.cif"
            confidence, fm = root / "conf.json", root / "fm.pt"
            write_cif(pred, predicted, "U")
            write_cif(
                target, native, ["PSU"],
                component_parents={"PSU": "U"},
            )
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [0] * 4,
                "atom_plddt": [0.9] * 4,
                "token_pair_pae": [[0.0]],
                "token_pair_pde": [[0.0]],
                "contact_probs": [[1.0]],
            }), encoding="utf-8")
            torch.save({
                "residue_embedding": torch.zeros(1, 640),
                "sequences": ["U"], "chain_offsets": [0, 1],
            }, fm)
            sample = module.build_sample(
                pred, confidence, target, fm, "native_extra", 0, 0
            )

        self.assertEqual(tuple(sample["pos"].shape), (4, 3))
        self.assertEqual(tuple(sample["pos_pred"].shape), (4, 3))
        self.assertEqual(sample["native_atom_site_row"].tolist(), [0, 1, 2, 3])
        self.assertTrue(bool(sample["target_mask"].all()))

    def test_whole_modified_residue_missing_from_prediction_is_rejected(self):
        predicted = [
            ("P", "P", "A", "A", 1, 0.0, 0.0, 0.0),
            ("C", "C4'", "A", "A", 1, 1.0, 0.0, 0.0),
            ("C", "C1'", "A", "A", 1, 0.0, 1.0, 0.0),
            ("N", "N9", "A", "A", 1, 0.0, 0.0, 1.0),
        ]
        native = [
            row[:-3] + (row[-3] + 7.0, row[-2] - 2.0, row[-1] + 3.0)
            for row in predicted
        ] + [
            ("P", "P", "PSU", "A", 2, 12.0, -2.0, 3.0),
            ("C", "C4'", "PSU", "A", 2, 13.0, -2.0, 3.0),
            ("C", "C1'", "PSU", "A", 2, 12.0, -1.0, 3.0),
            ("N", "N1", "PSU", "A", 2, 12.0, -2.0, 4.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, target = root / "pred.cif", root / "native.cif"
            confidence, fm = root / "conf.json", root / "fm.pt"
            # The predicted entity declares the modified residue but has no
            # atom_site rows (and therefore no Protenix tokens) for it.
            write_cif(pred, predicted, ["A", "PSU"])
            write_cif(
                target, native, ["A", "PSU"],
                component_parents={"PSU": "U"},
            )
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [0] * 4,
                "atom_plddt": [0.9] * 4,
                "token_pair_pae": [[0.0]],
                "token_pair_pde": [[0.0]],
                "contact_probs": [[1.0]],
            }), encoding="utf-8")
            torch.save({
                "residue_embedding": torch.zeros(2, 640),
                "sequences": ["AU"], "chain_offsets": [0, 2],
            }, fm)
            with self.assertRaisesRegex(
                ValueError, "residues have no Protenix atom/token coordinates"
            ):
                module.build_sample(
                    pred, confidence, target, fm, "missing_mod", 0, 0
                )

    def test_inosine_is_explicitly_excluded(self):
        rows = [
            ("P", "P", "I", "A", 1, 0.0, 0.0, 0.0),
            ("C", "C4'", "I", "A", 1, 1.0, 0.0, 0.0),
            ("C", "C1'", "I", "A", 1, 0.0, 1.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, native = root / "pred.cif", root / "native.cif"
            confidence, fm = root / "conf.json", root / "fm.pt"
            write_cif(pred, rows, "I")
            write_cif(native, rows, "I")
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [0, 1, 2],
                "atom_plddt": [0.9] * 3,
                "token_pair_pae": torch.zeros(3, 3).tolist(),
                "token_pair_pde": torch.zeros(3, 3).tolist(),
                "contact_probs": torch.eye(3).tolist(),
            }), encoding="utf-8")
            torch.save({
                "residue_embedding": torch.zeros(1, 640),
                "sequences": ["I"], "chain_offsets": [0, 1],
            }, fm)
            with self.assertRaisesRegex(ValueError, "unsupported.*I/N/X/T"):
                module.build_sample(pred, confidence, native, fm, "inosine", 0, 0)

    def test_dry_run_builds_and_reports_without_writing_pt(self):
        rows = [("P", "P", "A", "A", 1, 0., 0., 0.),
                ("C", "C4'", "A", "A", 1, 1., 0., 0.),
                ("C", "C1'", "A", "A", 1, 0., 1., 0.)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_root = root / "predictions"
            native_root = root / "native"
            rnafm_root = root / "rnafm"
            output_root = root / "output"
            sample_dir = prediction_root / "train" / "1abc" / "seed_7" / "predictions"
            sample_dir.mkdir(parents=True)
            bad_sample_dir = prediction_root / "train" / "2def" / "seed_7" / "predictions"
            bad_sample_dir.mkdir(parents=True)
            native_root.mkdir()
            (rnafm_root / "1ABC").mkdir(parents=True)
            pred = sample_dir / "1abc_sample_0.cif"
            confidence = sample_dir / "1abc_full_data_sample_0.json"
            write_cif(pred, rows, "A")
            write_cif(native_root / "1abc.cif", rows, "A")
            write_cif(bad_sample_dir / "2def_sample_0.cif", rows, "A")
            confidence.write_text(json.dumps({
                "atom_to_token_idx": [0, 0, 0],
                "atom_plddt": [0.9, 0.9, 0.9],
                "token_pair_pae": [[0.0]],
                "token_pair_pde": [[0.0]],
                "contact_probs": [[1.0]],
            }), encoding="utf-8")
            torch.save({"residue_embedding": torch.zeros(1, 640),
                        "sequences": ["A"], "chain_offsets": [0, 1]},
                       rnafm_root / "1ABC" / "rnafm_t12_residue_embeddings.pt")

            result = module.main([
                "--prediction-root", str(prediction_root),
                "--native-root", str(native_root),
                "--rnafm-root", str(rnafm_root),
                "--output-root", str(output_root),
                "--split", "train",
                "--dry-run",
                "--run-name", "unittest_dry_run",
            ])

            self.assertEqual(result, 1)
            self.assertEqual(list(output_root.rglob("*.pt")), [])
            for split in ("train", "val", "test"):
                self.assertTrue((output_root / split).is_dir())
            run_dir = output_root / "logs" / "unittest_dry_run"
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["mode"], "dry_run")
            self.assertEqual(summary["dry_run_samples"], 1)
            self.assertEqual(summary["failed_samples"], 1)
            self.assertEqual(summary["problem_pdb_ids"], ["2DEF"])
            self.assertGreater(summary["estimated_total_pt_bytes"], 0)
            self.assertIsNotNone(summary["estimated_full_generation_seconds"])
            self.assertEqual(Path(summary["run_directory"]), run_dir)
            self.assertTrue((run_dir / "manifest.tsv").is_file())
            self.assertTrue((run_dir / "issues.tsv").is_file())
            self.assertTrue((run_dir / "run.log").is_file())
            self.assertIn(
                "2DEF", (run_dir / "issues.tsv").read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
