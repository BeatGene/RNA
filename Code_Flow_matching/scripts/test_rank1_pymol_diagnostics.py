"""Numerical, mapping and actual headless-PyMOL checks for the triplet bundle.

Normal tests need numpy/gemmi. --pymol-smoke runs inside the local PyMOL
environment on a synthetic fixture; it requires no Torch or model weights.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock

import numpy as np
import repack_rank1_pymol_bundle as repack

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rna_diagnostics_tested", SCRIPTS / "build_rank1_pymol_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_cif(path, names, coordinates, residue_index, chain="V", residue_offset=101):
    import gemmi
    tags = ("group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
            "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
            "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
            "auth_atom_id", "pdbx_PDB_model_num")
    document = gemmi.cif.Document()
    loop = document.add_new_block("fixture").init_loop("_atom_site.", tags)
    for i, (name, xyz, residue) in enumerate(zip(names, coordinates, residue_index)):
        loop.add_row(["ATOM", str(i + 1), name[0], gemmi.cif.quote(name), ".", "A", chain,
                      "1", str(residue + 1), "?", *[f"{float(x):.4f}" for x in xyz],
                      "1.0", "20.0", str(residue + residue_offset), "A", chain,
                      gemmi.cif.quote(name), "1"])
    document.write_file(str(path))


def fixture(root, split="test", pdb="9TST"):
    directory = root / ("evaluation_" + split)
    structures = directory / "structures" / (pdb.lower() + "_seed_301_sample_2")
    structures.mkdir(parents=True)
    names = ["P", "C4'", "C3'", "O3'", "N9", "C8", "N7", "C5"] * 2
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1],
                       [2, 0, 0], [2, 1, 0], [3, 1, 0], [3, 0, 0]], dtype=float)
    target = np.concatenate([points, points + [5, 0, 0.2]])
    source = target.copy()
    source[4:8, 1] += 0.3
    source[12:16, 1] -= 0.2
    refined = source.copy()
    refined[7, 2] += 1.2
    refined[15, 2] += 0.7
    mask = np.ones(len(source), dtype=bool)
    mask[6] = False
    target[~mask] = source[~mask]
    residues = np.repeat([0, 1], 8)
    edges = np.array([(a + offset, b + offset) for offset in (0, 8)
                      for a, b in ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 4))]).T
    payload = {
        "native_structure_id": pdb, "protenix_seed": 301, "protenix_sample": 2,
        "sequence": "AA", "pos_pred": source, "pos": target, "target_mask": mask,
        "residue_index": residues, "atom_plddt": np.full(len(source), 0.8),
        "geometry_bond_index": edges, "ideal_bond_length": np.ones(edges.shape[1]),
        "predicted_atom_site_row": np.arange(len(source)),
        "native_atom_site_row": np.where(mask, np.arange(len(source)), -1),
        "native_to_prediction_rotation": np.eye(3), "native_to_prediction_translation": np.zeros(3),
    }
    payload_path = directory / "selected.pt"
    payload_path.touch()
    source_cif(structures / "input.cif", names, source, residues, chain="A", residue_offset=1)
    source_cif(structures / "refined.cif", names, refined, residues, chain="A", residue_offset=1)
    source_cif(structures / "native.cif", list(np.array(names)[mask]), target[mask], residues[mask])
    sample = {
        "split": split, "pdb_id": pdb, "sample_id": structures.name,
        "sample_path": str(payload_path), "protenix_seed": 301, "protenix_sample": 2,
        "native_chain_id": "V", "predicted_chain_id": "A", "length": 2,
        "atom_count": len(source), "observed_atom_count": int(mask.sum()),
        "input_aligned_rmsd": module.fit_coordinates(source, target, mask)[1],
        "refined_aligned_rmsd": module.fit_coordinates(refined, target, mask)[1],
        "input_backbone_aligned_rmsd": module.fit_coordinates(source, target, mask & np.isin(names, module.BACKBONE_NAMES))[1],
        "refined_backbone_aligned_rmsd": module.fit_coordinates(refined, target, mask & np.isin(names, module.BACKBONE_NAMES))[1],
        "input_bond_loss_a2": 0.01, "refined_bond_loss_a2": 0.2,
        "input_clash_loss_a2": 0.001, "refined_clash_loss_a2": 0.03,
        "input_plane_loss_a2": 0.0, "refined_plane_loss_a2": 0.1,
    }
    sample["aligned_improvement"] = sample["input_aligned_rmsd"] - sample["refined_aligned_rmsd"]
    structure = {
        "sample_id": sample["sample_id"],
        **{field: f"structures/{structures.name}/{name}.cif" for field, name in
           (("input_cif", "input"), ("refined_cif", "refined"), ("native_cif", "native"))},
        "source_native_cif": "original_native.cif", "source_predicted_cif": "original_input.cif",
    }
    selection = {"split": split, "pdb_id": pdb, "seed": 301, "sample": 2,
                 "ranking_score": 0.9, "candidate_count": 200, "complete_50x4": 1}
    module.write_tsv(directory / "samples.tsv", [sample])
    module.write_tsv(directory / "structure_manifest.tsv", [structure])
    module.write_json(directory / "summary.json", {
        "checkpoint": "epoch=49-step=157550.ckpt", "split": split, "candidate_count": 1,
    })
    return directory, payload, sample, structure, selection


def fake_torch(payloads):
    fake = types.ModuleType("torch")
    fake.load = lambda path, **kw: payloads[str(path)]
    return fake


def build_fixture_bundle(root):
    val = fixture(root, "val", "4TST")
    test = fixture(root, "test", "9TST")
    payloads = {str(item[2]["sample_path"]): item[1] for item in (val, test)}
    selection_path = root / "selection.tsv"
    module.write_tsv(selection_path, [val[4], test[4]])
    bundle = root / "pymol_bundle"
    with mock.patch.dict(sys.modules, {"torch": fake_torch(payloads)}):
        module.package(val[0], test[0], selection_path, bundle, {"val": 1, "test": 1}, True)
    return bundle


class DiagnosticTests(unittest.TestCase):
    def test_zip_clamps_1970_timestamp_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "pymol_bundle"
            bundle.mkdir()
            source = bundle / "old_file.py"
            source.write_bytes(b"original bytes\n")
            os.utime(source, (0, 0))
            original_mtime = source.stat().st_mtime
            destination = root / "repacked.zip"
            repack.write_bundle_zip(bundle, destination)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.getinfo("pymol_bundle/old_file.py").date_time[0], 1980)
                self.assertEqual(archive.read("pymol_bundle/old_file.py"), b"original bytes\n")
                self.assertIsNone(archive.testzip())
            self.assertEqual(source.stat().st_mtime, original_mtime)
            self.assertFalse(destination.with_name(destination.name + ".partial").exists())

    def test_zip_failure_does_not_publish_partial_or_overwrite_previous_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "pymol_bundle"
            bundle.mkdir()
            (bundle / "file").write_bytes(b"data")
            previous = root / "pymol_bundle.zip"
            previous.write_bytes(b"old incomplete zip")
            destination = root / "repacked.zip"
            with mock.patch.object(zipfile.ZipFile, "write", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    repack.write_bundle_zip(bundle, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".partial").exists())
            self.assertEqual(previous.read_bytes(), b"old incomplete zip")

    def test_recovery_validates_existing_triplets_without_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_fixture_bundle(Path(temp))
            repack.validate_bundle(bundle)
            target = bundle / "test/9TST/all_atom/native.cif"
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                repack.validate_bundle(bundle)

    def test_fits_rigid_motion_and_keeps_full_atom_set(self):
        rng = np.random.default_rng(5)
        target = rng.normal(size=(12, 3))
        theta = 0.7
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        mobile = target @ rotation + [10, -5, 8]
        observed = np.arange(len(target)) < 7
        fitted, rmsd, transform = module.fit_coordinates(mobile, target, observed)
        self.assertLess(rmsd, 1e-12)
        np.testing.assert_allclose(fitted, target, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform["rotation"]), 1.0)

    def test_does_not_fit_a_mirror_image_with_reflection(self):
        target = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], float)
        mobile = target * [-1, 1, 1]
        _, rmsd, transform = module.fit_coordinates(mobile, target, np.ones(4, bool))
        self.assertGreater(rmsd, 0.1)
        self.assertAlmostEqual(np.linalg.det(transform["rotation"]), 1.0)

    def test_missing_native_atoms_stay_missing_and_map_to_original_numbering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, payload, sample, structure, selection = fixture(root)
            destination = root / "output"
            with mock.patch.dict(sys.modules, {"torch": fake_torch({sample["sample_path"]: payload})}):
                module.prepare_target(directory, structure, sample, selection, destination)
            tags, rows, _ = module.read_cif(destination / "all_atom/native.cif")
            ids = [int(module.cif_value(tags, row, "id")) for row in rows]
            self.assertEqual(len(ids), 15)
            self.assertNotIn(7, ids)
            self.assertEqual(ids[-1], 16)
            atom_rows = module.read_tsv(destination / "atom_metrics.tsv")
            self.assertEqual(atom_rows[6]["input_error_all_atom_fit_a"], "")
            self.assertEqual(atom_rows[0]["source_native_resi"], "101")
            self.assertEqual(atom_rows[8]["source_native_resi"], "102")
            residue_rows = module.read_tsv(destination / "residue_metrics.tsv")
            self.assertEqual(residue_rows[0]["native_observed_atoms"], "7")
            self.assertGreater(float(residue_rows[1]["refined_base_plane_rms_a"]), 0.05)
            self.assertGreater(float(residue_rows[0]["mean_raw_movement_a"]), 0)

    def test_rank1_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, payload, _, _, selection = fixture(Path(temp))
            wrong = dict(selection, sample=1)
            with self.assertRaisesRegex(ValueError, "rank-1 identity mismatch"):
                module.validate_candidate(payload, wrong, "selected.pt")

    def test_bundle_contains_one_target_per_split_and_both_alignment_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_fixture_bundle(Path(temp))
            rows = module.read_tsv(bundle / "manifest.tsv")
            self.assertEqual(len(rows), 2)
            self.assertTrue((bundle.parent / "pymol_bundle.zip").is_file())
            for split, pdb in (("val", "4TST"), ("test", "9TST")):
                for mode in ("all_atom", "backbone"):
                    for name in ("native", "before", "after"):
                        self.assertTrue((bundle / split / pdb / mode / (name + ".cif")).is_file())
            self.assertIn(", module", (bundle / "view_test_9TST.pml").read_text())

    def test_duplicate_pdb_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            val, test = fixture(root, "val", "4TST"), fixture(root)
            module.write_tsv(val[0] / "samples.tsv", [val[2], val[2]])
            module.write_tsv(val[0] / "structure_manifest.tsv", [val[3], dict(val[3], sample_id="duplicate")])
            module.write_json(val[0] / "summary.json", {"checkpoint": "epoch=49-step=157550.ckpt", "split": "val", "candidate_count": 2})
            selection_path = root / "selection.tsv"
            module.write_tsv(selection_path, [val[4], test[4]])
            output = root / "output"
            with self.assertRaises(ValueError):
                module.package(val[0], test[0], selection_path, output, {"val": 2, "test": 1}, False)
            self.assertFalse(output.exists())


def pymol_smoke(bundle):
    # Use the installed open-source engine without opening a GUI.
    import pymol
    from pymol2 import SingletonPyMOL
    pymol.invocation.parse_args(["pymol", "-cq"])
    engine = SingletonPyMOL()
    engine.start()
    from pymol import cmd
    cmd.cd(str(bundle))
    cmd.do("@view_test_9TST.pml")
    for name, expected in (("native", 15), ("before", 16), ("after", 16)):
        assert cmd.count_atoms(name) == expected, (name, cmd.count_atoms(name))
    assert cmd.count_atoms("native and id 7") == 0
    assert cmd.count_atoms("before_observed") == 15
    expected_bonds = 14
    assert len(cmd.get_bonds("after")) == expected_bonds
    cmd.do("rna_sticks")
    cmd.do("rna_observed 1")
    cmd.do("rna_fit backbone")
    cmd.do("rna_fit all_atom")
    cmd.do("rna_residue 1, 2")
    cmd.do("rna_color_change")
    cmd.do("rna_worst 2")
    cmd.do("rna_next")
    cmd.do("rna_save")
    path = bundle / "sessions/test/9TST_all_atom.pse"
    assert path.is_file() and path.stat().st_size > 0
    cmd.reinitialize()
    cmd.load(str(path))
    assert cmd.count_atoms("after") == 16
    print("PYMOL_SMOKE_PASS actual_engine_loaded_triplets_bonds_views_saved_session")
    engine.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--make-fixture":
        location = Path(sys.argv[2]).resolve()
        location.mkdir(parents=True, exist_ok=False)
        print(build_fixture_bundle(location))
    elif len(sys.argv) > 1 and sys.argv[1] == "--pymol-smoke":
        pymol_smoke(Path(sys.argv[2]).resolve())
    else:
        unittest.main()
