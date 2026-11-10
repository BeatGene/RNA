"""Geometric, selection, train-only preprocessing and CLI integration checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

SCRIPT = Path(__file__).with_name("analyze_dataset_shift.py")
SPEC = importlib.util.spec_from_file_location("analyze_dataset_shift", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def payload(seed=1):
    rng = np.random.default_rng(seed)
    names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "N9", "C4"]
    atoms = [(r, n) for r in range(3) for n in names]
    lookup = {a: i for i, a in enumerate(atoms)}
    links = [("P", "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"),
             ("C3'", "O3'"), ("O4'", "C1'"), ("C1'", "N9"), ("N9", "C4")]
    pairs = [tuple(sorted((lookup[r, a], lookup[r, b]))) for r in range(3) for a, b in links]
    pairs += [(lookup[r, "O3'"], lookup[r+1, "P"]) for r in range(2)]
    xyz = rng.normal(size=(len(atoms), 3)) * 2
    native = xyz + rng.normal(size=xyz.shape) * 0.2
    mask = np.ones(len(atoms), bool)
    mask[0] = False
    native[0] = [9999, 9999, 9999]  # Must never enter supervised geometry.
    rmsd, _ = M.fitted_errors(xyz, native, mask)
    return dict(schema_version=2, sequence="AAA", pos_pred=torch.tensor(xyz), pos=torch.tensor(native),
                target_mask=torch.tensor(mask), residue_index=torch.tensor([r for r, _ in atoms]),
                atom_name_id=torch.tensor([M.ATOM_IDS[n] for _, n in atoms]), atomic_numbers=torch.full((len(atoms),), 6),
                atom_plddt=torch.tensor(rng.uniform(0.1, 1, len(atoms))), geometry_bond_index=torch.tensor(pairs).T,
                ideal_bond_length=torch.full((len(pairs),), 1.5), clash_exclusion_index=torch.tensor(pairs).T,
                predicted_label_seq_ids=["1", "2", "3"], prediction_to_native_residue_index=torch.arange(3),
                rnafm_embedding=torch.tensor(rng.normal(size=(3, 8))), ranking_score=seed/100,
                pre_refinement_aligned_rmsd=rmsd, dataset_max_pre_refinement_rmsd=30.0)


class ShiftTests(unittest.TestCase):
    def test_kabsch_excludes_unobserved_and_reflection(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(10, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, -1] *= np.linalg.det(q)
        y = x @ q + [3, 4, 5]
        y[-1] = 999
        mask = np.r_[np.ones(9, bool), False]
        rmsd, errors = M.fitted_errors(x, y, mask)
        self.assertLess(rmsd, 1e-12)
        self.assertTrue(np.isnan(errors[-1]))
        mirrored = x.copy(); mirrored[:, 0] *= -1
        self.assertGreater(M.fitted_errors(x, mirrored, np.ones(10, bool))[0], 0.1)

    def test_circular_wrap_and_degenerate_angles(self):
        error = np.degrees(M.circular_error(np.radians([179]), np.radians([-179])))
        self.assertAlmostEqual(error[0], 2)
        self.assertTrue(np.allclose(M.angle_hist(np.array([-np.pi])), M.angle_hist(np.array([np.pi]))))
        self.assertTrue(np.isnan(M.dihedral(np.zeros((1, 4, 3)))[0]))

    def test_torsions_do_not_cross_chain_break(self):
        p = payload()
        ids = M.torsion_indices(p)
        self.assertEqual(len(ids["alpha"]), 2)
        self.assertEqual(len(ids["chi"]), 3)
        p["predicted_label_seq_ids"] = ["1", "8", "9"]
        broken = M.torsion_indices(p)
        self.assertEqual(len(broken["alpha"]), 1)
        self.assertEqual(len(broken["eta"]), 0)

    def test_clashes_exclude_bonds_and_use_unique_pairs(self):
        xyz = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
        excluded = np.array([[0], [1]])
        result = M.clash_metrics(xyz, np.array([6, 6, 6]), excluded)
        self.assertEqual(result["clash_pair_count"], 2)
        self.assertAlmostEqual(result["clash_pairs_per_1000_atoms"], 2000/3)
        self.assertEqual(result["clash_atom_fraction"], 1)

    def test_rank1_ties_missing_scores_and_exposure(self):
        rows = [dict(pool="current", split="train", pdb_id="A", seed=301, sample=0, ranking_score=.8, length=51),
                dict(pool="current", split="train", pdb_id="A", seed=300, sample=1, ranking_score=.8, length=51),
                dict(pool="current", split="train", pdb_id="B", seed=300, sample=0, ranking_score=.9, length=20),
                dict(pool="current", split="train", pdb_id="B", seed=300, sample=1, ranking_score=np.nan, length=20)]
        rows += [dict(pool="current", split=s, pdb_id=s, seed=300, sample=0, ranking_score=.8, length=20) for s in ("val", "test")]
        views = M.make_views(pd.DataFrame(rows))
        self.assertEqual(views["train_rank1"].pdb_id.tolist(), ["A"])
        self.assertEqual(views["train_rank1"].seed.tolist(), [300])
        self.assertEqual(views["train_effective"].weight.tolist(), [2, 2, 1, 1])
        self.assertEqual(views["train_pdb_equal"].groupby("pdb_id").weight.sum().tolist(), [1, 1])

    def test_scaler_train_only_and_no_oracle_input(self):
        train = pd.DataFrame(dict(x=[0., 2, np.nan], constant=[1., 1, 1]))
        target = pd.DataFrame(dict(x=[1000.], constant=[999.]))
        a, b, cfg = M.bounded_reference(train, target, ["x", "constant"])
        self.assertEqual(cfg["medians"], [1])
        self.assertAlmostEqual(a.mean(), 0)
        self.assertGreater(b[0, 0], 100)
        self.assertFalse(set(M.INPUT_FEATURES) & set(M.ORACLE_FEATURES))

    def test_coverage_excludes_same_pdb(self):
        ref = np.array([[0.], [0.01], [10.], [10.01]])
        result = M.nearest_coverage(ref, np.array([[100.]]), np.array(["A", "A", "B", "B"]), 42)
        self.assertGreater(result["threshold"], 9)
        self.assertTrue(result["outside_reference_q95"][0])

    def test_weighted_shift_and_cluster_bootstrap(self):
        self.assertEqual(M.weighted_ks([0, 0], [0], [1, 1], [1]), 0)
        self.assertEqual(M.weighted_ks([0], [1], [1], [1]), 1)
        a = pd.DataFrame(dict(pdb_id=["A", "A", "B"], weight=[.5, .5, 1], x=[1., 1, 1]))
        b = pd.DataFrame(dict(pdb_id=["C", "D"], weight=[1., 1], x=[3., 3]))
        self.assertEqual(M.bootstrap_mean_difference(a, b, "x", 20, np.random.default_rng(1)), (2., 2.))

    def test_circular_histogram_distance_wraps(self):
        p, q = np.zeros(12), np.zeros(12)
        p[0], q[-1] = 1, 1
        self.assertEqual(M.circular_hist_wasserstein(p, q), 30)
        self.assertEqual(M.circular_hist_wasserstein(p, p), 0)

    def test_paired_pdb_bootstrap(self):
        a = pd.DataFrame(dict(pdb_id=["A", "B", "C"], weight=[1., 1, 1], x=[0., 10, 20]))
        b = a.copy(); b.x += 2
        lo, hi = M.bootstrap_mean_difference(a, b, "x", 30, np.random.default_rng(1), paired=True)
        self.assertAlmostEqual(lo, 2)
        self.assertAlmostEqual(hi, 2)

    def test_end_to_end_resume_and_report_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, out = root / "pt", root / "analysis"
            for split, pdbs in (("train", ["A", "B", "C"]), ("val", ["D"]), ("test", ["E"])):
                for i, pdb in enumerate(pdbs):
                    for sample in range(2 if split == "train" else 1):
                        path = data / split / pdb / "seed_300" / f"sample_{sample}.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(payload(i + sample + len(split)), path)
            command = [sys.executable, str(SCRIPT), "--data-root", str(data), "--output-dir", str(out), "--workers", "1", "--bootstrap", "10"]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["views"]["train_rank1"]["pdbs"], 3)
            self.assertEqual(summary["missing_scores"], 0)
            self.assertTrue((out / "pca_input.png").is_file())
            features = pd.read_csv(out / "features.tsv", sep="\t")
            self.assertEqual(len(features), 8)
            self.assertTrue(np.allclose(features.rmsd, features.stored_rmsd, atol=1e-7))
            self.assertTrue((features.beta_paired_count < features.pred_beta_count).all())
            second = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(json.loads((out / "extraction.json").read_text())["reused"], 8)
            report_only = subprocess.run(command + ["--report-only"], capture_output=True, text=True)
            self.assertEqual(report_only.returncode, 0, report_only.stdout + report_only.stderr)

    def test_archive_multiprocess_errors_and_performance_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, archive, out = root / "pt", root / "archive", root / "analysis"
            evaluation = []
            for split, pdbs in (("train", ["A", "B", "C"]), ("val", ["D"]), ("test", ["E"])):
                for i, pdb in enumerate(pdbs):
                    p = payload(i+len(split))
                    path = data / split / pdb / "seed_300" / "sample_0.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(p, path)
                    if split != "train":
                        for sample in range(2):
                            apath = archive / split / pdb / "seed_300" / f"sample_{sample}.pt"
                            apath.parent.mkdir(parents=True, exist_ok=True)
                            other = dict(p, ranking_score=p["ranking_score"] if sample == 0 else -1)
                            torch.save(other, apath)
                        evaluation.append(dict(split=split, pdb_id=pdb, protenix_seed=300, protenix_sample=0,
                            input_aligned_rmsd=p["pre_refinement_aligned_rmsd"], refined_aligned_rmsd=p["pre_refinement_aligned_rmsd"]-0.1,
                            aligned_improvement=0.1))
            # Ensure a corrupt first file does not break the streaming TSV schema.
            broken = data / "train" / "A" / "seed_300" / "sample_0.pt"
            original = broken.read_bytes()
            broken.write_bytes(b"corrupt pt")
            perf = root / "performance.tsv"
            pd.DataFrame(evaluation).to_csv(perf, sep="\t", index=False)
            command = [sys.executable, str(SCRIPT), "--data-root", str(data), "--archive-root", str(archive),
                       "--include-archive", "--output-dir", str(out), "--workers", "2", "--bootstrap", "10", "--performance-table", str(perf)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 2, first.stdout + first.stderr)
            self.assertEqual(json.loads((out / "summary.json").read_text())["extraction_errors"], 1)
            broken.write_bytes(original)
            second = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["views"]["val_all_pdb_equal"]["candidates"], 2)
            self.assertEqual(summary["selection_mismatches"], 0)
            self.assertEqual(summary["performance_link"]["matched_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
