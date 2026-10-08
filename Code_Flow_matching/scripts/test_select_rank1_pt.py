"""Test rank-one PT selection without loading model tensors."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

from scripts import select_rank1_pt as rank1


class RankOneSelectionTests(unittest.TestCase):
    def test_scores_all_samples_across_seeds_and_keeps_one_per_split_pdb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pt_root, pred_root = root / "Data_PT_V2", root / "Data_V2"
            scores = {
                ("val", "7sxp", 300, 0): 0.70,
                ("val", "7sxp", 300, 1): 0.60,
                ("val", "7sxp", 301, 0): 0.82,
                ("test", "8upt", 300, 0): 0.55,
            }
            for (split, pdb, seed, sample), score in scores.items():
                pt = pt_root / split / pdb / f"seed_{seed}" / f"sample_{sample}.pt"
                pt.parent.mkdir(parents=True, exist_ok=True)
                pt.touch()
                # The raw prediction can have moved from the PT split into test.
                summary = pred_root / "test" / pdb / f"seed_{seed}" / "predictions" / f"{pdb}_summary_confidence_sample_{sample}.json"
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps({"ranking_score": score}), encoding="utf-8")
            selected, coverage = rank1.select(rank1.discover(pt_root), pred_root, {}, "select")
            self.assertEqual(len(selected), 2)
            val = next(row for row in selected if row["split"] == "val")
            self.assertEqual((val["seed"], val["sample"], val["ranking_score"]), (301, 0, 0.82))
            self.assertEqual(val["candidate_count"], 3)
            self.assertEqual(len(coverage), 2)
            skipped, coverage = rank1.select(rank1.discover(pt_root), pred_root, {}, "skip")
            self.assertEqual(skipped, [])
            self.assertTrue(all(row["kept"] == 0 for row in coverage))

    def test_missing_score_fails_instead_of_silently_ranking_partial_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pt_root = root / "Data_PT_V2"
            for split in ("val", "test"):
                pt = pt_root / split / "7sxp" / "seed_300" / "sample_0.pt"
                pt.parent.mkdir(parents=True)
                pt.touch()
            with self.assertRaisesRegex(ValueError, "Ranking scores missing/invalid for 2"):
                rank1.select(rank1.discover(pt_root), root / "Data_V2", {}, "select")

    def test_manifest_fallback_matches_exact_pt_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pt = root / "Data_PT_V2" / "val" / "7sxp" / "seed_300" / "sample_0.pt"
            pt.parent.mkdir(parents=True)
            pt.touch()
            candidate = rank1.Candidate("val", "7SXP", 300, 0, pt)
            score, source = rank1.ranking_score(candidate, root / "Data_V2", {str(pt.resolve()): (0.91, "manifest.tsv")})
            self.assertEqual((score, source), (0.91, "manifest.tsv"))

    def test_main_writes_only_selected_pt_files_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pt_root, pred_root, output = root / "Data_PT_V2", root / "Data_V2", root / "rank1"
            for split, pdb in (("val", "7sxp"), ("test", "8upt")):
                for sample, score in ((0, 0.8), (1, 0.5)):
                    pt = pt_root / split / pdb / "seed_300" / f"sample_{sample}.pt"
                    pt.parent.mkdir(parents=True, exist_ok=True)
                    pt.write_text(f"{split}/{sample}", encoding="utf-8")
                    summary = pred_root / split / pdb / "seed_300" / "predictions" / f"{pdb}_summary_confidence_sample_{sample}.json"
                    summary.parent.mkdir(parents=True, exist_ok=True)
                    summary.write_text(json.dumps({"ranking_score": score}), encoding="utf-8")
            argv = ["select_rank1_pt.py", "--data-root", str(pt_root),
                    "--prediction-root", str(pred_root), "--output-root", str(output),
                    "--materialize", "copy"]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                rank1.main()
            self.assertEqual(len(list((output / "val").rglob("*.pt"))), 1)
            self.assertEqual(len(list((output / "test").rglob("*.pt"))), 1)
            self.assertEqual(len(list(rank1.discover(output))), 2)
            with (output / "selection.tsv").open(encoding="utf-8", newline="") as handle:
                selection = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual({row["sample"] for row in selection}, {"0"})
            self.assertEqual({row["candidate_count"] for row in selection}, {"2"})


if __name__ == "__main__":
    unittest.main()
