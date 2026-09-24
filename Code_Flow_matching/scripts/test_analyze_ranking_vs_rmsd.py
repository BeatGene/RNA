import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_ranking_vs_rmsd as analysis


class RankingVsRmsdTest(unittest.TestCase):
    def test_low_score_predicts_bad_and_old_json_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            pred = root / "pred"
            summary_path = pred / "train/1abc/seed_1/predictions/1abc_summary_confidence_sample_1.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps({"ranking_score": 0.1}), encoding="utf-8")
            with (run / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "pdb_id", "seed", "sample",
                                                           "pre_refinement_aligned_rmsd", "ranking_score"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"split": "train", "pdb_id": "1ABC", "seed": "1", "sample": "0",
                                 "pre_refinement_aligned_rmsd": "5", "ranking_score": "0.9"})
            with (run / "filtered_samples.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "pdb_id", "seed", "sample",
                                                           "pre_refinement_aligned_rmsd", "ranking_score"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"split": "train", "pdb_id": "1ABC", "seed": "1", "sample": "1",
                                 "pre_refinement_aligned_rmsd": "35", "ranking_score": ""})
            output = root / "out"
            with patch.object(sys, "argv", ["analyze", "--pt-run-dir", str(run),
                                            "--prediction-root", str(pred), "--output-dir", str(output)]):
                analysis.main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["auc_low_score_predicts_bad"], 1.0)
            self.assertEqual(summary["train"]["rmsd_gt_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
