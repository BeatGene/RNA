"""Small end-to-end checks for V3 report calculations without GPU packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("analyze_refinement_v3.py")
SELECT_SCRIPT = Path(__file__).with_name("select_refinement_v3_checkpoint.py")


def fixture_row(split: str, pdb_id: str, before: float, after: float) -> dict:
    row = {
        "split": split, "pdb_id": pdb_id, "native_chain_id": "A", "predicted_chain_id": "A",
        "length": 20, "mean_plddt": 0.8,
        "input_aligned_rmsd": before, "refined_aligned_rmsd": after,
        "aligned_improvement": before - after,
    }
    for name in ("bond", "clash", "plane"):
        row[f"input_{name}_loss_a2"] = 0.10
        row[f"refined_{name}_loss_a2"] = 0.20
    return row


def write_split(root: Path, split: str) -> Path:
    directory = root / split
    directory.mkdir()
    rows = [fixture_row(split, "7SXP", 1.5, 2.5), fixture_row(split, "8UPT", 6.0, 5.0)]
    with (directory / "samples.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(
        json.dumps({"split": split, "checkpoint": "/same.ckpt", "candidate_count": 2}),
        encoding="utf-8",
    )
    return directory


class AnalysisTests(unittest.TestCase):
    def test_low_rmsd_and_distribution_are_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            val = write_split(root, "val")
            test = write_split(root, "test")
            targets = root / "monomer_rna.csv"
            targets.write_text("pdb_id,chain_id\n7sxp-assembly1,A\n", encoding="utf-8")
            output = root / "analysis"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--val-dir", str(val),
                 "--test-dir", str(test), "--output-dir", str(output),
                 "--foldbench-targets", str(targets)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "rmsd_comparison.tsv").open(encoding="utf-8", newline="") as handle:
                comparisons = list(csv.DictReader(handle, delimiter="\t"))
            low = next(row for row in comparisons if row["split"] == "test" and
                       row["unit"] == "candidate" and row["group"] == "input RMSD <2 Å")
            self.assertEqual(low["count"], "1")
            self.assertEqual(float(low["share"]), 0.5)
            self.assertEqual(float(low["post_rmsd_ge_2_rate"]), 1.0)
            with (output / "rmsd_transitions.tsv").open(encoding="utf-8", newline="") as handle:
                transitions = list(csv.DictReader(handle, delimiter="\t"))
            moved = next(row for row in transitions if row["split"] == "test" and
                         row["unit"] == "candidate" and row["input_bin_a"] == "[0,2)" and
                         row["refined_bin_a"] == "[2,5)")
            self.assertEqual(moved["count"], "1")
            with (output / "foldbench_overlap.tsv").open(encoding="utf-8", newline="") as handle:
                overlap = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(overlap), 1)
            self.assertEqual(overlap[0]["foldbench_target_id"], "7sxp-assembly1")

    def test_checkpoint_selection_uses_matching_validation_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen = root / "screen"
            for name, score in (("a", 4.2), ("b", 3.8)):
                directory = screen / name
                directory.mkdir(parents=True)
                (directory / "summary.json").write_text(json.dumps({
                    "split": "val", "checkpoint": f"/{name}.ckpt",
                    "candidate_count": 1, "pdb_chain_count": 1,
                    "pdb_chain_macro_input_rmsd_mean": 5.0,
                    "pdb_chain_macro_refined_rmsd_mean": score,
                    "pdb_chain_macro_improvement_mean": 5.0 - score,
                    "pdb_chain_win_rate": 1.0,
                }), encoding="utf-8")
                (directory / "samples.tsv").write_text(
                    "pdb_id\tnative_chain_id\tprotenix_seed\tprotenix_sample\tsample_id\tsample_path\n"
                    "7SXP\tA\t300\t0\t7sxp_seed_300_sample_0\t/data/7sxp.pt\n",
                    encoding="utf-8",
                )
            output = root / "selected"
            command = [sys.executable, str(SELECT_SCRIPT), "--screen-root", str(screen),
                       "--output-dir", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "selected_checkpoint.txt").read_text().strip(), "/b.ckpt")
            sample_file = screen / "b" / "samples.tsv"
            sample_file.write_text(sample_file.read_text().replace("7SXP", "8UPT"), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different candidate identities", result.stderr)


if __name__ == "__main__":
    unittest.main()
