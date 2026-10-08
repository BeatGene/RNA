"""Check the nine-target export contract and paired score report."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts import compare_foldbench_refinement_v3 as runner


def write_rows(path: Path, rows: list[dict], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class PairedFoldBenchTests(unittest.TestCase):
    def test_requires_nine_paired_targets_and_writes_epoch49_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = []
            input_refs = []
            refined_refs = []
            target_rows = []
            paired = []
            for index, target in enumerate(sorted(runner.TARGET_IDS)):
                source = root / f"input_{index}.cif"
                refined = root / f"refined_{index}.cif"
                source.touch()
                refined.touch()
                chosen = {
                    "foldbench_target_id": target, "predicted_chain_id": "A",
                    "seed": str(300 + index), "sample": "0",
                    "input_cif": str(source), "refined_cif": str(refined),
                }
                selected.append(chosen)
                target_rows.append({"pdb_id": target, "chain_id": "A"})
                common = {"pdb_id": target, "seed": chosen["seed"], "sample": "0"}
                input_refs.append({**common, "prediction_path": str(source)})
                refined_refs.append({**common, "prediction_path": str(refined)})
                paired.append({
                    "foldbench_target_id": target, "seed": chosen["seed"], "sample": "0",
                    "input_lddt": "0.5000", "refined_lddt": "0.6000", "lddt_change": "0.1000",
                })
            write_rows(root / "selected_candidates.tsv", selected, "\t")
            write_rows(root / "targets" / "monomer_rna.csv", target_rows, ",")
            write_rows(root / "evaluation" / runner.ALGORITHMS[0] / "prediction_reference.csv", input_refs, ",")
            write_rows(root / "evaluation" / runner.ALGORITHMS[1] / "prediction_reference.csv", refined_refs, ",")
            self.assertEqual(len(runner.validate_export(root)), 9)
            write_rows(root / "paired_lddt.tsv", paired, "\t")
            runner.write_comparison(root / "report", {runner.MODEL_NAME: runner.load_paired(root / "paired_lddt.tsv")})
            markdown = (root / "report" / "paired_lddt_all.md").read_text(encoding="utf-8")
            self.assertIn("0.5000 | 0.6000 | +0.1000 | 9 | 0", markdown)
            self.assertEqual(len(runner.read_delimited(root / "report" / "paired_lddt_all.tsv", "\t")), 9)
            selected.pop()
            write_rows(root / "selected_candidates.tsv", selected, "\t")
            with self.assertRaisesRegex(ValueError, "exactly 9"):
                runner.validate_export(root)


if __name__ == "__main__":
    unittest.main()
