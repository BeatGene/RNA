"""Check FoldBench rank selection against the Protenix file layout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("export_foldbench_v3.py")


def load_exporter(payloads: dict[str, dict]):
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda path, **_kwargs: payloads[str(path)]
    fake_geometric = types.ModuleType("torch_geometric")
    fake_geometric_data = types.ModuleType("torch_geometric.data")
    fake_geometric_data.Batch = object
    fake_etflow = types.ModuleType("etflow")
    fake_etflow_data = types.ModuleType("etflow.data")
    fake_dataset = types.ModuleType("etflow.data.dataset")
    fake_dataset.EuclideanDataset = object
    fake_modules = {
        "gemmi": types.ModuleType("gemmi"),
        "torch": fake_torch,
        "torch_geometric": fake_geometric,
        "torch_geometric.data": fake_geometric_data,
        "evaluate_refinement": types.ModuleType("evaluate_refinement"),
        "etflow": fake_etflow,
        "etflow.data": fake_etflow_data,
        "etflow.data.dataset": fake_dataset,
    }
    spec = importlib.util.spec_from_file_location("export_foldbench_v3_tested", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class FoldBenchExportTests(unittest.TestCase):
    def test_selects_highest_summary_score_not_full_data_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {}
            samples = []
            for sample_number, score in ((0, 0.4), (1, 0.8)):
                cif = root / f"7sxp_pred_sample_{sample_number}.cif"
                cif.touch()
                full_data = root / f"7sxp_pred_full_data_sample_{sample_number}.json"
                summary = root / f"7sxp_pred_summary_confidence_sample_{sample_number}.json"
                full_data.write_text('{"atom_plddt": [0.7]}', encoding="utf-8")
                summary.write_text(json.dumps({"ranking_score": score}), encoding="utf-8")
                pt = root / f"sample_{sample_number}.pt"
                payloads[str(pt)] = {
                    "source_predicted_cif": str(cif),
                    "source_confidence_json": str(full_data),
                }
                samples.append({
                    "pdb_id": "7SXP", "native_chain_id": "V", "predicted_chain_id": "A",
                    "sample_path": str(pt), "protenix_seed": "300",
                    "protenix_sample": str(sample_number),
                    "input_aligned_rmsd": "2.0", "refined_aligned_rmsd": "1.9",
                })
            exporter = load_exporter(payloads)
            selected = exporter.select_candidates(
                samples, [{"pdb_id": "7sxp-assembly1", "chain_id": "A"}],
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["sample"], 1)
            self.assertEqual(selected[0]["ranking_score"], 0.8)
            self.assertEqual(selected[0]["native_chain_id"], "V")
            self.assertEqual(selected[0]["predicted_chain_id"], "A")
            self.assertEqual(Path(selected[0]["ranking_score_json"]).name,
                             "7sxp_pred_summary_confidence_sample_1.json")

    def test_missing_summary_score_reports_source_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            cif = Path(temporary) / "7sxp_pred_sample_0.cif"
            exporter = load_exporter({})
            with self.assertRaisesRegex(ValueError, "7sxp_pred_summary_confidence_sample_0.json"):
                exporter.ranking_score_from_source(cif, 0)

    def test_uses_original_build_manifest_score_when_json_was_moved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cif = root / "7zj4_sample_0.cif"
            cif.touch()
            pt = root / "sample_0.pt"
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "split\tpdb_id\tseed\tsample\tranking_score\toutput\n"
                f"test\t7ZJ4\t300\t0\t0.73\t{pt.resolve()}\n", encoding="utf-8",
            )
            exporter = load_exporter({str(pt): {"source_predicted_cif": str(cif)}})
            scores = exporter.read_manifest_scores([manifest], {"7ZJ4"})
            selected = exporter.select_candidates([{
                "pdb_id": "7ZJ4", "native_chain_id": "E", "predicted_chain_id": "A",
                "sample_path": str(pt), "protenix_seed": "300", "protenix_sample": "0",
                "input_aligned_rmsd": "4", "refined_aligned_rmsd": "3.9",
            }], [{"pdb_id": "7zj4-assembly1", "chain_id": "A"}], scores)
            self.assertEqual(selected[0]["ranking_score"], 0.73)
            self.assertEqual(selected[0]["ranking_score_json"], "")
            self.assertEqual(selected[0]["ranking_score_source"], str(manifest))

    def test_resolves_stale_val_cif_through_current_test_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "Data_V2" / "test" / "7zj4" / "seed_300" / "predictions"
            current.mkdir(parents=True)
            cif = current / "7zj4_sample_0.cif"
            cif.touch()
            summary = current / "7zj4_summary_confidence_sample_0.json"
            summary.write_text('{"ranking_score": 0.77}', encoding="utf-8")
            old = root / "Data_V2" / "val" / "7zj4" / "seed_300" / "predictions" / cif.name
            pt = root / "sample_0.pt"
            exporter = load_exporter({str(pt): {"source_predicted_cif": str(old)}})
            selected = exporter.select_candidates([{
                "pdb_id": "7ZJ4", "native_chain_id": "E", "predicted_chain_id": "A",
                "sample_path": str(pt), "protenix_seed": "300", "protenix_sample": "0",
                "input_aligned_rmsd": "4", "refined_aligned_rmsd": "3.9",
            }], [{"pdb_id": "7zj4-assembly1", "chain_id": "A"}],
                prediction_root=root / "Data_V2")
            self.assertEqual(selected[0]["source_cif"], str(cif))
            self.assertEqual(selected[0]["ranking_score"], 0.77)
            self.assertEqual(selected[0]["ranking_score_source"], str(summary))


if __name__ == "__main__":
    unittest.main()
