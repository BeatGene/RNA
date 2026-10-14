"""Regression for the probe's batched target-frame reconstruction.

Execute the actual centering utility and probe helper extracted from source,
without requiring the server-only Torch/PyG installation on this machine.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from test_fp32_graph_reductions import reductions


ROOT = Path(__file__).resolve().parents[1]


def function_from_source(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TensorArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def size(self, dim):
        return self.values.shape[dim]

    def __getitem__(self, indices):
        return TensorArray(self.values[indices])

    def __sub__(self, other):
        return TensorArray(self.values - other.values)


def scatter(values, groups, dim, reduce):
    if not isinstance(dim, int):
        raise AssertionError("batch was passed as dimension instead of batch keyword")
    if dim != 0 or reduce != "mean":
        raise AssertionError("unexpected centering operation")
    return TensorArray([values.values[groups == index].mean(axis=0) for index in range(groups.max() + 1)])


class ProbeRegressionTests(unittest.TestCase):
    def test_independent_graph_translations_do_not_change_residue_target_errors(self):
        production_center = function_from_source(ROOT / "etflow/models/utils.py", "center_of_mass",
                                                {"center_by_graph": reductions.center_by_graph})

        def centering(x, dim=0, batch=None):
            result = production_center(torch.tensor(x.values), dim=dim, batch=torch.tensor(batch))
            return TensorArray(result.numpy())
        helper = function_from_source(ROOT / "scripts/probe_rank1_mobility_gradients.py",
                                      "get_target_residue_errors", {"center_of_mass": centering})
        groups = np.array([0, 0, 1, 1])
        observed = np.array([True, True, True, True])
        batch = SimpleNamespace(
            pos_pred=TensorArray([[0, 0, 0], [2, 0, 0], [10, 4, 0], [14, 4, 0]]),
            pos=TensorArray([[5, 2, 0], [7, 2, 0], [110, -3, 0], [116, -3, 0]]),
            batch=groups, target_mask=observed,
        )

        def residue_errors(*, prediction, target, global_residue_index, total_residues, atom_mask):
            np.testing.assert_array_equal(global_residue_index, groups)
            np.testing.assert_array_equal(atom_mask, observed)
            self.assertEqual(total_residues, 2)
            square = np.square(prediction.values - target.values).sum(axis=1)
            return np.sqrt([square[groups == index].mean() for index in range(total_residues)]), np.array([True, True])

        errors, valid = helper(SimpleNamespace(residue_rms_error=residue_errors), batch,
                               {"global_residue_index": groups, "total_residues": 2})
        np.testing.assert_allclose(errors, [0, 1])
        np.testing.assert_array_equal(valid, [True, True])


if __name__ == "__main__":
    unittest.main()
