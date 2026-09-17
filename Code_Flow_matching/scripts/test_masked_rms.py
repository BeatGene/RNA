"""Dependency-light regression tests of the actual BaseFlow RMS method.

Extract the method unchanged via AST to avoid importing the Lightning/PyG
training stack on data-preparation machines. Its only PyG primitive here is
1-D scatter(sum), implemented with torch.index_add. The full stack is also
tested by smoke_test_synthetic.py on the training server.
"""
import ast
import unittest
from pathlib import Path
from typing import Optional

import torch


def scatter_sum(src, index, dim=0, dim_size=None, reduce="sum"):
    assert dim == 0 and reduce == "sum"
    return src.new_zeros(dim_size).index_add(0, index, src)


source = Path(__file__).resolve().parents[1] / "etflow/models/model.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseFlow")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "residue_rms_error")
method.decorator_list = []
namespace = dict(torch=torch, Tensor=torch.Tensor, Optional=Optional, scatter=scatter_sum)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
rms_error = namespace["residue_rms_error"]


class MaskedRmsTest(unittest.TestCase):
    def test_missing_residue_has_zero_finite_gradient(self):
        prediction = torch.tensor([[3., 4., 0.], [8., 6., 1.]], requires_grad=True)
        rms, valid = rms_error(prediction, torch.zeros_like(prediction), torch.tensor([0, 1]), 2,
                               torch.tensor([True, False]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(rms.tolist(), [5., 0.])
        rms[valid].mean().backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[.6, .8, 0.], [0., 0., 0.]]))

    def test_exact_match_has_zero_finite_gradient(self):
        prediction = torch.zeros(3, 3, requires_grad=True)
        rms, valid = rms_error(prediction, prediction.detach(), torch.tensor([0, 0, 1]), 2)
        rms[valid].sum().backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
        self.assertEqual(rms.tolist(), [0., 0.])

    def test_partial_mask_matches_observed_only_rms_and_gradient(self):
        prediction = torch.tensor([[3., 4., 0.], [0., 0., 0.], [100., 100., 100.]], dtype=torch.double, requires_grad=True)
        indices = torch.zeros(3, dtype=torch.long)
        mask = torch.tensor([True, True, False])
        rms, valid = rms_error(prediction, torch.zeros_like(prediction), indices, 1, mask)
        expected = torch.linalg.vector_norm(prediction[:2]) / (2 ** .5)
        torch.testing.assert_close(rms[0], expected)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: rms_error(x, torch.zeros_like(x), indices, 1, mask)[0], (prediction,)))
        rms.sum().backward()
        torch.testing.assert_close(prediction.grad[-1], torch.zeros(3, dtype=torch.double))

    def test_graph_indices_produce_per_sample_validation_rmsd(self):
        target = torch.zeros(4, 3)
        prediction = torch.tensor([
            [3., 4., 0.], [0., 0., 0.],
            [0., 0., 6.], [100., 100., 100.],
        ])
        graph_index = torch.tensor([0, 0, 1, 1])
        mask = torch.tensor([True, True, True, False])
        rms, valid = rms_error(
            prediction, target, graph_index, 2, mask
        )
        torch.testing.assert_close(
            rms,
            torch.tensor([5.0 / (2.0 ** 0.5), 6.0]),
        )
        self.assertEqual(valid.tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
