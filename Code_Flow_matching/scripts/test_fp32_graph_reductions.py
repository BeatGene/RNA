"""Test production reductions without importing Lightning or PyG."""
import ast
import importlib.util
from pathlib import Path
from typing import Optional
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graph_reductions", ROOT / "etflow/commons/reductions.py")
reductions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reductions)


def source_functions(relative, names, namespace):
    source = ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class GraphReductionTest(unittest.TestCase):
    def check_large_graphs(self, device):
        torch.manual_seed(12)
        sizes = [256, 576, 7984, 10629, 20000]
        batch = torch.arange(len(sizes), device=device).repeat_interleave(torch.tensor(sizes, device=device))
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(device=device, dtype=dtype):
                values = torch.randn(sum(sizes), 3, device=device).to(dtype).requires_grad_()
                expected_mean = torch.stack([values[batch == i].float().mean(0) for i in range(len(sizes))])
                with torch.autocast(device, dtype=torch.bfloat16):
                    mean = reductions.grouped_mean(values, batch)
                    centered = reductions.center_by_graph(values, batch)
                self.assertEqual(mean.dtype, torch.float32)
                self.assertEqual(centered.dtype, torch.float32)
                torch.testing.assert_close(mean, expected_mean, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(centered, values.float() - expected_mean[batch], atol=1e-6, rtol=1e-5)
                probe = torch.randn_like(centered)
                centered.mul(probe).sum().backward()
                expected_grad = probe - torch.stack([probe[batch == i].mean(0) for i in range(len(sizes))])[batch]
                torch.testing.assert_close(values.grad, expected_grad.to(dtype), atol=.016, rtol=.016)
                self.assertLess(float(reductions.center_by_graph(torch.full_like(values, .001), batch).abs().max()), 5e-7)

    def test_cpu_large_graphs(self):
        self.check_large_graphs("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this machine")
    def test_cuda_large_graphs(self):
        self.check_large_graphs("cuda")

    def test_double_gradcheck_translation_and_rotation(self):
        batch = torch.tensor([0, 1, 0, 1, 0])
        values = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: reductions.center_by_graph(x, batch), (values,)))
        shifts = torch.tensor([[12., -1., 3.], [-5., 8., -2.]], dtype=torch.double)
        centered = reductions.center_by_graph(values, batch)
        self.assertEqual(centered.dtype, torch.double)
        torch.testing.assert_close(reductions.center_by_graph(values + shifts[batch], batch), centered)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
        torch.testing.assert_close(reductions.center_by_graph(values @ q, batch), centered @ q)

    def test_nonzero_dim_empty_groups_and_no_batch(self):
        values = torch.tensor([[1., 3., 7.], [2., 4., 8.]])
        index = torch.tensor([0, 0, 2])
        expected = torch.tensor([[2., 0., 7.], [3., 0., 8.]])
        torch.testing.assert_close(reductions.grouped_mean(values, index, dim=1, dim_size=3), expected)
        torch.testing.assert_close(reductions.center_by_graph(values, index, dim=1), values - expected[:, index])
        torch.testing.assert_close(reductions.center_by_graph(values), values - values.mean(0))
        torch.testing.assert_close(reductions.grouped_mean(torch.empty(0, 3), torch.empty(0, dtype=torch.long)), torch.empty(0, 3))

    def test_gate_contraction_and_coordinate_gradient_bound(self):
        n = 10629
        batch = torch.zeros(n, dtype=torch.long)
        raw = reductions.center_by_graph(torch.randn(n, 3).bfloat16(), batch).detach().requires_grad_()
        gates = torch.linspace(.1, .99, n).bfloat16()[:, None]
        gated = reductions.center_by_graph(raw * gates, batch)
        self.assertLessEqual(float(gated.norm()), float(raw.norm()) + 1e-5)
        target = torch.randn_like(gated) + torch.tensor([2., -1., .5])
        grad, = torch.autograd.grad((gated - target).norm(dim=-1).mean(), raw)
        self.assertLessEqual(float(grad.norm()) * n ** .5, 1.0001)

    def test_actual_centering_wrappers(self):
        batch = torch.zeros(10629, dtype=torch.long)
        values = torch.full((10629, 3), .001, dtype=torch.bfloat16)
        for relative, names in [("etflow/models/utils.py", ["center_pos", "center_of_mass"]),
                                ("etflow/networks/torchmd_net/model_dynamics.py", ["center"])]:
            namespace = source_functions(relative, names, {"center_by_graph": reductions.center_by_graph})
            for name in names:
                result = namespace[name](values, batch=batch)
                self.assertEqual(result.dtype, torch.float32)
                self.assertLess(float(result.abs().max()), 5e-7)

    def test_coordinate_losses_large_masked_graphs(self):
        def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
            assert reduce == "sum"
            dim_size = dim_size if dim_size is not None else int(index.max()) + 1
            return src.new_zeros(dim_size).index_add(0, index, src)

        namespace = source_functions("etflow/models/loss.py", ["correct_tensor_shape", "batchwise_l2_loss", "batchwise_mse_loss"],
                                     dict(torch=torch, Optional=Optional, scatter=scatter,
                                          grouped_mean=reductions.grouped_mean, reduction_dtype=reductions.reduction_dtype))
        sizes = [576, 10629]
        batch = torch.arange(2).repeat_interleave(torch.tensor(sizes))
        values = torch.zeros(sum(sizes), 3, dtype=torch.bfloat16, requires_grad=True)
        target = torch.zeros_like(values)
        target[:sizes[0], 0] = 2
        target[sizes[0]:, 0] = 4
        mask = torch.ones(sum(sizes), dtype=torch.bool)
        mask[::3] = False
        for function, expected in [("batchwise_l2_loss", 3.), ("batchwise_mse_loss", 10.)]:
            loss = namespace[function](values, target, batch)
            self.assertEqual(loss.dtype, torch.float32)
            torch.testing.assert_close(loss, torch.tensor(expected))
        loss = namespace["batchwise_l2_loss"](values, target, batch, mask=mask)
        torch.testing.assert_close(loss, torch.tensor(3.))
        grad, = torch.autograd.grad(loss, values)
        self.assertTrue(bool(grad.isfinite().all()))
        self.assertEqual(float(grad[~mask].abs().max()), 0.)


if __name__ == "__main__":
    unittest.main()
