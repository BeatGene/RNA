"""Test the actual output block without importing the PyG training stack.

AST extraction only bypasses module-level PyG imports. All block operations,
parameters, forward and backward are the production implementation.
"""
import ast
from pathlib import Path
import unittest
import warnings

import torch
from torch import nn

source = Path(__file__).resolve().parents[1] / "etflow/networks/torchmd_net/utils.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GatedEquivariantBlock")
namespace = dict(torch=torch, nn=nn, warnings=warnings, act_class_mapping={"silu": nn.SiLU})
exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), namespace)
GatedEquivariantBlock = namespace["GatedEquivariantBlock"]


class VectorOutputAmpTest(unittest.TestCase):
    def check_bf16(self, device):
        torch.manual_seed(12)
        first = GatedEquivariantBlock(16, 8, scalar_activation=True).to(device)
        second = GatedEquivariantBlock(8, 16, vector_output=True).to(device)
        x = torch.randn(5, 16, device=device, requires_grad=True)
        vectors = torch.randn(5, 3, 16, device=device)
        vectors[0] = 0  # Exercise the masked zero-vector branch as well.
        vectors.requires_grad_()
        seen = []
        hook = first.update_net.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with torch.autocast(device, dtype=torch.bfloat16):
                hidden, v = first(x, vectors)
                scalar, output = second(hidden, v)
                loss = scalar.float().square().mean() + output.float().square().mean()
        hook.remove()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(seen[0].dtype, torch.float32)
        self.assertTrue(bool(loss.isfinite()))
        loss.backward()
        for value in (x, vectors, *first.parameters(), *second.parameters()):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(value.grad.isfinite().all()))

    def test_cpu_bf16_forward_backward(self):
        self.check_bf16("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this machine")
    def test_cuda_bf16_forward_backward(self):
        self.check_bf16("cuda")

    def test_float64_rotation_and_precision_preserved(self):
        torch.manual_seed(5)
        module = GatedEquivariantBlock(16, 8).double()
        x = torch.randn(5, 16, dtype=torch.double)
        v = torch.randn(5, 3, 16, dtype=torch.double)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
        q[:, -1] *= torch.det(q)
        rotate = lambda value: torch.einsum("ij,njc->nic", q, value)
        a, b = module(x, v)
        ar, br = module(x, rotate(v))
        self.assertEqual(b.dtype, torch.double)
        torch.testing.assert_close(a, ar, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(rotate(b), br, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
