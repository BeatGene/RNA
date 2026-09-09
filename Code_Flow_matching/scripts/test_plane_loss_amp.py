"""Exercise production base_plane_loss without importing Lightning/PyG.

AST extraction bypasses unrelated module imports, not the function body.
Check the dtype entering eigvalsh; CPU autocast may otherwise hide a BF16
covariance by automatically promoting the eigensolver input.
"""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

source = Path(__file__).resolve().parents[1] / "etflow/models/loss.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "base_plane_loss")
namespace = dict(torch=torch)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
base_plane_loss = namespace["base_plane_loss"]


class PlaneLossAmpTest(unittest.TestCase):
    def check_amp(self, device):
        torch.manual_seed(16)
        # Two RNA graphs with the same local residue id must stay separate.
        positions = torch.randn(12, 3, device=device, requires_grad=True)
        batch = torch.arange(2, device=device).repeat_interleave(6)
        residue = torch.zeros(12, dtype=torch.long, device=device)
        names = torch.arange(6, device=device).repeat(2)
        base_names = torch.arange(6, device=device)
        reference = base_plane_loss(positions, residue, names, batch, base_names)
        reference_grad, = torch.autograd.grad(reference, positions)
        eigvalsh = torch.linalg.eigvalsh
        seen = []

        def checked_eigvalsh(matrix):
            self.assertEqual(matrix.dtype, torch.float32)
            self.assertFalse(torch.is_autocast_enabled(device))
            seen.append(matrix.detach().clone())
            return eigvalsh(matrix)

        with patch.object(torch.linalg, "eigvalsh", checked_eigvalsh):
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = base_plane_loss(positions, residue, names, batch, base_names)
        self.assertEqual(len(seen), 2)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(bool(positions.grad.isfinite().all()))
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(positions.grad, reference_grad)

    def test_cpu_autocast_keeps_covariance_and_eigensolver_fp32(self):
        self.check_amp("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this machine")
    def test_cuda_autocast_keeps_covariance_and_eigensolver_fp32(self):
        self.check_amp("cuda")

    def test_planar_and_too_small_groups_have_finite_gradients(self):
        for n in (3, 4):
            with self.subTest(num_atoms=n):
                positions = torch.tensor([[-1., -1., 0.], [1., -1., 0.],
                                          [1., 1., 0.], [-1., 1., 0.]])[:n].requires_grad_()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = base_plane_loss(positions, torch.zeros(n, dtype=torch.long),
                                           torch.arange(n), None, torch.arange(n))
                loss.backward()
                self.assertEqual(float(loss.detach()), 0.)
                self.assertTrue(bool(positions.grad.isfinite().all()))


if __name__ == "__main__":
    unittest.main()
