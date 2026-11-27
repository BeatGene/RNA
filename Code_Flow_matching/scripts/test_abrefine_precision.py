"""ABRefine precision/DDP regressions; no dataset or training checkpoints needed.

Run from the project root: python -m unittest discover -s scripts -p test_abrefine_precision.py -v
CUDA BF16 is checked automatically when a compatible GPU is available.
"""
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from etflow.networks.factory import build_network
from etflow.networks.representations.abrefine_ops.edge_rot_mat import init_edge_rot_mat
from etflow.networks.representations.abrefine_ops.so3 import SO3_Embedding, SO3_Rotation


def small_network(source_conditioning=False):
    return build_network(
        network_type="TorchMDDynamics", representation_type="abrefine",
        output_head_type="abrefine", hidden_channels=8, num_layers=1,
        num_rbf=8, num_heads=2, cutoff_upper=6.0, max_z=20,
        node_attr_dim=674, edge_attr_dim=7,
        source_conditioning=source_conditioning,
        representation_args=dict(
            lmax_list=[2], mmax_list=[2], grid_resolution=6,
            attn_hidden_channels=8, attn_alpha_channels=4,
            attn_value_channels=4, ffn_hidden_channels=16, edge_channels=8,
            alpha_drop=0.0, drop_path_rate=0.0, proj_drop=0.0),
    )


def network_inputs(device="cpu"):
    # Two disjoint three-atom graphs, including both directions of every edge.
    generator = torch.Generator().manual_seed(20261127)
    return dict(
        z=torch.tensor([6, 7, 8, 6, 7, 8], device=device),
        t=torch.zeros(6, 1, device=device),
        pos=torch.tensor([[0., 0., 0.], [1., .2, .1], [.3, 1., .4],
                          [0., 0., 0.], [1., .3, .2], [.2, 1., .5]], device=device),
        pos_source=torch.zeros(6, 3, device=device),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                                 [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], device=device),
        node_attr=torch.randn(6, 674, generator=generator).to(device),
        edge_attr=torch.nn.functional.one_hot(torch.zeros(12, dtype=torch.long), 7).float().to(device),
    )


def check_gradients(model):
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    if missing:
        raise AssertionError(f"Unused trainable parameters: {missing}")
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise AssertionError(f"Nonfinite gradient: {name}")


def ddp_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store_path,
                            rank=rank, world_size=2)
    try:
        # Default find_unused_parameters=False must work over multiple updates.
        model = DistributedDataParallel(small_network(), find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            # Include no_sync accumulation, as used by the four-GPU baseline.
            with model.no_sync():
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model(**network_inputs()).square().mean() / 2
                loss.backward()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(**network_inputs()).square().mean() / 2
            loss.backward()
            check_gradients(model)
            optimizer.step()
    finally:
        dist.destroy_process_group()


class ABRefinePrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_source_switch_has_no_unused_parameters(self):
        for source_conditioning in (False, True):
            with self.subTest(source_conditioning=source_conditioning):
                model = small_network(source_conditioning)
                projection = model.representation_model.delta_projection
                self.assertEqual(projection is None, not source_conditioning)
                model(**network_inputs()).square().mean().backward()
                check_gradients(model)

    def test_cpu_bf16_forward_backward(self):
        model = small_network()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            velocity = model(**network_inputs())
            loss = velocity.square().mean()
        self.assertEqual(velocity.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        check_gradients(model)

    def test_full_baseline_bf16_vector_loss(self):
        from etflow.models.model import BaseFlow

        config = yaml.safe_load((PROJECT_ROOT / "config/RNA_train_abrefine_baseline.yaml").read_text())
        model = BaseFlow.from_config(config)
        inputs = network_inputs()
        inputs["t"] = torch.zeros(2, 1)
        inputs["pos_source"] = inputs["pos"].clone()
        inputs["bond_index"] = inputs.pop("edge_index")
        target_mask = torch.tensor([True, True, False, True, False, True])
        target = torch.linspace(-.3, .4, 18).reshape(6, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            velocity = model(**inputs)
            metrics = model.compute_losses(
                velocity, target, inputs["batch"], target_mask,
                inputs["pos_source"] + velocity, None, None, inputs["z"],
                None, None, None, inputs["pos_source"],
                inputs["pos_source"] + target, None)
        self.assertEqual(velocity.dtype, torch.float32)
        self.assertTrue(torch.isfinite(metrics["loss"]))
        torch.testing.assert_close(metrics["loss"], metrics["flow_matching_loss"])
        metrics["loss"].backward()
        check_gradients(model)

    def test_neighbor_sum_uses_fp32_and_keeps_gradients(self):
        generator = torch.Generator().manual_seed(42)
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    values = torch.randn(4096, 9, 2, generator=generator).to(device, dtype).requires_grad_()
                    index = torch.arange(4096, device=device) % 3
                    embedding = SO3_Embedding(0, [2], 2, device, dtype)
                    embedding.set_embedding(values)
                    with torch.autocast(device, dtype=torch.bfloat16):
                        embedding._reduce_edge(index, 4)
                    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                    self.assertEqual(embedding.embedding.dtype, expected_dtype)
                    self.assertEqual(embedding.dtype, expected_dtype)
                    reference = torch.zeros(4, 9, 2, device=device, dtype=torch.float64)
                    reference.index_add_(0, index, values.detach().double())
                    torch.testing.assert_close(embedding.embedding.double(), reference, atol=2e-4, rtol=2e-5)
                    embedding.embedding.square().mean().backward()
                    self.assertTrue(torch.isfinite(values.grad).all())
                    self.assertGreater(values.grad.abs().sum().item(), 0)

    def check_geometry(self, device, dtype):
        vectors = torch.tensor([[1., .3, -.2], [-.4, 1., .7]], device=device, dtype=dtype)
        rotation = SO3_Rotation(2).to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            frames = init_edge_rot_mat(vectors)
            rotation.set_wigner(frames)
            values = torch.ones(2, 9, 2, device=device, dtype=dtype, requires_grad=True)
            rotated = rotation.rotate(values, 2, 2)
            restored = rotation.rotate_inv(rotated, 2, 2)
        expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        for tensor in (frames, rotation.wigner, rotated, restored):
            self.assertEqual(tensor.dtype, expected_dtype)
        identity = torch.eye(9, device=device, dtype=expected_dtype).expand(2, -1, -1)
        torch.testing.assert_close(rotation.wigner @ rotation.wigner.transpose(-1, -2), identity,
                                   atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(restored, values.to(expected_dtype), atol=3e-5, rtol=3e-5)
        restored.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_geometry_precision_and_rotation_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.check_geometry("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_bf16_forward_backward_and_geometry(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA GPU does not support BF16")
        model = small_network().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = model(**network_inputs("cuda"))
            loss = velocity.square().mean()
        self.assertEqual(velocity.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        check_gradients(model)
        self.check_geometry("cuda", torch.bfloat16)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_ddp_bf16_with_accumulation(self):
        with tempfile.TemporaryDirectory(prefix="abrefine-ddp-") as directory:
            torch.multiprocessing.spawn(ddp_worker, args=(str(Path(directory) / "store"),),
                                        nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
