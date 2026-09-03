"""End-to-end smoke test for the RNA refinement model using synthetic data.

Run from the Code_Flow_matching directory:

    python scripts/smoke_test_synthetic.py --device auto

The test writes two temporary ``.pt`` files, loads them through the real
EuclideanDataset/PyG DataLoader, runs all training modes, backpropagates one
loss, exercises one-step/multi-step sampling, and checks rotational
equivariance. It does not train a useful model or validate scientific quality.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from torch_geometric.loader import DataLoader
except ImportError as exc:
    raise SystemExit(
        "Missing torch_geometric. Install the project dependencies before "
        "running this smoke test."
    ) from exc

try:
    import torch_cluster  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Missing torch_cluster. Install a build compatible with the server's "
        "PyTorch/CUDA versions before running this smoke test."
    ) from exc

from etflow.data.constants import ATOM_NAME_TO_ID
from etflow.data.dataset import EuclideanDataset
from etflow.models.model import BaseFlow


NUM_EDGE_TYPES = 7
NODE_ATTR_DIM = 8


def build_clash_exclusion_index(
    geometry_bond_index: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    """Build unique undirected 1-2 and 1-3 exclusion pairs."""
    neighbors = [set() for _ in range(num_atoms)]
    one_two_pairs = set()

    for atom_i, atom_j in geometry_bond_index.t().tolist():
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i == atom_j:
            continue

        pair = tuple(sorted((atom_i, atom_j)))
        one_two_pairs.add(pair)
        neighbors[atom_i].add(atom_j)
        neighbors[atom_j].add(atom_i)

    one_three_pairs = set()
    for center_atom in range(num_atoms):
        for atom_i, atom_j in combinations(sorted(neighbors[center_atom]), 2):
            one_three_pairs.add(tuple(sorted((atom_i, atom_j))))

    exclusion_pairs = sorted(one_two_pairs | one_three_pairs)
    if not exclusion_pairs:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(exclusion_pairs, dtype=torch.long).t().contiguous()


def make_synthetic_sample(sample_index: int) -> dict:
    """Create two idealized six-membered RNA base rings."""
    generator = torch.Generator().manual_seed(20260903 + sample_index)

    angles = torch.arange(6, dtype=torch.float32) * (2.0 * math.pi / 6.0)
    ring = torch.stack(
        [1.35 * torch.cos(angles), 1.35 * torch.sin(angles), torch.zeros(6)],
        dim=1,
    )
    second_ring = ring + torch.tensor([4.0, 0.2, 0.0])
    pos = torch.cat([ring, second_ring], dim=0)

    # Apply a rigid translation per sample and a small non-planar perturbation
    # to the source structure. The model itself centers each graph.
    pos = pos + torch.tensor([2.0 * sample_index, -sample_index, 0.5])
    perturbation = 0.12 * torch.randn(pos.shape, generator=generator)
    perturbation[:, 2] += 0.08 * torch.sin(torch.arange(pos.size(0)).float())
    pos_pred = pos + perturbation

    base_atom_names = ("N1", "C2", "N3", "C4", "C5", "C6") * 2
    atom_name_id = torch.tensor(
        [ATOM_NAME_TO_ID[name] for name in base_atom_names],
        dtype=torch.long,
    )
    atomic_numbers = torch.tensor(
        [7 if name.startswith("N") else 6 for name in base_atom_names],
        dtype=torch.long,
    )
    residue_index = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)

    first_ring_bonds = [(atom_i, (atom_i + 1) % 6) for atom_i in range(6)]
    second_ring_bonds = [
        (6 + atom_i, 6 + (atom_i + 1) % 6) for atom_i in range(6)
    ]
    geometry_pairs = first_ring_bonds + second_ring_bonds + [(0, 9)]
    geometry_bond_index = torch.tensor(
        geometry_pairs,
        dtype=torch.long,
    ).t().contiguous()
    ideal_bond_length = torch.linalg.vector_norm(
        pos[geometry_bond_index[0]] - pos[geometry_bond_index[1]],
        dim=-1,
    )

    # Static graph: bidirectional covalent/phosphodiester edges plus one
    # sequence-neighbor edge. Type 3 remains reserved for dynamic radius edges.
    typed_pairs = [(atom_i, atom_j, 0) for atom_i, atom_j in geometry_pairs[:-1]]
    typed_pairs.append((*geometry_pairs[-1], 1))
    typed_pairs.append((3, 6, 2))

    directed_edges = []
    directed_types = []
    for atom_i, atom_j, edge_type in typed_pairs:
        directed_edges.extend([(atom_i, atom_j), (atom_j, atom_i)])
        directed_types.extend([edge_type, edge_type])

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = F.one_hot(
        torch.tensor(directed_types, dtype=torch.long),
        num_classes=NUM_EDGE_TYPES,
    ).float()

    node_attr = torch.randn(
        pos.size(0),
        NODE_ATTR_DIM,
        generator=generator,
    )

    return {
        "pos": pos.float(),
        "pos_pred": pos_pred.float(),
        "atomic_numbers": atomic_numbers,
        "sequence": "UC",
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_attr": node_attr.float(),
        "geometry_bond_index": geometry_bond_index,
        "ideal_bond_length": ideal_bond_length.float(),
        "residue_index": residue_index,
        "atom_name_id": atom_name_id,
        "clash_exclusion_index": build_clash_exclusion_index(
            geometry_bond_index=geometry_bond_index,
            num_atoms=pos.size(0),
        ),
    }


def validate_batch(batch) -> None:
    num_atoms = batch.pos.size(0)
    assert batch.pos.shape == batch.pos_pred.shape == (num_atoms, 3)
    assert batch.atomic_numbers.dtype == torch.long
    assert batch.edge_index.dtype == torch.long
    assert batch.geometry_bond_index.dtype == torch.long
    assert batch.clash_exclusion_index.dtype == torch.long
    assert batch.residue_index.dtype == torch.long
    assert batch.atom_name_id.dtype == torch.long
    assert batch.edge_attr.shape == (batch.edge_index.size(1), NUM_EDGE_TYPES)
    assert batch.ideal_bond_length.numel() == batch.geometry_bond_index.size(1)
    assert batch.node_attr.shape == (num_atoms, NODE_ATTR_DIM)
    assert batch.edge_index.numel() == 0 or int(batch.edge_index.max()) < num_atoms
    assert (
        batch.geometry_bond_index.numel() == 0
        or int(batch.geometry_bond_index.max()) < num_atoms
    )
    assert (
        batch.clash_exclusion_index.numel() == 0
        or int(batch.clash_exclusion_index.max()) < num_atoms
    )


def make_model(training_objective: str, flow_path: str, device: torch.device):
    model = BaseFlow(
        hidden_channels=32,
        num_layers=2,
        num_rbf=16,
        trainable_rbf=True,
        cutoff_upper=4.5,
        max_z=20,
        node_attr_dim=NODE_ATTR_DIM,
        edge_attr_dim=NUM_EDGE_TYPES,
        num_heads=4,
        dynamic_graph=True,
        max_num_neighbors=16,
        num_edge_types=NUM_EDGE_TYPES,
        dynamic_edge_type=3,
        so3_equivariant=True,
        source_conditioning=True,
        sigma=0.1,
        training_objective=training_objective,
        flow_path=flow_path,
        bond_loss_weight=0.1,
        clash_loss_weight=0.01,
        plane_loss_weight=0.1,
        lr_scheduler_type=None,
    ).to(device)

    # generic_step normally logs through a Lightning Trainer. The smoke test
    # calls it directly, so logging is intentionally disabled here.
    model.log_helper = lambda *args, **kwargs: None
    return model


def run_training_mode(batch, training_objective: str, flow_path: str, device):
    model = make_model(training_objective, flow_path, device)
    model.train()
    model.zero_grad(set_to_none=True)

    loss = model.generic_step(batch, batch_idx=0, stage="train")
    assert loss.ndim == 0
    assert torch.isfinite(loss), (
        f"Non-finite loss for {training_objective=}, {flow_path=}"
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients, "No parameter received a gradient"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    sample_steps = (1,) if training_objective == "residual" else (1, 3)
    for n_timesteps in sample_steps:
        prediction = model.sample(
            z=batch.atomic_numbers,
            pos_pred=batch.pos_pred,
            bond_index=batch.edge_index,
            batch=batch.batch,
            node_attr=batch.node_attr,
            edge_attr=batch.edge_attr,
            n_timesteps=n_timesteps,
        )
        assert prediction.shape == batch.pos.shape
        assert torch.isfinite(prediction).all()

    print(
        f"PASS objective={training_objective:<8} path={flow_path:<13} "
        f"loss={loss.detach().item():.6f}"
    )
    return model


@torch.no_grad()
def check_rotational_equivariance(model, batch) -> None:
    model.eval()
    angle = 0.71
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=batch.pos.dtype,
        device=batch.pos.device,
    )
    t = torch.full(
        (int(batch.batch.max().item()) + 1, 1),
        0.37,
        dtype=batch.pos.dtype,
        device=batch.pos.device,
    )

    output = model(
        z=batch.atomic_numbers,
        t=t,
        pos=batch.pos_pred,
        pos_source=batch.pos_pred,
        bond_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        node_attr=batch.node_attr,
        batch=batch.batch,
    )
    rotated_output = model(
        z=batch.atomic_numbers,
        t=t,
        pos=batch.pos_pred @ rotation.T,
        pos_source=batch.pos_pred @ rotation.T,
        bond_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        node_attr=batch.node_attr,
        batch=batch.batch,
    )

    expected_rotated_output = output @ rotation.T
    max_error = (rotated_output - expected_rotated_output).abs().max().item()
    assert max_error < 2.0e-4, f"Rotational equivariance error: {max_error:.3e}"
    print(f"PASS rotational equivariance max_error={max_error:.3e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")

    torch.manual_seed(7)
    print(f"PyTorch={torch.__version__} device={device}")

    with tempfile.TemporaryDirectory(prefix="etflow_synthetic_") as temp_dir:
        train_dir = Path(temp_dir) / "train"
        train_dir.mkdir(parents=True)
        for sample_index in range(2):
            torch.save(
                make_synthetic_sample(sample_index),
                train_dir / f"synthetic_{sample_index}.pt",
            )

        dataset = EuclideanDataset(data_dir=Path(temp_dir), split="train")
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        validate_batch(batch)
        batch = batch.to(device)
        print(
            f"PASS dataset/batching graphs={batch.num_graphs} "
            f"atoms={batch.num_nodes} static_edges={batch.edge_index.size(1)}"
        )

        residual_model = run_training_mode(
            batch=batch,
            training_objective="residual",
            flow_path="deterministic",
            device=device,
        )
        run_training_mode(batch, "flow", "deterministic", device)
        run_training_mode(batch, "flow", "stochastic", device)
        check_rotational_equivariance(residual_model, batch)

    print("ALL SYNTHETIC SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
