"""End-to-end synthetic smoke test for the RNA refinement model.

Run from the Code_Flow_matching directory:

    python scripts/smoke_test_synthetic.py --device auto

This test exercises the real ``.pt`` data contract, PyG batching with
different token counts, confidence lookup, confidence on/off modes, all
training objectives, one-/multi-step sampling, validation errors, gradients,
and rotational equivariance. It does not assess scientific accuracy.
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
    from torch_geometric.utils import scatter
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

from etflow.data.constants import (
    ATOM_NAME_TO_ID,
    NUM_ATOM_NAME_TYPES,
    NUM_RNA_RESIDUE_TYPES,
    RNAFM_EMBEDDING_DIM,
    RNA_RESIDUE_TO_ID,
)
from etflow.data.dataset import EuclideanDataset
from etflow.models.model import BaseFlow
from etflow.models.utils import merge_dynamic_radius_edges


NUM_EDGE_TYPES = 7
CONFIDENCE_EDGE_DIM = 4
NODE_ATTR_DIM = (
    RNAFM_EMBEDDING_DIM
    + NUM_ATOM_NAME_TYPES
    + NUM_RNA_RESIDUE_TYPES
)
ATOM_NAMES_PER_RESIDUE = ("N1", "C2", "N3", "C4", "C5", "C6")


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
    """Create a 2- or 3-residue sample using the intended raw ``.pt`` schema."""
    generator = torch.Generator().manual_seed(20260904 + sample_index)
    num_residues = sample_index + 2
    atoms_per_residue = len(ATOM_NAMES_PER_RESIDUE)

    angles = torch.arange(atoms_per_residue, dtype=torch.float32)
    angles = angles * (2.0 * math.pi / atoms_per_residue)
    ring = torch.stack(
        [1.35 * torch.cos(angles), 1.35 * torch.sin(angles), torch.zeros(6)],
        dim=1,
    )
    rings = [
        ring + torch.tensor([4.0 * residue_id, 0.2 * residue_id, 0.0])
        for residue_id in range(num_residues)
    ]
    pos = torch.cat(rings, dim=0)
    pos = pos + torch.tensor([2.0 * sample_index, -sample_index, 0.5])
    perturbation = 0.12 * torch.randn(pos.shape, generator=generator)
    perturbation[:, 2] += 0.08 * torch.sin(torch.arange(pos.size(0)).float())
    pos_pred = pos + perturbation

    atom_names = ATOM_NAMES_PER_RESIDUE * num_residues
    atom_name_id = torch.tensor(
        [ATOM_NAME_TO_ID[name] for name in atom_names], dtype=torch.long
    )
    atomic_numbers = torch.tensor(
        [7 if name.startswith("N") else 6 for name in atom_names],
        dtype=torch.long,
    )
    residue_index = torch.arange(num_residues).repeat_interleave(
        atoms_per_residue
    )

    ring_bonds = []
    for residue_id in range(num_residues):
        offset = residue_id * atoms_per_residue
        ring_bonds.extend(
            (offset + atom_id, offset + (atom_id + 1) % atoms_per_residue)
            for atom_id in range(atoms_per_residue)
        )
    phosphodiester_bonds = [
        (
            residue_id * atoms_per_residue,
            (residue_id + 1) * atoms_per_residue + 3,
        )
        for residue_id in range(num_residues - 1)
    ]
    geometry_pairs = ring_bonds + phosphodiester_bonds
    geometry_bond_index = torch.tensor(
        geometry_pairs, dtype=torch.long
    ).t().contiguous()
    ideal_bond_length = torch.linalg.vector_norm(
        pos[geometry_bond_index[0]] - pos[geometry_bond_index[1]], dim=-1
    )

    typed_pairs = [(atom_i, atom_j, 0) for atom_i, atom_j in ring_bonds]
    typed_pairs.extend(
        (atom_i, atom_j, 1) for atom_i, atom_j in phosphodiester_bonds
    )
    typed_pairs.extend(
        (
            residue_id * atoms_per_residue + 3,
            (residue_id + 1) * atoms_per_residue + 3,
            2,
        )
        for residue_id in range(num_residues - 1)
    )

    directed_edges = []
    directed_types = []
    for atom_i, atom_j, edge_type in typed_pairs:
        directed_edges.extend([(atom_i, atom_j), (atom_j, atom_i)])
        directed_types.extend([edge_type, edge_type])
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = F.one_hot(
        torch.tensor(directed_types), num_classes=NUM_EDGE_TYPES
    ).float()

    rnafm_embedding = torch.randn(
        num_residues, RNAFM_EMBEDDING_DIM, generator=generator
    )
    if sample_index == 1:
        rnafm_embedding = rnafm_embedding.half()

    residue_letters = "UCGA"[:num_residues]
    residue_type_id = torch.tensor(
        [RNA_RESIDUE_TO_ID[letter] for letter in residue_letters],
        dtype=torch.long,
    )
    atom_plddt = torch.linspace(0.55, 0.95, pos.size(0))
    atom_to_token_idx = residue_index.clone()

    token_row = torch.arange(num_residues, dtype=torch.float32).view(-1, 1)
    token_col = torch.arange(num_residues, dtype=torch.float32).view(1, -1)
    token_pair_pae = 1.0 + 3.0 * token_row + token_col + sample_index
    token_pair_pde = 0.5 + token_row + 2.0 * token_col
    contact_probs = torch.exp(-(token_row - token_col).abs())

    return {
        "pos": pos.float(),
        "pos_pred": pos_pred.float(),
        "atomic_numbers": atomic_numbers,
        "sequence": residue_letters,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "geometry_bond_index": geometry_bond_index,
        "ideal_bond_length": ideal_bond_length.float(),
        "residue_index": residue_index.long(),
        "atom_name_id": atom_name_id,
        "clash_exclusion_index": build_clash_exclusion_index(
            geometry_bond_index=geometry_bond_index,
            num_atoms=pos.size(0),
        ),
        "rnafm_embedding": rnafm_embedding,
        "residue_type_id": residue_type_id,
        "atom_plddt": atom_plddt,
        "atom_to_token_idx": atom_to_token_idx.long(),
        "token_pair_pae": token_pair_pae,
        "token_pair_pde": token_pair_pde,
        "contact_probs": contact_probs,
    }


def validate_batch(batch) -> None:
    """Check dimensions, dtypes, local IDs, and PyG index increments."""
    num_atoms = batch.pos.size(0)
    assert batch.num_graphs == 2
    assert batch.num_tokens.tolist() == [2, 3]
    assert batch.pos.shape == batch.pos_pred.shape == (num_atoms, 3)
    assert batch.atomic_numbers.dtype == torch.long
    assert batch.edge_index.dtype == torch.long
    assert batch.geometry_bond_index.dtype == torch.long
    assert batch.clash_exclusion_index.dtype == torch.long
    assert batch.residue_index.dtype == torch.long
    assert batch.atom_name_id.dtype == torch.long
    assert batch.atom_to_token_idx.dtype == torch.long
    assert batch.edge_attr.shape == (batch.edge_index.size(1), NUM_EDGE_TYPES)
    assert batch.ideal_bond_length.numel() == batch.geometry_bond_index.size(1)
    assert batch.node_attr.shape == (num_atoms, NODE_ATTR_DIM)
    assert batch.node_attr.dtype == torch.float32
    assert batch.token_pair_confidence.shape == (2**2 + 3**2, 4)
    assert batch.atom_mobility_attr.shape == (num_atoms, 4)
    assert torch.isfinite(batch.node_attr).all()
    assert torch.isfinite(batch.token_pair_confidence).all()

    for index_name in (
        "edge_index",
        "geometry_bond_index",
        "clash_exclusion_index",
    ):
        index = getattr(batch, index_name)
        assert index.numel() == 0 or int(index.max()) < num_atoms

    second_graph_mask = batch.batch == 1
    assert batch.residue_index[second_graph_mask].min().item() == 0
    assert batch.residue_index[second_graph_mask].max().item() == 2
    assert batch.atom_to_token_idx[second_graph_mask].min().item() == 0
    assert batch.atom_to_token_idx[second_graph_mask].max().item() == 2

    atom_role = batch.node_attr[
        :, RNAFM_EMBEDDING_DIM:RNAFM_EMBEDDING_DIM + NUM_ATOM_NAME_TYPES
    ]
    residue_type = batch.node_attr[:, -NUM_RNA_RESIDUE_TYPES:]
    assert torch.equal(atom_role.sum(dim=-1), torch.ones(num_atoms))
    assert torch.equal(residue_type.sum(dim=-1), torch.ones(num_atoms))
    assert torch.equal(
        batch.node_attr[0, :RNAFM_EMBEDDING_DIM],
        batch.node_attr[5, :RNAFM_EMBEDDING_DIM],
    )

    expected_first_pair = torch.tensor(
        [2.0 / 32.0, 4.0 / 32.0, 2.5 / 32.0, math.exp(-1.0)]
    )
    assert torch.allclose(
        batch.token_pair_confidence[1], expected_first_pair, atol=1.0e-7
    )
    expected_first_mobility = torch.tensor(
        [
            1.5 / 32.0,
            2.5 / 32.0,
            1.5 / 32.0,
            (1.0 + math.exp(-1.0)) / 2.0,
        ]
    )
    assert torch.allclose(
        batch.atom_mobility_attr[0],
        expected_first_mobility,
        atol=1.0e-7,
    )


def check_dataset_validation(temp_dir: Path) -> None:
    invalid_dir = temp_dir / "invalid"
    invalid_dir.mkdir()
    invalid_sample = make_synthetic_sample(0)
    invalid_sample["atom_plddt"][0] = 1.5
    torch.save(invalid_sample, invalid_dir / "invalid_plddt.pt")
    invalid_dataset = EuclideanDataset(data_dir=temp_dir, split="invalid")
    try:
        invalid_dataset[0]
    except ValueError as exc:
        assert "atom_plddt" in str(exc)
    else:
        raise AssertionError("Out-of-range atom_plddt was not rejected")
    print("PASS dataset confidence validation")


def make_model(
    training_objective: str,
    flow_path: str,
    confidence_conditioning: bool,
    device: torch.device,
    use_mobility_v1: bool = False,
):
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
        # Keep synthetic distances away from the discontinuous radius cutoff.
        # One perturbed pair lies at 4.50054 A and is numerically ambiguous on
        # CUDA when the cutoff is exactly 4.5 A.
        dynamic_radius_cutoff=4.2,
        so3_equivariant=True,
        source_conditioning=True,
        confidence_conditioning=confidence_conditioning,
        use_mobility_v1=use_mobility_v1,
        sigma=0.1,
        training_objective=training_objective,
        flow_path=flow_path,
        bond_loss_weight=0.1,
        clash_loss_weight=0.01,
        plane_loss_weight=0.1,
        lr_scheduler_type=None,
    ).to(device)
    model.log_helper = lambda *args, **kwargs: None

    representation = model.network.representation_model
    expected_node_dim = NODE_ATTR_DIM + int(confidence_conditioning)
    expected_edge_dim = NUM_EDGE_TYPES + (
        CONFIDENCE_EDGE_DIM if confidence_conditioning else 0
    )
    assert representation.node_attr_dim == expected_node_dim
    assert representation.edge_attr_dim == expected_edge_dim
    return model


def confidence_kwargs(batch) -> dict:
    return {
        "atom_plddt": batch.atom_plddt,
        "atom_to_token_idx": batch.atom_to_token_idx,
        "token_pair_confidence": batch.token_pair_confidence,
        "num_tokens": batch.num_tokens,
    }


def mobility_kwargs(batch) -> dict:
    return {
        "atom_mobility_attr": batch.atom_mobility_attr,
        "geometry_bond_index": batch.geometry_bond_index,
        "ideal_bond_length": batch.ideal_bond_length,
        "clash_exclusion_index": batch.clash_exclusion_index,
    }


def run_training_mode(
    batch,
    training_objective: str,
    flow_path: str,
    confidence_conditioning: bool,
    device,
    use_mobility_v1: bool = False,
):
    model = make_model(
        training_objective,
        flow_path,
        confidence_conditioning,
        device,
        use_mobility_v1=use_mobility_v1,
    )
    model.train()
    model.zero_grad(set_to_none=True)

    loss = model.generic_step(batch, batch_idx=0, stage="train")
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "No parameter received a gradient"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if use_mobility_v1:
        mobility_gradients = [
            parameter.grad
            for parameter in model.mobility_head.parameters()
            if parameter.grad is not None
        ]
        assert mobility_gradients, "mobility_head received no gradient"
        assert any(
            gradient.abs().sum().item() > 0.0
            for gradient in mobility_gradients
        ), "mobility_head gradients are all zero"

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
            **confidence_kwargs(batch),
            **mobility_kwargs(batch),
        )
        assert prediction.shape == batch.pos.shape
        assert torch.isfinite(prediction).all()

    confidence_label = "on" if confidence_conditioning else "off"
    mobility_label = "on" if use_mobility_v1 else "off"
    print(
        f"PASS confidence={confidence_label:<3} "
        f"mobility_v1={mobility_label:<3} "
        f"objective={training_objective:<8} path={flow_path:<13} "
        f"loss={loss.detach().item():.6f}"
    )
    return model


@torch.no_grad()
def check_mobility_v1(model, batch) -> None:
    model.eval()
    num_graphs = int(batch.batch.max().item()) + 1
    t = torch.zeros(
        num_graphs,
        1,
        dtype=batch.pos.dtype,
        device=batch.pos.device,
    )
    velocity, auxiliary = model(
        z=batch.atomic_numbers,
        t=t,
        pos=batch.pos_pred,
        pos_source=batch.pos_pred,
        bond_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        node_attr=batch.node_attr,
        batch=batch.batch,
        return_aux=True,
        **confidence_kwargs(batch),
        **mobility_kwargs(batch),
    )
    assert velocity.shape == batch.pos.shape
    assert auxiliary["mobility_residue"].shape == (5, 1)
    assert auxiliary["mobility_atom"].shape == (batch.num_nodes, 1)
    assert torch.equal(
        auxiliary["mobility_atom"],
        auxiliary["mobility_residue"][auxiliary["global_residue_index"]],
    )
    assert (auxiliary["mobility_residue"] > 0).all()
    assert (auxiliary["mobility_residue"] < 1).all()
    assert torch.isfinite(velocity).all()

    x0 = batch.pos_pred.clone()
    x1 = batch.pos
    x0 = x0 - scatter(x0, batch.batch, dim=0, reduce="mean")[batch.batch]
    x1 = x1 - scatter(x1, batch.batch, dim=0, reduce="mean")[batch.batch]

    original_values = (
        model.identity_pair_probability,
        model.near_native_pair_probability,
        model.near_native_min_alpha,
        model.near_native_max_alpha,
    )
    try:
        model.identity_pair_probability = 1.0
        model.near_native_pair_probability = 0.0
        identity_source, identity_fraction, near_fraction = (
            model.augment_residual_source(x0, x1, batch.batch)
        )
        assert torch.allclose(identity_source, x1)
        assert identity_fraction.item() == 1.0
        assert near_fraction.item() == 0.0

        model.identity_pair_probability = 0.0
        model.near_native_pair_probability = 1.0
        model.near_native_min_alpha = 0.1
        model.near_native_max_alpha = 0.1
        near_source, identity_fraction, near_fraction = (
            model.augment_residual_source(x0, x1, batch.batch)
        )
        assert torch.allclose(
            near_source,
            x1 + 0.1 * (x0 - x1),
            atol=1.0e-6,
        )
        assert identity_fraction.item() == 0.0
        assert near_fraction.item() == 1.0
    finally:
        (
            model.identity_pair_probability,
            model.near_native_pair_probability,
            model.near_native_min_alpha,
            model.near_native_max_alpha,
        ) = original_values

    print(
        "PASS mobility_v1 residue gate and identity/near-native augmentation "
        f"mobility_mean={auxiliary['mobility_residue'].mean().item():.3f}"
    )


def check_mobility_configuration(device) -> None:
    try:
        make_model(
            training_objective="flow",
            flow_path="deterministic",
            confidence_conditioning=True,
            device=device,
            use_mobility_v1=True,
        )
    except ValueError as exc:
        assert "only supported" in str(exc)
    else:
        raise AssertionError(
            "use_mobility_v1=True with flow objective was not rejected"
        )
    print("PASS mobility_v1 residual-only configuration guard")


@torch.no_grad()
def check_confidence_pair_lookup(model, batch) -> None:
    """Verify flattened pair lookup across graphs with T=2 and T=3."""
    first_graph_atoms = int((batch.batch == 0).sum().item())
    atoms_per_residue = len(ATOM_NAMES_PER_RESIDUE)
    edge_index = torch.tensor(
        [
            [0, first_graph_atoms],
            [atoms_per_residue, first_graph_atoms + 2 * atoms_per_residue],
        ],
        dtype=torch.long,
        device=batch.pos.device,
    )
    actual = model.build_confidence_edge_attr(
        edge_index=edge_index,
        batch=batch.batch,
        atom_to_token_idx=batch.atom_to_token_idx,
        token_pair_confidence=batch.token_pair_confidence,
        num_tokens=batch.num_tokens,
    )
    expected = batch.token_pair_confidence[
        torch.tensor([1, 6], dtype=torch.long, device=batch.pos.device)
    ]
    assert torch.equal(actual, expected)
    print("PASS confidence pair lookup for variable-token batches")


@torch.no_grad()
def check_confidence_switch(enabled_model, disabled_model, batch) -> None:
    """The switch must use confidence when on and ignore it when off.

    Use the already batched static graph here so this unit check does not mix
    confidence sensitivity with CUDA radius-graph/scatter nondeterminism.
    Dynamic-graph execution is covered separately by every training/sampling
    test above.
    """
    num_graphs = int(batch.batch.max().item()) + 1
    t = torch.full((num_graphs, 1), 0.37, device=batch.pos.device)
    common = {
        "z": batch.atomic_numbers,
        "t": t,
        "pos": batch.pos_pred,
        "pos_source": batch.pos_pred,
        "bond_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "node_attr": batch.node_attr,
        "batch": batch.batch,
    }
    original = {
        "atom_plddt": batch.atom_plddt,
        "atom_to_token_idx": batch.atom_to_token_idx,
        "token_pair_confidence": batch.token_pair_confidence,
        "num_tokens": batch.num_tokens,
    }
    changed = {
        "atom_plddt": 1.0 - batch.atom_plddt,
        "atom_to_token_idx": batch.atom_to_token_idx,
        "token_pair_confidence": torch.zeros_like(batch.token_pair_confidence),
        "num_tokens": batch.num_tokens,
    }
    deliberately_invalid = {
        "atom_plddt": torch.full(
            (1,), float("nan"), device=batch.pos.device
        ),
        "atom_to_token_idx": torch.full(
            (1,), -1, dtype=torch.long, device=batch.pos.device
        ),
        "token_pair_confidence": torch.full(
            (1, 1), float("nan"), device=batch.pos.device
        ),
        "num_tokens": torch.full(
            (1,), -1, dtype=torch.long, device=batch.pos.device
        ),
    }

    enabled_dynamic_graph = enabled_model.dynamic_graph
    disabled_dynamic_graph = disabled_model.dynamic_graph
    enabled_model.dynamic_graph = False
    disabled_model.dynamic_graph = False
    try:
        enabled_a = enabled_model(**common, **original)
        enabled_b = enabled_model(**common, **changed)
        disabled_output = disabled_model(**common, **deliberately_invalid)
        disabled_repeat_a = disabled_model(**common, **original)
        disabled_repeat_b = disabled_model(**common, **original)
    finally:
        enabled_model.dynamic_graph = enabled_dynamic_graph
        disabled_model.dynamic_graph = disabled_dynamic_graph

    enabled_difference = (enabled_a - enabled_b).abs().max().item()
    disabled_repeat_noise = (
        disabled_repeat_a - disabled_repeat_b
    ).abs().max().item()
    assert enabled_difference > 1.0e-6
    assert torch.isfinite(disabled_output).all()
    print(
        "PASS confidence switch sensitivity "
        f"enabled_change={enabled_difference:.3e} "
        "disabled_ignored_invalid_inputs=True "
        f"repeat_noise={disabled_repeat_noise:.3e}"
    )


def check_confidence_validation(model, batch) -> None:
    num_graphs = int(batch.batch.max().item()) + 1
    t = torch.zeros((num_graphs, 1), device=batch.pos.device)
    base_kwargs = {
        "z": batch.atomic_numbers,
        "t": t,
        "pos": batch.pos_pred,
        "pos_source": batch.pos_pred,
        "bond_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "node_attr": batch.node_attr,
        "batch": batch.batch,
        **confidence_kwargs(batch),
    }
    missing_plddt = dict(base_kwargs)
    missing_plddt["atom_plddt"] = None
    try:
        model(**missing_plddt)
    except ValueError as exc:
        assert "atom_plddt" in str(exc)
    else:
        raise AssertionError("Missing atom_plddt was not rejected")

    malformed_pairs = dict(base_kwargs)
    malformed_pairs["token_pair_confidence"] = batch.token_pair_confidence[:-1]
    try:
        model(**malformed_pairs)
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("Malformed token_pair_confidence was not rejected")
    print("PASS confidence input validation")


@torch.no_grad()
def check_rotational_equivariance(model, batch) -> None:
    model.eval()
    angle = 0.71
    graph_rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=batch.pos.dtype,
        device=batch.pos.device,
    )
    rotated_graph_pos = batch.pos_pred @ graph_rotation.T
    edge_index, edge_attr = merge_dynamic_radius_edges(
        pos=batch.pos_pred,
        batch=batch.batch,
        bond_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        cutoff=model.cutoff,
        max_num_neighbors=model.max_num_neighbors,
        num_edge_types=model.num_edge_types,
        dynamic_edge_type=model.dynamic_edge_type,
    )
    rotated_edge_index, rotated_edge_attr = merge_dynamic_radius_edges(
        pos=rotated_graph_pos,
        batch=batch.batch,
        bond_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        cutoff=model.cutoff,
        max_num_neighbors=model.max_num_neighbors,
        num_edge_types=model.num_edge_types,
        dynamic_edge_type=model.dynamic_edge_type,
    )
    assert torch.equal(edge_index, rotated_edge_index), (
        "Dynamic radius graph changed after rotation; keep synthetic "
        "distances away from the radius cutoff and neighbor-cap boundary"
    )
    assert torch.equal(edge_attr, rotated_edge_attr)

    # Float32 CUDA scatter reductions can amplify last-bit differences across
    # otherwise identical forward launches. Use float64 for this mathematical
    # equivariance check; float32 CUDA execution is already exercised by all
    # training and sampling cases above.
    evaluation_dtype = (
        torch.float64 if batch.pos.is_cuda else batch.pos.dtype
    )
    rotation = graph_rotation.to(dtype=evaluation_dtype)
    pos = batch.pos_pred.to(dtype=evaluation_dtype)
    rotated_pos = pos @ rotation.T
    t = torch.full(
        (int(batch.batch.max().item()) + 1, 1),
        0.37,
        dtype=evaluation_dtype,
        device=batch.pos.device,
    )
    common = {
        "z": batch.atomic_numbers,
        "t": t,
        "bond_index": edge_index,
        "edge_attr": edge_attr.to(dtype=evaluation_dtype),
        "node_attr": batch.node_attr.to(dtype=evaluation_dtype),
        "batch": batch.batch,
        "atom_plddt": batch.atom_plddt.to(dtype=evaluation_dtype),
        "atom_to_token_idx": batch.atom_to_token_idx,
        "token_pair_confidence": batch.token_pair_confidence.to(
            dtype=evaluation_dtype
        ),
        "num_tokens": batch.num_tokens,
    }
    dynamic_graph = model.dynamic_graph
    original_dtype = next(model.parameters()).dtype
    model.dynamic_graph = False
    model.to(dtype=evaluation_dtype)
    try:
        output = model(
            pos=pos,
            pos_source=pos,
            **common,
        )
        rotated_output = model(
            pos=rotated_pos,
            pos_source=rotated_pos,
            **common,
        )
    finally:
        model.to(dtype=original_dtype)
        model.dynamic_graph = dynamic_graph
    max_error = (rotated_output - output @ rotation.T).abs().max().item()
    assert max_error < 2.0e-4, f"Rotational equivariance error: {max_error:.3e}"
    print(
        "PASS dynamic-graph rotation invariance and network equivariance "
        f"dtype={evaluation_dtype} max_error={max_error:.3e}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
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
        check_dataset_validation(Path(temp_dir))
        batch = batch.to(device)
        print(
            f"PASS dataset/batching graphs={batch.num_graphs} "
            f"atoms={batch.num_nodes} static_edges={batch.edge_index.size(1)} "
            f"node_dim={batch.node_attr.size(1)} pair_rows="
            f"{batch.token_pair_confidence.size(0)}"
        )

        models = {}
        for confidence_conditioning in (False, True):
            for training_objective, flow_path in (
                ("residual", "deterministic"),
                ("flow", "deterministic"),
                ("flow", "stochastic"),
            ):
                key = (confidence_conditioning, training_objective, flow_path)
                models[key] = run_training_mode(
                    batch,
                    training_objective,
                    flow_path,
                    confidence_conditioning,
                    device,
                )

        mobility_model = run_training_mode(
            batch=batch,
            training_objective="residual",
            flow_path="deterministic",
            confidence_conditioning=True,
            device=device,
            use_mobility_v1=True,
        )
        check_mobility_v1(mobility_model, batch)
        check_mobility_configuration(device)

        enabled_model = models[(True, "residual", "deterministic")]
        disabled_model = models[(False, "residual", "deterministic")]
        check_confidence_pair_lookup(enabled_model, batch)
        check_confidence_switch(enabled_model, disabled_model, batch)
        check_confidence_validation(enabled_model, batch)
        check_rotational_equivariance(enabled_model, batch)

    print("ALL SYNTHETIC SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
