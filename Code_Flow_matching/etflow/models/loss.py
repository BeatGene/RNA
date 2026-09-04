"""Loss Functions"""

from typing import Optional
# Modify_3
from etflow.models.utils import build_dynamic_radius_graph
import torch
from torch_geometric.utils import scatter


def correct_tensor_shape(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return ((prediction - target) ** 2).sum(dim=-1).mean(dim=0)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return (torch.abs(prediction - target)).sum(dim=-1).mean(dim=0)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor):
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return torch.norm(prediction - target, p=2, dim=-1).mean(dim=0)


def batchwise_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    """Mean Squared Error Loss
    This computes the average MSE loss per molecule and then
    averages over number of molecules in the batch.
    """
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        ((prediction - target) ** 2).sum(dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)


def batchwise_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        torch.norm(prediction - target, p=2, dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)

# Modify_3
def bond_length_loss(
        prediction: torch.Tensor,
        geometry_bond_index: torch.Tensor,
        ideal_bond_length: torch.Tensor,
) -> torch.Tensor:
    if geometry_bond_index.numel() == 0:
        return prediction.sum() * 0.0

    atom_i = geometry_bond_index[0]
    atom_j = geometry_bond_index[1]

    predicted_bond_length = torch.linalg.vector_norm(
        prediction[atom_i] - prediction[atom_j],
        dim=-1,
    )

    ideal_bond_length = ideal_bond_length.to(
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(-1)

    return (
            predicted_bond_length
            - ideal_bond_length
    ).square().mean()


# Modify_5
def atomwise_bond_error(
        prediction: torch.Tensor,
        geometry_bond_index: torch.Tensor,
        ideal_bond_length: torch.Tensor,
) -> torch.Tensor:
    """Return one normalized source-bond error for every atom."""
    num_atoms = prediction.size(0)
    if geometry_bond_index.numel() == 0:
        return prediction.new_zeros((num_atoms, 1))

    atom_i = geometry_bond_index[0]
    atom_j = geometry_bond_index[1]
    ideal_bond_length = ideal_bond_length.to(
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(-1)

    predicted_bond_length = torch.linalg.vector_norm(
        prediction[atom_i] - prediction[atom_j],
        dim=-1,
    )
    relative_error = (
        predicted_bond_length - ideal_bond_length
    ).abs() / ideal_bond_length.clamp_min(1.0e-6)

    atom_index = torch.cat([atom_i, atom_j], dim=0)
    atom_error = torch.cat([relative_error, relative_error], dim=0)
    return scatter(
        atom_error,
        atom_index,
        dim=0,
        dim_size=num_atoms,
        reduce="mean",
    ).unsqueeze(-1)


# Modify_5
def atomwise_steric_clash_score(
        prediction: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        geometry_bond_index: torch.Tensor,
        vdw_radius_table: torch.Tensor,
        clash_cutoff: float = 4.5,
        clash_distance_scale: float = 0.8,
        clash_max_num_neighbors: int = 64,
        clash_exclusion_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return each atom's maximum normalized nonbonded overlap."""
    num_atoms = prediction.size(0)
    zero_score = prediction.new_zeros((num_atoms, 1))
    clash_edge_index = build_dynamic_radius_graph(
        pos=prediction,
        batch=batch,
        cutoff=clash_cutoff,
        max_num_neighbors=clash_max_num_neighbors,
    )
    if clash_edge_index.numel() == 0:
        return zero_score

    atom_i = clash_edge_index[0]
    atom_j = clash_edge_index[1]
    unique_pair_mask = atom_i < atom_j
    atom_i = atom_i[unique_pair_mask]
    atom_j = atom_j[unique_pair_mask]

    if clash_exclusion_index is None:
        clash_exclusion_index = geometry_bond_index

    candidate_pair_id = atom_i * num_atoms + atom_j
    exclusion_atom_i = torch.minimum(
        clash_exclusion_index[0],
        clash_exclusion_index[1],
    )
    exclusion_atom_j = torch.maximum(
        clash_exclusion_index[0],
        clash_exclusion_index[1],
    )
    excluded_pair_id = exclusion_atom_i * num_atoms + exclusion_atom_j
    nonbonded_mask = ~torch.isin(candidate_pair_id, excluded_pair_id)
    atom_i = atom_i[nonbonded_mask]
    atom_j = atom_j[nonbonded_mask]
    if atom_i.numel() == 0:
        return zero_score

    atom_radius = vdw_radius_table[atomic_numbers.view(-1)]
    supported_atom_mask = (
        (atom_radius[atom_i] > 0)
        & (atom_radius[atom_j] > 0)
    )
    atom_i = atom_i[supported_atom_mask]
    atom_j = atom_j[supported_atom_mask]
    if atom_i.numel() == 0:
        return zero_score

    predicted_distance = torch.linalg.vector_norm(
        prediction[atom_i] - prediction[atom_j],
        dim=-1,
    )
    minimum_distance = clash_distance_scale * (
        atom_radius[atom_i] + atom_radius[atom_j]
    )
    normalized_overlap = torch.relu(
        minimum_distance - predicted_distance
    ) / minimum_distance.clamp_min(1.0e-6)

    atom_index = torch.cat([atom_i, atom_j], dim=0)
    atom_overlap = torch.cat(
        [normalized_overlap, normalized_overlap],
        dim=0,
    )
    return scatter(
        atom_overlap,
        atom_index,
        dim=0,
        dim_size=num_atoms,
        reduce="max",
    ).unsqueeze(-1)

# Modify_3
def steric_clash_loss(
        prediction: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        geometry_bond_index: torch.Tensor,
        vdw_radius_table: torch.Tensor,
        clash_cutoff: float = 4.5,
        clash_distance_scale: float = 0.8,
        clash_max_num_neighbors: int = 64,
        clash_exclusion_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    clash_edge_index = build_dynamic_radius_graph(
        pos=prediction,
        batch=batch,
        cutoff=clash_cutoff,
        max_num_neighbors=clash_max_num_neighbors,
    )

    if clash_edge_index.numel() == 0:
        return prediction.sum() * 0.0

    atom_i = clash_edge_index[0]
    atom_j = clash_edge_index[1]

    # 双向边只保留一次。
    unique_pair_mask = atom_i < atom_j
    atom_i = atom_i[unique_pair_mask]
    atom_j = atom_j[unique_pair_mask]

    num_nodes = prediction.size(0)

    candidate_pair_id = atom_i * num_nodes + atom_j

    if clash_exclusion_index is None:
        clash_exclusion_index = geometry_bond_index

    exclusion_atom_i = torch.minimum(
        clash_exclusion_index[0],
        clash_exclusion_index[1],
    )
    exclusion_atom_j = torch.maximum(
        clash_exclusion_index[0],
        clash_exclusion_index[1],
    )

    excluded_pair_id = (
            exclusion_atom_i * num_nodes
            + exclusion_atom_j
    )

    nonbonded_mask = ~torch.isin(
        candidate_pair_id,
        excluded_pair_id,
    )

    atom_i = atom_i[nonbonded_mask]
    atom_j = atom_j[nonbonded_mask]

    if atom_i.numel() == 0:
        return prediction.sum() * 0.0

    atom_radius = vdw_radius_table[
        atomic_numbers.view(-1)
    ]

    supported_atom_mask = (
            (atom_radius[atom_i] > 0)
            & (atom_radius[atom_j] > 0)
    )

    atom_i = atom_i[supported_atom_mask]
    atom_j = atom_j[supported_atom_mask]

    if atom_i.numel() == 0:
        return prediction.sum() * 0.0

    predicted_distance = torch.linalg.vector_norm(
        prediction[atom_i] - prediction[atom_j],
        dim=-1,
    )

    minimum_distance = clash_distance_scale * (
            atom_radius[atom_i]
            + atom_radius[atom_j]
    )

    overlap = torch.relu(
        minimum_distance - predicted_distance
    )

    return overlap.square().mean()

# Modify_3
def base_plane_loss(
        prediction: torch.Tensor,
        residue_index: torch.Tensor,
        atom_name_id: torch.Tensor,
        batch: torch.Tensor,
        base_atom_name_ids: torch.Tensor,
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            prediction.size(0),
            dtype=torch.long,
            device=prediction.device,
        )

    base_atom_mask = torch.isin(
        atom_name_id.view(-1),
        base_atom_name_ids,
    )

    # 不同RNA都可能存在residue_index=0，
    # 因此必须结合batch区分残基。
    residue_key = torch.stack(
        [
            batch.view(-1),
            residue_index.view(-1),
        ],
        dim=1,
    )

    _, residue_group = torch.unique(
        residue_key,
        dim=0,
        return_inverse=True,
    )

    plane_loss_list = []

    for group_index in torch.unique(residue_group):
        atom_mask = (
                (residue_group == group_index)
                & base_atom_mask
        )

        if atom_mask.sum() < 4:
            continue

        base_position = prediction[atom_mask].float()
        base_position = (
                base_position
                - base_position.mean(
            dim=0,
            keepdim=True,
        )
        )

        covariance = (
                base_position.transpose(0, 1)
                @ base_position
                / base_position.size(0)
        )

        eigenvalues = torch.linalg.eigvalsh(
            covariance
        )

        plane_loss_list.append(
            eigenvalues[0].clamp_min(0)
        )

    if not plane_loss_list:
        return prediction.sum() * 0.0

    return torch.stack(plane_loss_list).mean()
