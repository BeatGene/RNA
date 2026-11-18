from typing import Optional
import torch
from torch_geometric.utils import scatter
from etflow.models.utils import build_dynamic_radius_graph

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


