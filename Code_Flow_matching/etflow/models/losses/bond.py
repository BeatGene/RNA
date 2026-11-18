import torch
from etflow.commons.reductions import grouped_mean, reduction_dtype

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
    return grouped_mean(
        atom_error,
        atom_index,
        dim=0,
        dim_size=num_atoms,
    ).unsqueeze(-1)


