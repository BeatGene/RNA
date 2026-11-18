import torch

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

        # .float() on positions alone is insufficient: autocast can cast
        # the covariance matmul back to BF16. Keep both its accumulation
        # and eigvalsh in FP32, with autograd still enabled.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            base_position = prediction[atom_mask].float()
            base_position = base_position - base_position.mean(dim=0, keepdim=True)
            covariance = (
                base_position.transpose(0, 1)
                @ base_position
                / base_position.size(0)
            )
            eigenvalues = torch.linalg.eigvalsh(covariance)

        plane_loss_list.append(
            eigenvalues[0].clamp_min(0)
        )

    if not plane_loss_list:
        return prediction.sum() * 0.0

    return torch.stack(plane_loss_list).mean()


