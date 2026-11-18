from typing import Optional
import torch
from torch_geometric.utils import scatter
from etflow.commons.reductions import grouped_mean, reduction_dtype

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

    dtype = torch.promote_types(reduction_dtype(prediction), reduction_dtype(target))
    atom_loss = ((prediction.to(dtype) - target.to(dtype)) ** 2).sum(dim=-1)
    pooled = grouped_mean(atom_loss, batch) if reduce == "mean" else scatter(atom_loss, index=batch, reduce=reduce)
    return pooled.mean(dim=0)


def batchwise_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    dtype = torch.promote_types(reduction_dtype(prediction), reduction_dtype(target))
    atom_loss = torch.norm(prediction.to(dtype) - target.to(dtype), p=2, dim=-1)
    if mask is None:
        pooled = grouped_mean(atom_loss, batch) if reduce == "mean" else scatter(atom_loss, index=batch, reduce=reduce)
        return pooled.mean(dim=0)

    mask = mask.to(device=prediction.device).bool().view(-1)
    if mask.numel() != prediction.size(0):
        raise ValueError("mask must contain one value per atom")

    # Normalize each graph by its number of observed native atoms, so long RNAs
    # and structures with more resolved atoms do not dominate a batch.
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    weights = mask.to(dtype=atom_loss.dtype)
    loss_sum = scatter(
        atom_loss * weights,
        index=batch,
        dim=0,
        dim_size=num_graphs,
        reduce="sum",
    )
    observed_count = torch.bincount(batch[mask], minlength=num_graphs)
    valid_graph = observed_count > 0
    if not valid_graph.any():
        raise ValueError("batch has no native-observed atoms")
    return (loss_sum[valid_graph] / observed_count[valid_graph]).mean()



def flow_matching_loss(prediction, target, batch=None, mask=None):
    """Mandatory base loss: masked atom-vector L2, averaged equally across valid graphs."""
    return batchwise_l2_loss(prediction, target, batch=batch, reduce="mean", mask=mask)
