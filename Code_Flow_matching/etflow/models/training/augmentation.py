import torch
from torch import Tensor


class SourceAugmentationMixin:
    def augment_residual_source(self,x0_centered:Tensor,x1_centered:Tensor,batch:Tensor, target_mask=None):
        num_graphs = int(batch.max().item()) + 1
        random_value = torch.rand(num_graphs,device=x0_centered.device,)

        identity_graph_mask = (
                random_value < self.identity_pair_probability
        )

        near_native_graph_mask = (
            (random_value >= self.identity_pair_probability)
            & (
                random_value
                < self.identity_pair_probability
                + self.near_native_pair_probability
            )
        )

        alpha = torch.empty(
            num_graphs,
            1,
            dtype=x0_centered.dtype,
            device=x0_centered.device,
        ).uniform_(
            self.near_native_min_alpha,
            self.near_native_max_alpha,
        )

        if target_mask is not None:
            x1_centered = torch.where(target_mask.view(-1, 1), x1_centered, x0_centered)
        near_native_source = (
            x1_centered
            + alpha[batch] * (x0_centered - x1_centered)
        )

        identity_atom_mask = identity_graph_mask[batch].unsqueeze(-1)
        near_native_atom_mask = near_native_graph_mask[batch].unsqueeze(-1)

        augmented_source = torch.where(
            identity_atom_mask,
            x1_centered,
            x0_centered,
        )

        augmented_source = torch.where(
            near_native_atom_mask,
            near_native_source,
            augmented_source,
        )

        return (
            augmented_source,
            identity_graph_mask.float().mean(),
            near_native_graph_mask.float().mean(),
        )

