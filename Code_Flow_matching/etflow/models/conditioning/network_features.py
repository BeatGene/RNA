"""Explicit x_t/x_pred conditioning of edges, node scalars, and vector initialization."""
import torch

def initialize_conditioned_features(self, z, t, pos, pos_source, edge_index, edge_attr, x_initial):
    # compute distances of the current structure
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_weight = torch.linalg.vector_norm(edge_vec,dim=-1,)

    current_rbf = self.distance_expansion(edge_weight)

    # Combine current RBF features with the original edge attributes.
    if edge_attr is not None:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        if edge_attr.size(1) != self.edge_attr_dim:
            raise ValueError(
                f"edge_attr has dim {edge_attr.size(1)}, "
                f"but edge_attr_dim={self.edge_attr_dim}"
            )
        current_edge_attr = torch.cat([current_rbf, edge_attr],dim=-1,)
    else:
        if self.edge_attr_dim != 0:
            raise ValueError(
                f"edge_attr_dim={self.edge_attr_dim}, but edge_attr is None"
            )
        current_edge_attr = current_rbf

    # Source-structure conditioning branch
    if self.source_conditioning:
        source_edge_vec = (pos_source[edge_index[0]]- pos_source[edge_index[1]])
        source_edge_dist = torch.linalg.vector_norm(source_edge_vec,dim=-1,)
        source_rbf = self.distance_expansion(source_edge_dist)

        if edge_attr is not None:
            source_edge_attr = torch.cat([source_rbf, edge_attr],dim=-1,)
        else:
            source_edge_attr = source_rbf

        source_distance_delta = source_rbf - current_rbf

        source_edge_update = self.source_edge_fusion(torch.cat([current_edge_attr, source_distance_delta],dim=-1,))

        f_ij = current_edge_attr + source_edge_update
    else:
        source_edge_dist = None
        source_edge_attr = None
        f_ij = current_edge_attr

    # Normalize the current edge vectors.
    mask = edge_index[0] == edge_index[1]
    masked_edge_weight = edge_weight.masked_fill(mask,1,).unsqueeze(1)

    if self.clip_during_norm:
        masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)

    edge_vec = edge_vec / masked_edge_weight
    # Current/source node embeddings
    if self.neighbor_embedding is not None:
        x = self.neighbor_embedding(
            z=z,
            x=x_initial,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=current_edge_attr,
        )

        if self.source_conditioning:
            source_x = self.neighbor_embedding(
                z=z,
                x=x_initial,
                edge_index=edge_index,
                edge_weight=source_edge_dist,
                edge_attr=source_edge_attr,
            )
        else:
            source_x = None
    else:
        x = x_initial
        source_x = None

    # Initialize the vector features.
    if self.source_conditioning:
        delta = pos - pos_source
        delta_norm = torch.linalg.vector_norm(delta,dim=-1,keepdim=True,)
        delta_dir = delta / delta_norm.clamp(min=1.0e-8)
        delta_scalar = torch.cat([delta_norm / self.cutoff_upper,t,],dim=-1,)
        delta_channels = self.source_delta_mlp(
            delta_scalar
        )

        x = x + delta_channels
        if source_x is not None:
            x = x + source_x
        vec = (
                delta_dir.unsqueeze(-1)
                * delta_channels.unsqueeze(1)
        )
    else:
        vec = torch.zeros(x.size(0),3,x.size(1),dtype=x.dtype,device=x.device,)

    return x, vec, source_x, edge_weight, f_ij, edge_vec
