"""RNA adapter for ABRefine's spherical EquiformerV2 representation.

Scalar l=0, Cartesian l=1, and higher spherical latents remain separate.
The same current/source conditioning contract is used as in TorchMD.
"""
import torch
from torch import nn
from .abrefine_core import ABRefineCore
from .abrefine_ops.so3 import SO3_Embedding
from .abrefine_ops.edge_rot_mat import init_edge_rot_mat


class ABRefineRepresentation(nn.Module):
    def __init__(self, hidden_channels, num_layers, num_rbf, num_heads,
                 cutoff_upper, max_z, node_attr_dim, edge_attr_dim,
                 source_conditioning=True, **options):
        super().__init__()
        self.source_conditioning = source_conditioning
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff_upper
        # One resolution is required by the RNA adapter. l>=1 yields vector output.
        if len(options.get("lmax_list", [2])) != 1 or options.get("lmax_list", [2])[0] < 1:
            raise ValueError("ABRefine requires one spherical resolution with lmax >= 1")
        allowed = {'attn_hidden_channels', 'grid_resolution', 'ffn_hidden_channels', 'attn_alpha_channels', 'use_grid_mlp', 'use_atom_edge_embedding', 'attn_activation', 'use_m_share_rad', 'use_attn_renorm', 'drop_path_rate', 'use_s2_act_attn', 'ffn_activation', 'share_atom_edge_embedding', 'lmax_list', 'proj_drop', 'edge_channels', 'alpha_drop', 'use_gate_act', 'use_sep_s2_act', 'mmax_list', 'norm_type', 'attn_value_channels', 'weight_init'}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown ABRefine options: {sorted(unknown)}")
        lmax = options.get("lmax_list", [2])[0]
        mmax = options.get("mmax_list", [2])
        if len(mmax) != 1 or not 0 <= mmax[0] <= lmax:
            raise ValueError("require one mmax with 0 <= mmax <= lmax")
        self.core = ABRefineCore(
            sphere_channels=hidden_channels, num_layers=num_layers, num_heads=num_heads,
            num_distance_basis=num_rbf, max_radius=cutoff_upper, max_num_elements=max_z,
            **options)
        self.node_projection = nn.Linear(node_attr_dim, hidden_channels) if node_attr_dim else None
        self.time_projection = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(),
                                             nn.Linear(hidden_channels, hidden_channels))
        # Disabled conditioning must not leave trainable parameters unused by DDP.
        self.delta_projection = (
            nn.Sequential(nn.Linear(2, hidden_channels), nn.SiLU(),
                          nn.Linear(hidden_channels, hidden_channels))
            if source_conditioning else None
        )
        # Current/source distances and edge attributes are fused before every block.
        edge_input = num_rbf * (2 if source_conditioning else 1) + edge_attr_dim
        self.edge_projection = nn.Sequential(nn.Linear(edge_input, num_rbf), nn.SiLU(),
                                             nn.Linear(num_rbf, num_rbf))
        # l=1 e3nn ordering is Cartesian (x,y,z), verified by equivariance tests.

    def forward(self, z, t, pos, pos_source, batch, edge_index, node_attr=None, edge_attr=None):
        core = self.core
        src, dst = edge_index
        # Self/coincident edges have undefined local frames. Exclude them here.
        relative = pos[src] - pos[dst]
        distance = relative.norm(dim=-1)
        keep = distance > 1e-6
        edge_index, relative, distance = edge_index[:, keep], relative[keep], distance[keep]
        if edge_index.shape[1] == 0:
            raise ValueError("Equiformer needs at least one noncoincident edge")
        current = core.distance_expansion(distance)
        features = [current]
        if self.source_conditioning:
            source_distance = (pos_source[edge_index[0]] - pos_source[edge_index[1]]).norm(dim=-1)
            features.append(core.distance_expansion(source_distance) - current)
        if edge_attr is not None:
            features.append(edge_attr[keep])
        edges = self.edge_projection(torch.cat(features, dim=-1))
        if core.share_atom_edge_embedding and core.use_atom_edge_embedding:
            edges = torch.cat([edges, core.source_embedding(z[edge_index[0]]),
                               core.target_embedding(z[edge_index[1]])], dim=-1)
        rotation = init_edge_rot_mat(relative)
        for rot in core.SO3_rotation:
            rot.set_wigner(rotation)
        x = SO3_Embedding(len(z), core.lmax_list, self.hidden_channels, pos.device, pos.dtype)
        scalar = core.sphere_embedding(z.view(-1)) + self.time_projection(t.reshape(-1, 1))
        if self.node_projection is not None:
            scalar = scalar + self.node_projection(node_attr)
        if self.source_conditioning:
            delta = pos - pos_source
            norm = delta.norm(dim=-1, keepdim=True)
            channels = self.delta_projection(torch.cat([norm / self.cutoff, t.reshape(-1, 1)], dim=-1))
            scalar = scalar + channels
            x.embedding[:, 1:4, :] = delta.unsqueeze(-1) * channels.unsqueeze(1) / norm.clamp_min(1e-8).unsqueeze(-1)
        x.embedding[:, 0, :] = scalar
        x.embedding = x.embedding + core.edge_degree_embedding(z, edges, edge_index).embedding
        for block in core.blocks:
            x = block(x, z, edges, edge_index, batch=batch)
        x.embedding = core.norm(x.embedding)
        return x.embedding[:, 0], x.embedding[:, 1:4], z, pos, batch, (x, edges, edge_index)
