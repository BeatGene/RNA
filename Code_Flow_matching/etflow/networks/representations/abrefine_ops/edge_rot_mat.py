"""Deterministic local frames, adapted from ABRefine's random gauge construction."""
import torch
from etflow.commons.reductions import reduction_dtype


def init_edge_rot_mat(edge_distance_vec):
    # Geometry stays at least FP32 even inside a BF16/FP16 autocast region.
    with torch.autocast(device_type=edge_distance_vec.device.type, enabled=False):
        vectors = edge_distance_vec.to(reduction_dtype(edge_distance_vec))
        norm_x = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)
        # Least-aligned axis guarantees a stable cross product and consumes no RNG.
        axis = torch.nn.functional.one_hot(norm_x.abs().argmin(-1), 3).to(norm_x)
        norm_z = torch.nn.functional.normalize(torch.cross(norm_x, axis, dim=-1), dim=-1)
        norm_y = torch.cross(norm_x, norm_z, dim=-1)
        return torch.stack([norm_z, norm_x, -norm_y], dim=1).detach()
