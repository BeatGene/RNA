from typing import Tuple

import torch
from torch_cluster import radius_graph
from torch.nn.functional import pad
from torch_geometric.utils import get_laplacian, scatter, to_dense_adj

# from etflow.commons.utils import extend_graph_order_radius


def center_pos(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


def linear_schedule(low, high, max_steps, total_steps) -> torch.Tensor:
    schedule = torch.linspace(low, high, steps=max_steps)

    if max_steps < total_steps:
        pad_size = abs(total_steps - max_steps)
        schedule = pad(schedule, pad=(0, pad_size), mode="constant", value=high)

    return schedule


#Modify_2
@torch.no_grad()
def build_dynamic_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int = 32,
) -> torch.Tensor:
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, but got {cutoff}")
    if max_num_neighbors <= 0:
        raise ValueError(
            "max_num_neighbors must be positive, "
            f"but got {max_num_neighbors}"
        )

    if batch is None:
        batch = torch.zeros(
            pos.size(0),
            dtype=torch.long,
            device=pos.device,
        )

    edge_index = radius_graph(
        x=pos,
        r=cutoff,
        batch=batch,
        loop=False,
        max_num_neighbors=max_num_neighbors,
        flow="source_to_target",
    )

    # radius_graph may become asymmetric when the neighbor cap is reached.
    # Explicitly add reverse edges and remove duplicates.
    edge_index = torch.cat(
        [edge_index, edge_index.flip(0)],
        dim=1,
    )

    num_nodes = pos.size(0)
    edge_id = edge_index[0] * num_nodes + edge_index[1]
    edge_id = torch.unique(edge_id)

    return torch.stack(
        [
            torch.div(edge_id, num_nodes, rounding_mode="floor"),
            edge_id % num_nodes,
        ],
        dim=0,
    )

#Modify_2
@torch.no_grad()
def merge_dynamic_radius_edges(
    pos: torch.Tensor,
    batch: torch.Tensor,
    bond_index: torch.Tensor,
    edge_attr: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int,
    num_edge_types: int,
    dynamic_edge_type: int = 3,
):
    if edge_attr.dim() != 2:
        raise ValueError(
            "edge_attr must have shape [num_edges, num_edge_types], "
            f"but got {tuple(edge_attr.shape)}"
        )
    if edge_attr.size(0) != bond_index.size(1):
        raise ValueError(
            f"edge_attr has {edge_attr.size(0)} rows, but bond_index "
            f"contains {bond_index.size(1)} edges"
        )
    if edge_attr.size(1) != num_edge_types:
        raise ValueError(
            f"edge_attr has dim {edge_attr.size(1)}, but "
            f"num_edge_types={num_edge_types}"
        )
    if not 0 <= dynamic_edge_type < num_edge_types:
        raise ValueError(
            f"dynamic_edge_type={dynamic_edge_type} must be in "
            f"[0, {num_edge_types})"
        )

    dynamic_edge_index = build_dynamic_radius_graph(
        pos=pos,
        batch=batch,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
    )

    num_nodes = pos.size(0)

    # 删除与永久静态边重复的 dynamic edges。
    static_edge_id = bond_index[0] * num_nodes + bond_index[1]
    dynamic_edge_id = (
        dynamic_edge_index[0] * num_nodes
        + dynamic_edge_index[1]
    )

    keep = ~torch.isin(
        dynamic_edge_id,
        static_edge_id,
    )
    dynamic_edge_index = dynamic_edge_index[:, keep]

    dynamic_edge_attr = torch.zeros(
        dynamic_edge_index.size(1),
        num_edge_types,
        dtype=edge_attr.dtype,
        device=edge_attr.device,
    )
    dynamic_edge_attr[:, dynamic_edge_type] = 1

    edge_index = torch.cat(
        [bond_index, dynamic_edge_index],
        dim=1,
    )
    edge_type = torch.cat(
        [edge_attr, dynamic_edge_attr],
        dim=0,
    )

    return edge_index, edge_type


def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com


def assert_zero_mean(x: torch.Tensor, batch: torch.Tensor, eps=1e-10) -> bool:
    largest_value = x.abs().max().item()
    a = scatter(x, batch, dim=0, reduce="mean") if batch is not None else x.mean(dim=0)
    error = a.abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def extend_bond_index(
    pos: torch.Tensor,
    bond_index: torch.Tensor,
    batch: torch.Tensor,
    bond_attr: torch.Tensor,
    device: torch.device,
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bond_attr is None:
        bond_type = torch.ones(bond_index.shape[1], dtype=torch.long, device=device)
        # all molecular graph edges are type 1, radius based become 0
    else:
        bond_type = bond_attr.view(-1).long() + 1  # we reserve 0 for radius based edges
        assert (
            bond_type.shape[0] == bond_index.shape[1]
        ), "Edge type should have same shape as number of edges."

    edge_index, edge_type = extend_graph_order_radius(
        pos=pos,
        edge_index=bond_index,
        edge_type=bond_type,
        batch=batch,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        extend_radius=True,
    )
    assert (
        bond_index.shape[1] == (edge_type > 0).sum().item()
    ), "Edge Type should be greater than 0 when edge is a molecular bond."

    # make one_hot if provided
    if one_hot:
        # +1 to account for radius based edges
        edge_type = torch.nn.functional.one_hot(
            edge_type, num_classes=one_hot_types + 1
        ).float()

    return edge_index, edge_type


def unsqueeze_like(x: torch.Tensor, target: torch.Tensor):
    shape = (x.size(0), *([1] * (target.dim() - 1)))
    return x.view(shape)


"""
Following code adapted from HarmonicFlow
https://github.com/HannesStark/FlowSite/blob/main/utils/diffusion.py
"""


class HarmonicSampler:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.eig_val_cache = {}
        self.eig_vec_cache = {}

    def diagonalize(self, n_nodes, edges=[], batch=None, smiles=None):
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        edge_index, edge_weight = get_laplacian(
            edges.T,
            a,
            num_nodes=n_nodes,
        )

        H = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            D, P = torch.linalg.eigh(H)
            return D, P

        Ds, Ps = [], []
        batch_size = batch.max() + 1

        for i in range(batch_size):
            idx = torch.where(batch == i)[0]
            start = idx.min()
            end = idx.max() + 1

            D, P = None, None
            if smiles is not None:
                D, P = self.check_cache(smiles[i])

                if (D is not None) and (P is not None):
                    D = D.to(edge_index.device)
                    P = P.to(edge_index.device)

            if (D is None) or (P is None):
                D, P = torch.linalg.eigh(H[start:end, start:end])

                if smiles is not None:
                    self.eig_val_cache[smiles[i]] = D.cpu()
                    self.eig_vec_cache[smiles[i]] = P.cpu()

            Ds.append(D)
            Ps.append(P)

        return torch.cat(Ds), torch.block_diag(*Ps)

    def check_cache(self, smiles):
        D = self.eig_val_cache.get(smiles, None)
        P = self.eig_vec_cache.get(smiles, None)
        return D, P

    def sample(self, size, edge_index, batch=None, smiles=None):
        # transpose if (2, n_edges)
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]
        D, P = self.diagonalize(
            n_nodes=n_nodes, edges=edge_index, batch=batch, smiles=smiles
        )

        # get starting index per sample in batch
        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        std = 1.0 / torch.sqrt(D)
        std[start_index] = 0.0

        noise = torch.randn(size).to(D.device)
        noise = std[:, None] * noise
        noise[noise.isnan()] = 0.0
        sample = P @ (noise)

        return sample

    def energy(self, x, edge_index, batch=None, smiles=None):
        n_nodes = x.size(0)
        x = center_of_mass(x)

        if batch is None:
            batch = torch.zeros(n_nodes).to(x.device).long()

        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        D, P = self.diagonalize(n_nodes, edges=edge_index, batch=batch, smiles=smiles)

        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        energy_unpooled = D[:, None] * (P.T @ x) ** 2
        energy_unpooled[start_index] = 0.0
        energy_unpooled = energy_unpooled.sum(-1)
        energy = 0.5 * scatter(energy_unpooled, batch)

        return energy.view(-1, 1)


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor(
            [[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float
        )
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Ensure R is a proper rotation matrix
    if torch.det(R) < 0:  # reflection
        V[:, -1] *= -1  # flip the sign of the last column of V
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def rmsd_align(pos, ref_pos, batch):
    aligned_pos = []
    batch_size = batch.max() + 1
    for i in range(batch_size):
        index = torch.where(batch == i)[0]
        pos_i = pos[index]
        ref_pos_i = ref_pos[index]
        R, t = find_rigid_alignment(pos_i, ref_pos_i)

        pos_i = (R @ pos_i.T).T + t
        aligned_pos.append(pos_i)

    return torch.concat(aligned_pos, dim=0)
