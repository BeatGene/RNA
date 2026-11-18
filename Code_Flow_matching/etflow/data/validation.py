"""Reusable checks for raw records and batched model inputs."""
import torch


def finite(name, tensor):
    if tensor is None or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be present and finite")


def validate_edges(name, edges, n, batch=None):
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError(f"{name} must be int64 [2,E]")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError(f"{name} contains out-of-range atom indices")
    if batch is not None and edges.numel() and (batch[edges[0]] != batch[edges[1]]).any():
        raise ValueError(f"{name} contains cross-graph edges")


def validate_forward(z, pos, source, batch, edges, node_attr, plddt, token_idx,
                     pairs, num_tokens, confidence, mobility):
    n = len(pos)
    if n == 0 or pos.shape != (n, 3) or source.shape != pos.shape:
        raise ValueError("pos and pos_source must be nonempty [N,3]")
    finite("pos", pos); finite("pos_source", source)
    if z.numel() != n or batch.shape != (n,) or batch.dtype != torch.long:
        raise ValueError("z and int64 batch must contain N entries")
    if batch.min() != 0 or not torch.equal(torch.unique(batch), torch.arange(int(batch.max()) + 1, device=batch.device)):
        raise ValueError("batch IDs must be contiguous from zero")
    validate_edges("bond_index", edges, n, batch)
    if node_attr is not None:
        finite("node_attr", node_attr)
        if node_attr.ndim != 2 or node_attr.shape[0] != n:
            raise ValueError("node_attr must be [N,F]")
    if confidence or mobility:
        if token_idx is None or num_tokens is None:
            raise ValueError("confidence/mobility requires atom_to_token_idx and num_tokens")
        if token_idx.dtype != torch.long or num_tokens.dtype != torch.long:
            raise ValueError("token indices/counts must be int64")
        if token_idx.numel() != n or num_tokens.numel() != int(batch.max()) + 1:
            raise ValueError("token index/count shape mismatch")
        if (num_tokens <= 0).any() or (token_idx < 0).any() or (token_idx.view(-1) >= num_tokens.view(-1)[batch]).any():
            raise ValueError("atom_to_token_idx outside local token range")
    if confidence:
        finite("atom_plddt", plddt); finite("token_pair_confidence", pairs)
        if plddt.numel() != n or (plddt < 0).any() or (plddt > 1).any():
            raise ValueError("atom_plddt must be [N], normalized to [0,1]")
        if pairs.shape != (int(num_tokens.square().sum()), 4):
            raise ValueError("token_pair_confidence must be concatenated [sum(T_g^2),4]")


def validate_record(data):
    """Checks common to training/evaluation before featurization."""
    pos, pred = data["pos"], data["pos_pred"]
    finite("pos", pos); finite("pos_pred", pred)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape != pred.shape or len(pos) == 0:
        raise ValueError("pos/pos_pred must have matching nonempty [N,3] shapes")
    n = len(pos)
    if data["target_mask"].numel() != n or not data["target_mask"].bool().any():
        raise ValueError("target_mask must have N entries and at least one observed atom")
    for key in ("edge_index", "geometry_bond_index"):
        validate_edges(key, data[key], n)
    if data["edge_attr"].shape[0] != data["edge_index"].shape[1]:
        raise ValueError("edge_attr length does not match edge_index")
    for key in ("atomic_numbers", "residue_index", "atom_name_id"):
        if data[key].numel() != n:
            raise ValueError(f"{key} must have N entries")
    finite("ideal_bond_length", data["ideal_bond_length"])
    if data["ideal_bond_length"].numel() != data["geometry_bond_index"].shape[1]:
        raise ValueError("ideal_bond_length must match geometry_bond_index")
