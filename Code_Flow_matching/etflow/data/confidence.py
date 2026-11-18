"""Raw confidence -> normalized node, flattened pair and residue summary features.

All-or-none raw fields avoid silently filling partially corrupt confidence records.
atom_to_token_idx is a local mapping, never a numeric confidence feature.
"""
import torch
from .validation import finite


def confidence_features(data, residue_index, mode="auto"):
    if mode not in {"auto", "required", "off"}:
        raise ValueError("confidence_mode must be auto, required or off")
    keys = ("atom_plddt", "token_pair_pae", "token_pair_pde", "contact_probs")
    present = [k in data for k in keys]
    use = mode != "off" and all(present)
    if mode == "required" and not use:
        raise ValueError("confidence_mode=required needs all confidence fields")
    if mode == "auto" and any(present) and not all(present):
        raise ValueError("Partial confidence record; provide all fields or use confidence_mode=off")
    idx = data.get("atom_to_token_idx", residue_index).long().view(-1)
    n = residue_index.numel()
    if use:
        plddt = data["atom_plddt"].float().view(-1)
        pae, pde, contact = [data[k].float() for k in keys[1:]]
        for name, value in zip(keys, (plddt, pae, pde, contact)):
            finite(name, value)
        t = pae.shape[0]
        if pae.shape != (t, t) or pde.shape != (t, t) or contact.shape != (t, t):
            raise ValueError("PAE/PDE/contact must have matching square [T,T] shapes")
        if plddt.numel() != n or (plddt < 0).any() or (plddt > 1).any():
            raise ValueError("atom_plddt must contain N values in [0,1]")
        pairs = torch.stack([pae.clamp(0,32)/32, pae.T.clamp(0,32)/32,
                             pde.clamp(0,32)/32, contact.clamp(0,1)], -1)
    else:
        t = int(idx.max()) + 1
        plddt, pairs = torch.zeros(n), torch.zeros(t, t, 4)
    if idx.numel() != n or (idx < 0).any() or (idx >= t).any():
        raise ValueError("atom_to_token_idx must contain N local token indices")
    return dict(atom_plddt=plddt, atom_to_token_idx=idx,                # [N]  [N]  [t*t,4]    [1]   [N,4]
                token_pair_confidence=pairs.reshape(-1,4).contiguous(),
                num_tokens=torch.tensor([t], dtype=torch.long),
                atom_mobility_attr=pairs.mean(1)[idx].contiguous())
