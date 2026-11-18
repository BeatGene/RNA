"""RNAFM + atom role + RNA residue identity -> atom scalar attributes."""
import torch
import torch.nn.functional as F
from .constants import NUM_ATOM_NAME_TYPES, NUM_RNA_RESIDUE_TYPES


def node_features(data, residue_index, atom_name_id):
    rnafm_embedding = data["rnafm_embedding"].float().contiguous()
    residue_type_id = data["residue_type_id"].long().view(-1)
    rnafm_atom = rnafm_embedding[residue_index]

    atom_role_one_hot = F.one_hot(atom_name_id,num_classes=NUM_ATOM_NAME_TYPES,).float()
    residue_type_atom = residue_type_id[residue_index]
    residue_type_one_hot = F.one_hot(residue_type_atom,num_classes=NUM_RNA_RESIDUE_TYPES,).float()

    node_attr = torch.cat([rnafm_atom,atom_role_one_hot,residue_type_one_hot,],dim=-1,).contiguous()

    return node_attr
