import torch
from etflow.models.utils import center_of_mass, merge_dynamic_radius_edges


def prepare_network_inputs(self, z, t, pos, bond_index, pos_source,
                           edge_attr, node_attr, batch, atom_plddt,
                           atom_to_token_idx, token_pair_confidence, num_tokens):
    from etflow.data.validation import validate_forward
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    t = t.reshape(-1, 1)
    if t.numel() == 1:
        t = t.reshape(1, 1).expand(int(batch.max()) + 1, 1)
    if t.shape[0] != int(batch.max()) + 1 or not torch.isfinite(t).all():
        raise ValueError("t must contain one finite time per graph")
    if self.validate_inputs:
        validate_forward(z, pos, pos_source, batch, bond_index, node_attr,
                         atom_plddt, atom_to_token_idx, token_pair_confidence,
                         num_tokens, self.confidence_conditioning, self.use_mobility_v1)
    if node_attr is None:
        node_attr = pos.new_zeros((len(pos), 0))
    # Center both structures in the same translation-invariant convention.
    pos = center_of_mass(pos, batch=batch)
    # Modify_1
    pos_source = center_of_mass(pos_source, batch=batch)
    # Modify_4
    if self.confidence_conditioning:
        atom_plddt = atom_plddt.to(dtype=node_attr.dtype,device=node_attr.device,).view(-1, 1)

        node_attr = torch.cat([node_attr,self.confidence_strength * atom_plddt.clamp(0.0, 1.0),],dim=-1,)
    # Modify_2
    if self.dynamic_graph:
        edge_index, edge_type = merge_dynamic_radius_edges(
            pos=pos,
            batch=batch,
            bond_index=bond_index,
            edge_attr=edge_attr,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            num_edge_types=self.num_edge_types,
            dynamic_edge_type=self.dynamic_edge_type,
        )
    else:
        edge_index = bond_index
        edge_type = edge_attr
    if edge_type is None:
        edge_type = pos.new_zeros((edge_index.shape[1], 0))
    # Modify_4
    if self.confidence_conditioning:

        confidence_edge_attr = self.build_confidence_edge_attr(edge_index=edge_index,batch=batch,atom_to_token_idx=atom_to_token_idx,token_pair_confidence=token_pair_confidence,num_tokens=num_tokens,)
        confidence_edge_attr = confidence_edge_attr.to(dtype=edge_type.dtype,device=edge_type.device,)

        edge_type = torch.cat([edge_type,self.confidence_strength * confidence_edge_attr,],dim=-1,)
    network_kwargs = {
        "z": z.view(-1),
        "t": t[batch],
        "pos": pos,
        "pos_source": pos_source,
        "edge_index": edge_index,
        "edge_attr": edge_type,
        "node_attr": node_attr,
        "batch": batch,
    }
    return network_kwargs
