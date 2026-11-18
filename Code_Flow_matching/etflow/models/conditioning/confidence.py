from typing import Optional
import torch
from torch import Tensor


class ConfidenceMixin:
    def build_confidence_edge_attr(self,edge_index: Tensor,batch: Optional[Tensor],atom_to_token_idx: Tensor,token_pair_confidence: Tensor,num_tokens: Tensor,) -> Tensor:
        atom_to_token_idx = atom_to_token_idx.long().view(-1)
        num_tokens = num_tokens.long().view(-1)

        if batch is None:
            batch = torch.zeros(atom_to_token_idx.numel(),dtype=torch.long,device=atom_to_token_idx.device,)

        pair_count = num_tokens.square()

        pair_offset = torch.cat([pair_count.new_zeros(1),pair_count.cumsum(dim=0)[:-1],],dim=0,)

        edge_source = edge_index[0]
        edge_target = edge_index[1]

        edge_batch = batch[edge_source]
        token_source = atom_to_token_idx[edge_source]
        token_target = atom_to_token_idx[edge_target]

        edge_num_tokens = num_tokens[edge_batch]

        flat_pair_index = (pair_offset[edge_batch]+ token_source * edge_num_tokens+ token_target)

        return token_pair_confidence[flat_pair_index]

