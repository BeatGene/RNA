from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils import scatter
from etflow.commons.reductions import grouped_mean
from etflow.models.losses.bond import atomwise_bond_error
from etflow.models.losses.clash import atomwise_steric_clash_score
from etflow.models.utils import center_of_mass


class MobilityMixin:
    def build_global_residue_index(self,batch: Tensor,atom_to_token_idx: Tensor,num_tokens: Tensor,):
        """Convert per-graph token IDs into batch-global residue IDs."""
        if batch is None:
            batch = torch.zeros(atom_to_token_idx.numel(),dtype=torch.long,device=atom_to_token_idx.device,)

        atom_to_token_idx = atom_to_token_idx.long().view(-1)
        num_tokens = num_tokens.long().view(-1)

        token_offset = torch.cat([num_tokens.new_zeros(1),num_tokens.cumsum(dim=0)[:-1],],dim=0,)
        global_residue_index = (token_offset[batch] + atom_to_token_idx)
        total_residues = int(num_tokens.sum().item())

        return global_residue_index, total_residues

    @staticmethod
    def residue_rms_error(prediction: Tensor,target: Tensor,global_residue_index: Tensor,total_residues: int,atom_mask: Optional[Tensor] = None,) -> tuple[Tensor, Tensor]:
        atom_square_error = (prediction - target).square().sum(dim=-1)

        if atom_mask is None:
            atom_mask = torch.ones_like(atom_square_error, dtype=torch.bool)

        atom_mask = atom_mask.to(atom_square_error.device).bool().view(-1)

        weights = atom_mask.to(atom_square_error.dtype)

        #按照残基汇总平方误差
        square_error_sum = scatter(atom_square_error * weights,global_residue_index,dim=0,dim_size=total_residues,reduce="sum",)

        #按照残基汇总有效原子数目
        observed_count = scatter(weights,global_residue_index,dim=0,dim_size=total_residues,reduce="sum",)

        valid_residue = observed_count > 0

        mean_square_error = square_error_sum / observed_count.clamp_min(1.0)
        positive_error = valid_residue & (mean_square_error > 0)

        safe_mean_square_error = torch.where(positive_error, mean_square_error, torch.ones_like(mean_square_error))

        rms_error = torch.where(positive_error,torch.sqrt(safe_mean_square_error),torch.zeros_like(mean_square_error),)

        return rms_error, valid_residue

    def apply_residue_mobility_gate(self,
            raw_velocity: Tensor,
            hidden: Tensor,
            pos_source: Tensor,
            z: Tensor,
            batch: Tensor,
            atom_plddt: Tensor,
            atom_mobility_attr: Tensor,
            atom_to_token_idx: Tensor,
            num_tokens: Tensor,
            geometry_bond_index: Tensor,
            ideal_bond_length: Tensor,
            clash_exclusion_index: Tensor,
    ):
        if not self.confidence_conditioning:
            atom_plddt = hidden.new_zeros(hidden.size(0))
            atom_mobility_attr = hidden.new_zeros(hidden.size(0), 4)
        elif atom_mobility_attr is None:
            raise ValueError("mobility with confidence requires atom_mobility_attr [N,4]")
        atom_plddt = atom_plddt.to(dtype=hidden.dtype,device=hidden.device,).view(-1, 1)
        atom_mobility_attr = atom_mobility_attr.to(dtype=hidden.dtype,device=hidden.device,)

        source_bond_error = atomwise_bond_error(prediction=pos_source,geometry_bond_index=geometry_bond_index,ideal_bond_length=ideal_bond_length,).to(dtype=hidden.dtype, device=hidden.device)

        source_clash_score = atomwise_steric_clash_score(prediction=pos_source,atomic_numbers=z,batch=batch,geometry_bond_index=geometry_bond_index,vdw_radius_table=self.vdw_radius_table,clash_exclusion_index=clash_exclusion_index,).to(dtype=hidden.dtype, device=hidden.device)

        atom_gate_input = torch.cat([hidden,self.confidence_strength * atom_plddt.clamp(0.0, 1.0),self.confidence_strength * atom_mobility_attr,source_bond_error,source_clash_score,],dim=-1,)

        global_residue_index, total_residues = self.build_global_residue_index(batch=batch,atom_to_token_idx=atom_to_token_idx,num_tokens=num_tokens,)

        residue_gate_input = grouped_mean(
            atom_gate_input,
            global_residue_index,
            dim=0,
            dim_size=total_residues,
        )
        mobility_residue = torch.sigmoid(
            self.mobility_head(residue_gate_input)
        )
        mobility_atom = mobility_residue[global_residue_index]
        gated_velocity = center_of_mass(
            mobility_atom * raw_velocity,
            batch=batch,
        )
        return gated_velocity, {
            "raw_velocity": raw_velocity,
            "mobility_residue": mobility_residue,
            "mobility_atom": mobility_atom,
            "global_residue_index": global_residue_index,
            "total_residues": total_residues,
        }

