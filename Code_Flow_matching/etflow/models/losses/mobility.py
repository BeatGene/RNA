import torch
import torch.nn.functional as F
from .safeguards import no_regret, protect, safe_mean


def mobility_supervision(prediction, target, mask):
    return safe_mean(F.smooth_l1_loss(prediction[mask], target[mask], reduction="none"))


def mobility_terms(self, x0_centered, x1_centered, pos_estimate, target_mask, mobility_aux):
    global_residue_index = mobility_aux["global_residue_index"]
    total_residues = mobility_aux["total_residues"]

    raw_residue_error, valid_residue_mask = self.residue_rms_error(prediction=x0_centered,target=x1_centered,global_residue_index=global_residue_index,total_residues=total_residues,atom_mask=target_mask,)
    refined_residue_error, refined_valid_residue_mask = self.residue_rms_error(prediction=pos_estimate,target=x1_centered,global_residue_index=global_residue_index,total_residues=total_residues,atom_mask=target_mask,)
    valid_residue_mask = valid_residue_mask & refined_valid_residue_mask

    mobility_target =((raw_residue_error-self.mobility_good_error)/(self.mobility_move_error-self.mobility_good_error)).clamp(0.0,1.0)
    mobility_residue = mobility_aux["mobility_residue"].view(-1)
    mobility_loss = mobility_supervision(mobility_residue, mobility_target, valid_residue_mask)

    no_regret_loss = no_regret(refined_residue_error, raw_residue_error, valid_residue_mask)

    correct_residue_mask = ((raw_residue_error<self.protect_error_threshold)&valid_residue_mask)

    protect_loss = protect(pos_estimate, x0_centered,
                           correct_residue_mask[global_residue_index] & target_mask)

    raw_velocity_square = mobility_aux["raw_velocity"].square().sum(dim=-1, keepdim=True)
    velocity_budget_loss = ((1.0 - mobility_aux["mobility_atom"].detach())* raw_velocity_square).mean()

    raw_velocity_rms = (
        mobility_aux["raw_velocity"].square().sum(dim=-1).mean().sqrt()
    )

    return {'mobility_loss': mobility_loss, 'no_regret_loss': no_regret_loss, 'protect_loss': protect_loss, 'velocity_budget_loss': velocity_budget_loss, 'raw_velocity_rms': raw_velocity_rms, 'mobility_residue': mobility_residue, 'valid_residue_mask': valid_residue_mask, 'mobility_target': mobility_target}
