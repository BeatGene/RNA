"""Always-on flow matching plus optional geometry and residue penalties."""
from .regression import flow_matching_loss
from .bond import bond_length_loss
from .clash import steric_clash_loss
from .plane import base_plane_loss
from .mobility import mobility_terms
from .safeguards import safe_mean


class LossMixin:
    def compute_losses(self, v_t, u_t, batch, target_mask, pos_estimate,
                       geometry_bond_index, ideal_bond_length, z,
                       clash_exclusion_index, residue_index, atom_name_id,
                       x0_centered, x1_centered, mobility_aux):
        fm = flow_matching_loss(v_t, u_t, batch=batch, mask=target_mask)
        metrics = {"flow_matching_loss": fm}
        zero = fm * 0
        geometry = zero
        terms = {
            "bond_loss": (self.bond_loss_weight, lambda: bond_length_loss(
                pos_estimate, geometry_bond_index, ideal_bond_length)),
            "clash_loss": (self.clash_loss_weight, lambda: steric_clash_loss(
                prediction=pos_estimate, atomic_numbers=z, batch=batch,
                geometry_bond_index=geometry_bond_index, vdw_radius_table=self.vdw_radius_table,
                clash_exclusion_index=clash_exclusion_index)),
            "plane_loss": (self.plane_loss_weight, lambda: base_plane_loss(
                prediction=pos_estimate, residue_index=residue_index, atom_name_id=atom_name_id,
                batch=batch, base_atom_name_ids=self.base_atom_name_ids))}
        for name, (weight, compute) in terms.items():
            metrics[name] = compute() if weight else zero
            geometry = geometry + weight * metrics[name]
        metrics["geometry_regularization_loss"] = geometry
        metrics["displacement_rms"] = (pos_estimate - x0_centered).square().sum(-1).mean().sqrt()
        total = fm + geometry
        if self.use_mobility_v1:
            metrics.update(mobility_terms(self, x0_centered, x1_centered, pos_estimate,
                                          target_mask, mobility_aux))
            regularization = (self.mobility_gate_loss_weight * metrics["mobility_loss"]
                              + self.no_regret_loss_weight * metrics["no_regret_loss"]
                              + self.protect_loss_weight * metrics["protect_loss"]
                              + self.velocity_budget_loss_weight * metrics["velocity_budget_loss"])
            metrics["mobility_regularization_loss"] = regularization
            total = total + regularization
        metrics["loss"] = total
        return metrics
