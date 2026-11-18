import torch
from etflow.models.losses.safeguards import safe_mean


class MetricsMixin:
    def record_metrics(self, stage, batch_size, batch, pos_estimate, x0_centered, x1_centered, target_mask, identity_pair_fraction, near_native_pair_fraction, loss=None, flow_matching_loss=None, bond_loss=None, clash_loss=None, plane_loss=None, geometry_regularization_loss=None, displacement_rms=None, mobility_loss=None, no_regret_loss=None, protect_loss=None, velocity_budget_loss=None, mobility_regularization_loss=None, raw_velocity_rms=None, mobility_residue=None, valid_residue_mask=None, mobility_target=None):
        if not torch.isfinite(loss):
            raise ValueError("Loss 出现 NaN，请检查数据集是否异常！")

        # 记录 Loss
        self.log_helper(f"{stage}/flow_matching_loss",flow_matching_loss,batch_size=batch_size,)
        self.log_helper(f"{stage}/bond_loss",bond_loss,batch_size=batch_size,)
        self.log_helper(f"{stage}/clash_loss",clash_loss,batch_size=batch_size,)
        self.log_helper(f"{stage}/plane_loss",plane_loss,batch_size=batch_size,)
        self.log_helper(
            f"{stage}/geometry_regularization_loss",
            geometry_regularization_loss,
            batch_size=batch_size,
        )
        self.log_helper(
            f"{stage}/displacement_rms",
            displacement_rms,
            batch_size=batch_size,
        )

        if self.use_mobility_v1:
            for metric_name, metric_value in {
                "mobility_loss": mobility_loss,
                "no_regret_loss": no_regret_loss,
                "protect_loss": protect_loss,
                "velocity_budget_loss": velocity_budget_loss,
                "mobility_regularization_loss": mobility_regularization_loss,
                "raw_velocity_rms": raw_velocity_rms,
                "mobility_mean": safe_mean(mobility_residue[valid_residue_mask]),
                "mobility_target_mean": safe_mean(mobility_target[valid_residue_mask]),
                "identity_pair_fraction": identity_pair_fraction,
                "near_native_pair_fraction": near_native_pair_fraction,
            }.items():
                self.log_helper(f"{stage}/{metric_name}",metric_value,batch_size=batch_size,)
        if stage != "train":
            graph_index = batch
            if graph_index is None:
                graph_index = torch.zeros(pos_estimate.size(0),dtype=torch.long,device=pos_estimate.device,)
            input_rmsd, input_valid = self.residue_rms_error(prediction=x0_centered,target=x1_centered,global_residue_index=graph_index,total_residues=batch_size,atom_mask=target_mask,)
            refined_rmsd, refined_valid = self.residue_rms_error(prediction=pos_estimate,target=x1_centered,global_residue_index=graph_index,total_residues=batch_size,atom_mask=target_mask,)
            valid_graph = input_valid & refined_valid

            input_rmsd = input_rmsd[valid_graph]
            refined_rmsd = refined_rmsd[valid_graph]

            for metric_name, metric_value in {
                "input_rmsd": safe_mean(input_rmsd),
                "refined_rmsd": safe_mean(refined_rmsd),
                "rmsd_improvement": safe_mean(input_rmsd - refined_rmsd),
                "worsened_fraction": safe_mean((refined_rmsd > input_rmsd).float()),
            }.items():
                self.log_helper(f"{stage}/{metric_name}",metric_value,batch_size=batch_size,)
        self.log_helper(f"{stage}/loss",loss,batch_size=batch_size,)

