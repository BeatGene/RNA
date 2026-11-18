import torch
from etflow.models.utils import center_of_mass, unsqueeze_like


class TrainingMixin:
    def generic_step(self, batched_data, batch_idx: int, stage: str):
        """
        核心训练步。
        """
        # 从 Dataloader 中获取数据
        z = batched_data["atomic_numbers"]
        pos = batched_data["pos"]  # X_1: 真实目标坐标
        pos_pred = batched_data["pos_pred"]  # X_0: 流的起点（Protenix预测坐标）
        target_mask = batched_data["target_mask"].bool().view(-1)
        bond_index = batched_data["edge_index"]  # 边
        node_attr = batched_data.get("node_attr", None)
        edge_attr = batched_data.get("edge_attr", None)
        batch = batched_data.get("batch", None)
        geometry_bond_index = batched_data["geometry_bond_index"]
        ideal_bond_length = batched_data["ideal_bond_length"]
        residue_index = batched_data["residue_index"]
        atom_name_id = batched_data["atom_name_id"]
        clash_exclusion_index = batched_data["clash_exclusion_index"]
        atom_plddt = batched_data.get("atom_plddt",None,)
        atom_to_token_idx = batched_data.get("atom_to_token_idx",None,)
        token_pair_confidence = batched_data.get("token_pair_confidence",None,)
        num_tokens = batched_data.get("num_tokens",None,)
        atom_mobility_attr = batched_data.get("atom_mobility_attr",None,)
        if batch is None:
            batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        batch_size = int(batch.max().item()) + 1

        x0 = pos_pred

        x0_centered = center_of_mass(x0, batch=batch)
        x1_centered = center_of_mass(pos, batch=batch)

        identity_pair_fraction = x0_centered.new_zeros(())
        near_native_pair_fraction = x0_centered.new_zeros(())
        if (
            self.use_source_augmentation
            and stage == "train"
        ):
            if batch is None:
                raise ValueError("use_mobility_v1=True requires a batch vector")
            (
                x0_centered,
                identity_pair_fraction,
                near_native_pair_fraction,
            ) = self.augment_residual_source(
                x0_centered=x0_centered,
                x1_centered=x1_centered,
                batch=batch,
                target_mask=target_mask,
            )
            x0_centered = center_of_mass(x0_centered, batch=batch)

        if self.training_objective == "residual":
            t = torch.zeros(batch_size,1,dtype=x0_centered.dtype,device=x0_centered.device,)

            x_t = x0_centered
            u_t = x1_centered - x0_centered
            eps = torch.zeros_like(x_t)
        else:
            t = self.sample_time(num_samples=batch_size,stage=stage,)

            x_t, u_t, eps = self.compute_conditional_vector_field(x0=x0_centered,x1=x1_centered,t=t,batch=batch,)


        # 模型前向传播，预测向量场 v_t
        forward_result = self(
            z=z,
            t=t,
            pos=x_t,
            pos_source=x0_centered,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
            # Modify_4
            atom_plddt=atom_plddt,
            atom_to_token_idx=atom_to_token_idx,
            token_pair_confidence=token_pair_confidence,
            num_tokens=num_tokens,
            # Modify_5
            atom_mobility_attr=atom_mobility_attr,
            geometry_bond_index=geometry_bond_index,
            ideal_bond_length=ideal_bond_length,
            clash_exclusion_index=clash_exclusion_index,
            return_aux=self.use_mobility_v1,
        )
        if self.use_mobility_v1:
            v_t, mobility_aux = forward_result
        else:
            v_t = forward_result
            mobility_aux = None

        # Modify_3
        # 根据网络输出构造几何 loss 使用的预测终点
        t_atom = unsqueeze_like(t[batch] if batch is not None else t,target=x_t,)

        if self.training_objective == "residual":
            # v_t 直接表示 x1 - x0
            pos_estimate = x0_centered + v_t
        elif self.flow_path == "deterministic":
            # x_t = (1 - t) * x0 + t * x1
            # 理想情况下 v_t = x1 - x0
            pos_estimate = x_t + (1 - t_atom) * v_t
        else:
            # 去掉随机路径中的已知噪声速度
            clean_displacement = (v_t- self.sigma_dot_t(t_atom) * eps)
            pos_estimate = x0_centered + clean_displacement

        metrics = self.compute_losses(v_t, u_t, batch, target_mask, pos_estimate, geometry_bond_index, ideal_bond_length, z, clash_exclusion_index, residue_index, atom_name_id, x0_centered, x1_centered, mobility_aux)
        self.record_metrics(stage, batch_size, batch, pos_estimate, x0_centered,
                            x1_centered, target_mask, identity_pair_fraction,
                            near_native_pair_fraction, **metrics)
        return metrics["loss"]
