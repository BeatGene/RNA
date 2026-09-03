from typing import Any, Dict, Optional, TypeVar

import torch
from torch import Tensor
from pytorch_lightning import seed_everything


from etflow.models.base import BaseModel
from etflow.data.constants import BASE_ATOM_NAME_IDS
# Modify_3
from etflow.models.loss import (
      base_plane_loss,
      batchwise_l2_loss,
      bond_length_loss,
      steric_clash_loss,
  )
#Modify_2
from etflow.models.utils import (
      center_of_mass,
      merge_dynamic_radius_edges,
      unsqueeze_like,
  )
from etflow.networks.torchmd_net import TorchMDDynamics

__all__ = ["BaseFlow"]

Config = TypeVar("Config", str, Dict[str, Any])


class BaseFlow(BaseModel):

    def __init__(
            self,
            # flow matching network args
            network_type: str = "TorchMDDynamics",
            hidden_channels: int = 128,
            num_layers: int = 8,
            num_rbf: int = 64,
            rbf_type: str = "expnorm",
            trainable_rbf: bool = False,
            activation: str = "silu",
            neighbor_embedding: int = True,
            cutoff_lower: float = 0.0,
            cutoff_upper: float = 10.0,
            max_z: int = 100,
            node_attr_dim: int = 0,
            edge_attr_dim: int = 7,
            attn_activation: str = "silu",
            num_heads: int = 8,
            distance_influence: str = "both",
            reduce_op: str = "sum",
            qk_norm: bool = False,
            output_layer_norm: bool = False,
            clip_during_norm: bool = False,
            # Modify_2
            dynamic_graph: bool = True,
            max_num_neighbors: int = 32,
            num_edge_types: int = 7,
            dynamic_edge_type: int = 3,
            # max_num_neighbors: int = 32,
            so3_equivariant: bool = False,
            source_conditioning: bool = True,
            # flow matching args
            sigma: float = 0.1,
            # prior_type: str = "gaussian",
            sample_time_dist: str = "uniform",
            edge_one_hot: bool = False,
            edge_one_hot_types: int = 3,
            # Modify_3
            training_objective: str = "flow",
            flow_path: str = "deterministic",
            bond_loss_weight: float = 0.1,
            clash_loss_weight: float = 0.01,
            plane_loss_weight: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # Modify_2
        if dynamic_graph and edge_attr_dim != num_edge_types:
            raise ValueError(
                f"dynamic_graph=True requires edge_attr_dim "
                f"({edge_attr_dim}) to equal num_edge_types "
                f"({num_edge_types})"
            )

        # Modify_3
        if training_objective not in {"flow", "residual"}:
            raise ValueError(
                f"Unknown training_objective: {training_objective}"
            )

        if flow_path not in {"deterministic", "stochastic"}:
            raise ValueError(
                f"Unknown flow_path: {flow_path}"
            )

        if flow_path == "stochastic" and sigma <= 0:
            raise ValueError(
                "stochastic flow requires sigma > 0"
            )

        # Modify_3
        vdw_radius_table = torch.zeros(max_z)

        for atomic_number, radius in {
            6: 1.70,
            7: 1.55,
            8: 1.52,
            15: 1.80,
            16: 1.80,
        }.items():
            if atomic_number < max_z:
                vdw_radius_table[atomic_number] = radius

        self.register_buffer(
            "vdw_radius_table",
            vdw_radius_table,
        )
        self.register_buffer(
            "base_atom_name_ids",
            torch.tensor(
                BASE_ATOM_NAME_IDS,
                dtype=torch.long,
            ),
        )
        # setup network
        if network_type == "TorchMDDynamics":
            self.network = TorchMDDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
                clip_during_norm=clip_during_norm,
                so3_equivariant=so3_equivariant,
                source_conditioning=source_conditioning,
            )
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")

        # Modify_2
        self.dynamic_graph = dynamic_graph
        self.max_num_neighbors = max_num_neighbors
        self.num_edge_types = num_edge_types
        self.dynamic_edge_type = dynamic_edge_type
        self.sigma = sigma
        self.sample_time_dist = sample_time_dist
        self.cutoff = cutoff_upper
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types

        # Modify_3
        self.bond_loss_weight = bond_loss_weight
        self.clash_loss_weight = clash_loss_weight
        self.plane_loss_weight = plane_loss_weight

        # Modify_3
        self.training_objective = training_objective
        self.flow_path = flow_path

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg))
        if isinstance(cfg, dict):
            return cls(**cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")

    # Modify_3
    def sigma_t(self, t):
        if self.flow_path == "deterministic":
            return torch.zeros_like(t)

        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        if self.flow_path == "deterministic":
            return torch.zeros_like(t)

        return (
                self.sigma
                * 0.5
                * (1 - 2 * t)
                / torch.sqrt(t * (1 - t))
        )

    def sample_conditional_pt(self, x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor):
        """
        采样时间 t 时的中间态结构 (插值 + 噪声)
        x0: Protenix 预测的结构
        x1: 真实的 RNA 结构
        """
        # 将起点和终点都在各自的质心上对齐，消除平移带来的误差
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # 采样高斯噪声

        # Modify_3
        if self.flow_path == "deterministic":
            eps = torch.zeros_like(x1)
        else:
            eps = torch.randn_like(x1)
            eps = center_of_mass(eps, batch=batch)

        # 线性插值轨迹: t=0 时是 x0(预测), t=1 时是 x1(真实)
        mu_t = (1 - t) * x0 + t * x1

        # 加入随时间变化的噪声 (两端无噪声，中间噪声最大)
        x_t = mu_t + self.sigma_t(t) * eps

        return x_t, eps

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros(x1.size(0), dtype=torch.long, device=self.device)

        # Modify_1
        x0_centered = center_of_mass(x0, batch=batch)
        x1_centered = center_of_mass(x1, batch=batch)
        # 获取 t 时刻的扰动坐标 x_t 和噪声 eps
        # Modify_1
        x_t, eps = self.sample_conditional_pt(x0_centered, x1_centered, t, batch=batch)

        # Modify_1
        t_atom = unsqueeze_like(t[batch], x1_centered)

        # 真实的目标向量场 u_t：指向 x1 - x0 的方向，并加上噪声的导数
        # Modify_1
        u_t = x1_centered - x0_centered + self.sigma_dot_t(t_atom) * eps

        # Modify_3
        return x_t, u_t, eps

    def sample_time(
            self,
            num_samples: int,
            low: float = 1e-4,
            high: float = 0.9999,
            stage: str = "train",
    ):
        """均匀采样时间 t"""
        if self.sample_time_dist == "uniform" or stage == "val":
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        raise NotImplementedError(f"Time sampling {self.sample_time_dist} not implemented")

    def forward(
            self,
            z: Tensor,
            t: Tensor,
            pos: Tensor,
            bond_index: Tensor,
            # Modify_1
            pos_source: Tensor,
            edge_attr: Optional[Tensor] = None,
            node_attr: Optional[Tensor] = None,
            batch: Optional[Tensor] = None,
    ):
        """
        前向传播：网络预测向量场 v_t
        """
        # 为了等变性，每次输入网络前都进行质心居中
        pos = center_of_mass(pos, batch=batch)
        # Modify_1
        pos_source = center_of_mass(pos_source, batch=batch)
        # ToDo
        # edge_index, edge_type = extend_bond_index(
        #     pos=pos,
        #     bond_index=bond_index,
        #     batch=batch,
        #     bond_attr=edge_attr,
        #     device=self.device,
        #     one_hot=self.edge_one_hot,
        #     one_hot_types=self.edge_one_hot_types,
        #     cutoff=self.cutoff,
        #     max_num_neighbors=self.max_num_neighbors,
        # )
        # 【核心修改】：直接使用传入的 pre-computed edge_index
        # 摒弃了原版消耗极大的 extend_bond_index
        # Modify_2
        if self.dynamic_graph:
            if edge_attr is None:
                raise ValueError(
                    "dynamic_graph=True requires typed static edge_attr"
                )

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
        v_t = self.network(
            z=z,
            t=t[batch],
            pos=pos,
            # Modify_1
            pos_source=pos_source,
            edge_index=edge_index,
            edge_attr=edge_type,
            node_attr=node_attr,
            batch=batch,
        )

        return v_t

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        """
        核心训练步。
        """
        # 从 Dataloader 中获取数据
        z = batched_data["atomic_numbers"]
        pos = batched_data["pos"]  # X_1: 真实目标坐标
        pos_pred = batched_data["pos_pred"]  # X_0: 流的起点（Protenix预测坐标）
        bond_index = batched_data["edge_index"]  # 边
        node_attr = batched_data.get("node_attr", None)
        edge_attr = batched_data.get("edge_attr", None)
        batch = batched_data.get("batch", None)
        # Modify_3
        geometry_bond_index = batched_data["geometry_bond_index"]
        ideal_bond_length = batched_data["ideal_bond_length"]
        residue_index = batched_data["residue_index"]
        atom_name_id = batched_data["atom_name_id"]
        clash_exclusion_index = batched_data[
            "clash_exclusion_index"
        ]
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # 【核心修改】：流匹配的起点不再是噪声，而是预测结构
        x0 = pos_pred

        # Modify_3
        x0_centered = center_of_mass(x0, batch=batch)
        x1_centered = center_of_mass(pos, batch=batch)

        if self.training_objective == "residual":
            t = torch.zeros(
                batch_size,
                1,
                dtype=x0.dtype,
                device=x0.device,
            )

            x_t = x0_centered
            u_t = x1_centered - x0_centered
            eps = torch.zeros_like(x_t)

        else:
            t = self.sample_time(
                num_samples=batch_size,
                stage=stage,
            )

            x_t, u_t, eps = (
                self.compute_conditional_vector_field(
                    x0=x0,
                    x1=pos,
                    t=t,
                    batch=batch,
                )
            )

        # 模型前向传播，预测向量场 v_t
        v_t = self(
            z=z,
            t=t,
            pos=x_t,
            # Modify_1
            pos_source=x0,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
        )

        # Modify_3
        # 根据网络输出构造几何 loss 使用的预测终点
        t_atom = unsqueeze_like(
            t[batch] if batch is not None else t,
            target=x_t,
        )

        if self.training_objective == "residual":
            # v_t 直接表示 x1 - x0
            pos_estimate = x0_centered + v_t

        elif self.flow_path == "deterministic":
            # x_t = (1 - t) * x0 + t * x1
            # 理想情况下 v_t = x1 - x0
            pos_estimate = x_t + (1 - t_atom) * v_t

        else:
            # 去掉随机路径中的已知噪声速度
            clean_displacement = (
                    v_t
                    - self.sigma_dot_t(t_atom) * eps
            )
            pos_estimate = x0_centered + clean_displacement

        # Modify_3
        flow_matching_loss = batchwise_l2_loss(
            v_t,
            u_t,
            batch=batch,
            reduce="mean",
        )

        bond_loss = bond_length_loss(
            prediction=pos_estimate,
            geometry_bond_index=geometry_bond_index,
            ideal_bond_length=ideal_bond_length,
        )

        clash_loss = steric_clash_loss(
            prediction=pos_estimate,
            atomic_numbers=z,
            batch=batch,
            geometry_bond_index=geometry_bond_index,
            vdw_radius_table=self.vdw_radius_table,
            clash_exclusion_index=clash_exclusion_index,
        )

        plane_loss = base_plane_loss(
            prediction=pos_estimate,
            residue_index=residue_index,
            atom_name_id=atom_name_id,
            batch=batch,
            base_atom_name_ids=self.base_atom_name_ids,
        )

        loss = (
                flow_matching_loss
                + self.bond_loss_weight * bond_loss
                + self.clash_loss_weight * clash_loss
                + self.plane_loss_weight * plane_loss
        )

        if torch.isnan(loss):
            raise ValueError("Loss 出现 NaN，请检查数据集是否异常！")

        # 记录 Loss
        #Modify_3
        self.log_helper(
            f"{stage}/flow_matching_loss",
            flow_matching_loss,
            batch_size=batch_size,
        )
        self.log_helper(
            f"{stage}/bond_loss",
            bond_loss,
            batch_size=batch_size,
        )
        self.log_helper(
            f"{stage}/clash_loss",
            clash_loss,
            batch_size=batch_size,
        )
        self.log_helper(
            f"{stage}/plane_loss",
            plane_loss,
            batch_size=batch_size,
        )
        self.log_helper(
            f"{stage}/loss",
            loss,
            batch_size=batch_size,
        )

        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0
        t_curr, t_next = t_schedule[t: t + 2]
        return t_next - t_curr

    @torch.no_grad()
    def sample(
            self,
            z: Tensor,
            pos_pred: Tensor,
            bond_index: Tensor,
            batch: Tensor,
            node_attr: Tensor = None,
            edge_attr: Tensor = None,
            n_timesteps: int = 50,

            s_churn: float = 1.0,
            std: float = 1.0,
    ):
        """
        推理 (Inference) / Refinement 阶段。
        输入：
            pos_pred: Protenix 预测的粗糙坐标 (作为初始状态 x)
        输出：
            x: 经过模型优化 (ODE积分) 后的最终精细坐标
        """
        #Modify_3
        batch_size = (
            int(batch.max().item()) + 1
            if batch is not None
            else 1
        )
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)

        # 【核心修改】：推理起点从随机噪声变成了质心居中的 pos_pred
        # Modify_1
        source = center_of_mass(pos_pred, batch=batch)
        x = source.clone()

        n = t_schedule.size(0) - 1

        # Modify_3
        if self.training_objective == "residual":
            t = torch.zeros(
                batch_size,
                1,
                dtype=source.dtype,
                device=source.device,
            )

            residual = self(
                z=z,
                t=t,
                pos=source,
                pos_source=source,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )

            return source + residual

        # 欧拉法 (Euler Method) 解常微分方程 (ODE)
        for i in range(n):
            # Modify_3
            t = torch.full(
                (batch_size, 1),
                fill_value=t_schedule[i].item(),
                dtype=x.dtype,
                device=x.device,
            )
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # 获取当前 t 下的向量场方向
            v_t = self(
                z=z,
                t=t,
                pos=x,
                # Modify_1
                pos_source=source,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )
            # 沿着向量场前进一小步
            x = x + delta_t * v_t

        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
