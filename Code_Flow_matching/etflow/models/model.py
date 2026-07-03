from typing import Any, Dict, Optional, TypeVar

import torch
from torch import Tensor
from pytorch_lightning import seed_everything


from etflow.models.base import BaseModel
from etflow.models.loss import batchwise_l2_loss
from etflow.models.utils import (
    center_of_mass,
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
            edge_attr_dim: int = 0,
            attn_activation: str = "silu",
            num_heads: int = 8,
            distance_influence: str = "both",
            reduce_op: str = "sum",
            qk_norm: bool = False,
            output_layer_norm: bool = False,
            clip_during_norm: bool = False,
            so3_equivariant: bool = False,
            # flow matching args
            sigma: float = 0.1,
            sample_time_dist: str = "uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)

        # 1. 搭建等变图神经网络主干
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
            )
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")


        self.sigma = sigma
        self.sample_time_dist = sample_time_dist

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg))
        if isinstance(cfg, dict):
            return cls(**cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

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
        eps = torch.randn_like(x1)
        eps = center_of_mass(eps, batch=batch)

        # 线性插值轨迹: t=0 时是 x0(预测), t=1 时是 x1(真实)
        mu_t = (1 - t) * x0 + t * x1

        # 加入随时间变化的噪声 (两端无噪声，中间噪声最大)
        x_t = mu_t + self.sigma_t(t) * eps

        return x_t, eps

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros((x1.size(0),), dtype=torch.long, device=self.device)

        # 获取 t 时刻的扰动坐标 x_t 和噪声 eps
        x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch)
        t = unsqueeze_like(t[batch], x1)

        # 真实的目标向量场 u_t：指向 x1 - x0 的方向，并加上噪声的导数
        u_t = x1 - x0 + self.sigma_dot_t(t) * eps

        return x_t, u_t

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
            edge_index: Tensor,
            node_attr: Optional[Tensor] = None,
            batch: Optional[Tensor] = None,
    ):
        """
        前向传播：网络预测向量场 v_t
        """
        # 为了等变性，每次输入网络前都进行质心居中
        pos = center_of_mass(pos, batch=batch)
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
        v_t = self.network(
            z=z,
            t=t[batch],
            pos=pos,
            edge_index=edge_index,
            edge_attr=None,  # 我们暂时没有对 RNA 提取边特征，设为 None
            node_attr=node_attr,
            batch=batch,
        )

        return v_t

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        """
        核心训练步。
        """
        # 从 Dataloader 中获取数据
        z = batched_data["z"]
        pos = batched_data["pos"]  # X_1: 真实目标坐标
        pos_pred = batched_data["pos_pred"]  # X_0: 流的起点（Protenix预测坐标）
        edge_index = batched_data["edge_index"]  # 空间邻居边
        node_attr = batched_data.get("node_attr", None)
        batch = batched_data.get("batch", None)

        batch_size = batch.max().item() + 1 if batch is not None else 1

        # 【核心修改】：流匹配的起点不再是噪声，而是你的预测结构！
        x0 = pos_pred

        # 采样时间步 t
        t = self.sample_time(num_samples=batch_size, stage=stage)

        # 获取 t 时刻的加噪状态 x_t，以及模型应该去回归的真实向量场 u_t
        x_t, u_t = self.compute_conditional_vector_field(
            x0=x0, x1=pos, t=t, batch=batch
        )

        # 模型前向传播，预测向量场 v_t
        v_t = self(
            z=z,
            t=t,
            pos=x_t,
            edge_index=edge_index,
            node_attr=node_attr,
            batch=batch,
        )

        # 计算 MSE 损失：让模型预测的 v_t 逼近真实的推移方向 u_t
        loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")

        if torch.isnan(loss):
            raise ValueError("Loss 出现 NaN，请检查数据集是否异常！")

        # 记录 Loss
        self.log_helper(f"{stage}/flow_matching_loss", loss, batch_size=batch_size)
        self.log_helper(f"{stage}/loss", loss, batch_size=batch_size)

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
            edge_index: Tensor,
            batch: Tensor,
            node_attr: Tensor = None,
            n_timesteps: int = 50,
    ):
        """
        推理 (Inference) / Refinement 阶段。
        输入：
            pos_pred: Protenix 预测的粗糙坐标 (作为初始状态 x)
        输出：
            x: 经过模型优化 (ODE积分) 后的最终精细坐标
        """
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)

        # 【核心修改】：推理起点从随机噪声变成了质心居中的 pos_pred
        x = center_of_mass(pos_pred, batch=batch)

        n = t_schedule.size(0) - 1

        # 欧拉法 (Euler Method) 解常微分方程 (ODE)
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # 获取当前 t 下的向量场方向
            v_t = self(
                z=z,
                t=t,
                pos=x,
                edge_index=edge_index,
                node_attr=node_attr,
                batch=batch,
            )
            # 沿着向量场前进一小步
            x = x + delta_t * v_t

        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")