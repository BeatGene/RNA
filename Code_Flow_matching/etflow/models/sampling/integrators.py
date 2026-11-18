import torch
from torch import Tensor
from etflow.models.utils import center_of_mass
from .solvers import SOLVERS
from .evaluation import evaluation_mode


class SamplingMixin:
    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0
        t_curr, t_next = t_schedule[t: t + 2]
        return t_next - t_curr

    @torch.no_grad()
    @evaluation_mode
    def sample(
            self,
            z: Tensor,
            pos_pred: Tensor,
            bond_index: Tensor,
            batch: Tensor,
            node_attr: Tensor = None,
            edge_attr: Tensor = None,
            #Modify_4
            atom_plddt: Tensor = None,
            atom_to_token_idx: Tensor = None,
            token_pair_confidence: Tensor = None,
            num_tokens: Tensor = None,
            # Modify_5
            atom_mobility_attr: Tensor = None,
            geometry_bond_index: Tensor = None,
            ideal_bond_length: Tensor = None,
            clash_exclusion_index: Tensor = None,
            n_timesteps: int = 50,
            s_churn: float = 1.0,
            std: float = 1.0,
            sampler_type: str = "ode",
            t_min: float = 0.0,
            t_max: float = 1.0,
    ):
        """
        推理 (Inference) / Refinement 阶段。
        输入：
            pos_pred: Protenix 预测的粗糙坐标 (作为初始状态 x)
        输出：
            x: 经过模型优化 (ODE积分) 后的最终精细坐标
        """
        if not isinstance(n_timesteps, int) or n_timesteps < 1:
            raise ValueError("n_timesteps must be a positive integer")
        if sampler_type not in {"ode", "heun", "sde", "residual"}:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
        if not 0 <= t_min < t_max <= 1:
            raise ValueError("require 0 <= t_min < t_max <= 1")
        if std < 0 or s_churn < 0:
            raise ValueError("std and s_churn must be non-negative")
        if sampler_type == "residual" and self.training_objective != "residual":
            raise ValueError("residual sampler requires a residual model")
        batch_size = int(batch.max().item()) + 1 if batch is not None else 1
        t_schedule = torch.linspace(t_min, t_max, steps=n_timesteps + 1, device=self.device)

        # 【核心修改】：推理起点从随机噪声变成了质心居中的 pos_pred
        source = center_of_mass(pos_pred, batch=batch)
        x = source.clone()

        def velocity_at(position, time):
            return self(z=z, t=time, pos=position, pos_source=source,
                        bond_index=bond_index, edge_attr=edge_attr, node_attr=node_attr,
                        batch=batch, atom_plddt=atom_plddt,
                        atom_to_token_idx=atom_to_token_idx,
                        token_pair_confidence=token_pair_confidence, num_tokens=num_tokens,
                        atom_mobility_attr=atom_mobility_attr,
                        geometry_bond_index=geometry_bond_index,
                        ideal_bond_length=ideal_bond_length,
                        clash_exclusion_index=clash_exclusion_index)

        n = t_schedule.size(0) - 1

        if self.training_objective == "residual":
            t = torch.zeros(batch_size,1,dtype=source.dtype,device=source.device,)
            residual = velocity_at(source, t)

            return source + residual

        # 欧拉法 (Euler Method) 解常微分方程 (ODE)
        for i in range(n):
            # Modify_3
            t = torch.full((batch_size, 1),fill_value=t_schedule[i].item(),dtype=x.dtype,device=x.device,)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # 获取当前 t 下的向量场方向
            #Modify_4
            v_t = velocity_at(x, t)
            # 沿着向量场前进一小步
            proposal = SOLVERS[sampler_type](
                x=x, velocity=v_t, dt=delta_t, evaluate=velocity_at,
                next_time=torch.full_like(t, t_schedule[i + 1].item()),
                batch=batch, std=std, s_churn=s_churn)
            x = center_of_mass(proposal, batch=batch)

        return x

