import torch
from torch import Tensor
from etflow.models.utils import center_of_mass, unsqueeze_like


class FlowPathMixin:
    def sigma_t(self, t):
        if self.flow_path == "deterministic":
            return torch.zeros_like(t)

        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        if self.flow_path == "deterministic":
            return torch.zeros_like(t)

        return (self.sigma* 0.5* (1 - 2 * t)/ torch.sqrt(t * (1 - t)))

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

        x0_centered = center_of_mass(x0, batch=batch)
        x1_centered = center_of_mass(x1, batch=batch)
        # 获取 t 时刻的扰动坐标 x_t 和噪声 eps
        x_t, eps = self.sample_conditional_pt(x0_centered, x1_centered, t, batch=batch)

        t_atom = unsqueeze_like(t[batch], x1_centered)

        # 真实的目标向量场 u_t：指向 x1 - x0 的方向，并加上噪声的导数
        u_t = x1_centered - x0_centered + self.sigma_dot_t(t_atom) * eps

        return x_t, u_t, eps

    def sample_time(self,num_samples: int,low: float = 1e-4,high: float = 0.9999,stage: str = "train",):
        """均匀采样时间 t"""
        if self.sample_time_dist == "uniform" or stage == "val":
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(low, high)
        raise NotImplementedError(f"Time sampling {self.sample_time_dist} not implemented")

