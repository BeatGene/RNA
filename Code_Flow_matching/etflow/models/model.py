from typing import Any, Dict, Optional, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning import seed_everything
from torch_geometric.utils import scatter


from etflow.models.base import BaseModel
from etflow.data.constants import BASE_ATOM_NAME_IDS
# Modify_3
from etflow.models.loss import (
      atomwise_bond_error,
      atomwise_steric_clash_score,
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
            dynamic_radius_cutoff: float = 4.5,
            # max_num_neighbors: int = 32,
            so3_equivariant: bool = False,
            source_conditioning: bool = True,
            confidence_conditioning: bool = False,
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

            # Modify_5
            use_mobility_v1: bool = False,
            mobility_gate_loss_weight: float = 0.1,
            no_regret_loss_weight: float = 0.5,
            protect_loss_weight: float = 0.2,
            velocity_budget_loss_weight: float = 0.01,
            identity_pair_probability: float = 0.1,
            near_native_pair_probability: float = 0.2,
            near_native_min_alpha: float = 0.05,
            near_native_max_alpha: float = 0.25,
            mobility_good_error: float = 0.3,
            mobility_move_error: float = 1.5,
            protect_error_threshold: float = 0.5,
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

        if use_mobility_v1 and training_objective != "residual":
            raise ValueError(
                "use_mobility_v1=True is only supported when "
                "training_objective='residual'"
            )

        if not 0.0 <= identity_pair_probability <= 1.0:
            raise ValueError("identity_pair_probability must be in [0, 1]")

        if not 0.0 <= near_native_pair_probability <= 1.0:
            raise ValueError("near_native_pair_probability must be in [0, 1]")

        if identity_pair_probability + near_native_pair_probability > 1.0:
            raise ValueError(
                "identity_pair_probability + near_native_pair_probability "
                "must not exceed 1"
            )

        if not 0.0 <= near_native_min_alpha <= near_native_max_alpha <= 1.0:
            raise ValueError(
                "near-native alpha bounds must satisfy "
                "0 <= min <= max <= 1"
            )

        if not 0.0 <= mobility_good_error < mobility_move_error:
            raise ValueError(
                "mobility error bounds must satisfy "
                "0 <= good_error < move_error"
            )

        if protect_error_threshold < 0.0:
            raise ValueError("protect_error_threshold must be non-negative")

        if flow_path not in {"deterministic", "stochastic"}:
            raise ValueError(
                f"Unknown flow_path: {flow_path}"
            )

        if flow_path == "stochastic" and sigma <= 0:
            raise ValueError(
                "stochastic flow requires sigma > 0"
            )

        if dynamic_radius_cutoff <= 0:
            raise ValueError(
                "dynamic_radius_cutoff must be positive"
            )

        if dynamic_radius_cutoff > cutoff_upper:
            raise ValueError(
                "dynamic_radius_cutoff should not exceed cutoff_upper"
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
        #Modify_4
        self.confidence_conditioning = confidence_conditioning
        self.confidence_node_attr_dim = 1
        self.confidence_edge_attr_dim = 4

        network_node_attr_dim = node_attr_dim
        network_edge_attr_dim = edge_attr_dim

        if self.confidence_conditioning:
            network_node_attr_dim += self.confidence_node_attr_dim
            network_edge_attr_dim += self.confidence_edge_attr_dim
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
                # Modify_4
                node_attr_dim=network_node_attr_dim,
                edge_attr_dim=network_edge_attr_dim,
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

        # Modify_5: first-version residue mobility refinement.
        self.use_mobility_v1 = use_mobility_v1
        self.mobility_gate_loss_weight = mobility_gate_loss_weight
        self.no_regret_loss_weight = no_regret_loss_weight
        self.protect_loss_weight = protect_loss_weight
        self.velocity_budget_loss_weight = velocity_budget_loss_weight
        self.identity_pair_probability = identity_pair_probability
        self.near_native_pair_probability = near_native_pair_probability
        self.near_native_min_alpha = near_native_min_alpha
        self.near_native_max_alpha = near_native_max_alpha
        self.mobility_good_error = mobility_good_error
        self.mobility_move_error = mobility_move_error
        self.protect_error_threshold = protect_error_threshold

        if self.use_mobility_v1:
            # h_i + pLDDT + four pair summaries + bond error + clash score.
            self.mobility_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels + 7, hidden_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_channels, 1),
            )
            torch.nn.init.zeros_(self.mobility_head[-1].weight)
            torch.nn.init.constant_(self.mobility_head[-1].bias, -1.4)

        # Modify_2
        self.dynamic_graph = dynamic_graph
        self.max_num_neighbors = max_num_neighbors
        self.num_edge_types = num_edge_types
        self.dynamic_edge_type = dynamic_edge_type
        self.sigma = sigma
        self.sample_time_dist = sample_time_dist
        self.cutoff = dynamic_radius_cutoff
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

    #Modify_4
    def build_confidence_edge_attr(
            self,
            edge_index: Tensor,
            batch: Optional[Tensor],
            atom_to_token_idx: Tensor,
            token_pair_confidence: Tensor,
            num_tokens: Tensor,
    ) -> Tensor:
        if atom_to_token_idx is None:
            raise ValueError(
                "confidence_conditioning=True requires atom_to_token_idx"
            )

        if token_pair_confidence is None:
            raise ValueError(
                "confidence_conditioning=True requires "
                "token_pair_confidence"
            )

        if num_tokens is None:
            raise ValueError(
                "confidence_conditioning=True requires num_tokens"
            )

        atom_to_token_idx = atom_to_token_idx.long().view(-1)
        num_tokens = num_tokens.long().view(-1)

        if batch is not None and atom_to_token_idx.numel() != batch.numel():
            raise ValueError(
                "atom_to_token_idx and batch must have the same number of atoms"
            )

        if token_pair_confidence.dim() != 2:
            raise ValueError(
                "token_pair_confidence must have shape "
                "[sum(num_tokens**2), 4]"
            )

        if (
                token_pair_confidence.size(1)
                != self.confidence_edge_attr_dim
        ):
            raise ValueError(
                "token_pair_confidence must contain exactly "
                f"{self.confidence_edge_attr_dim} features"
            )

        if batch is None:
            batch = torch.zeros(
                atom_to_token_idx.numel(),
                dtype=torch.long,
                device=atom_to_token_idx.device,
            )

        num_graphs = (
            int(batch.max().item()) + 1
            if batch.numel() > 0
            else 0
        )
        if num_tokens.numel() != num_graphs:
            raise ValueError(
                "num_tokens must contain exactly one value per graph"
            )

        pair_count = num_tokens.square()

        expected_pair_rows = pair_count.sum()

        if (
                token_pair_confidence.size(0)
                != int(expected_pair_rows.item())
        ):
            raise ValueError(
                "token_pair_confidence row count does not equal "
                "sum(num_tokens ** 2)"
            )

        pair_offset = torch.cat(
            [
                pair_count.new_zeros(1),
                pair_count.cumsum(dim=0)[:-1],
            ],
            dim=0,
        )

        edge_source = edge_index[0]
        edge_target = edge_index[1]

        edge_batch = batch[edge_source]

        if not torch.equal(
                edge_batch,
                batch[edge_target],
        ):
            raise ValueError(
                "Found an edge connecting two different graphs"
            )

        token_source = atom_to_token_idx[edge_source]
        token_target = atom_to_token_idx[edge_target]

        edge_num_tokens = num_tokens[edge_batch]

        invalid_token_mask = (
                (token_source < 0)
                | (token_target < 0)
                | (token_source >= edge_num_tokens)
                | (token_target >= edge_num_tokens)
        )

        if invalid_token_mask.any():
            raise ValueError(
                "atom_to_token_idx contains an invalid local token id"
            )

        flat_pair_index = (
                pair_offset[edge_batch]
                + token_source * edge_num_tokens
                + token_target
        )

        return token_pair_confidence[
            flat_pair_index
        ]

    # Modify_5
    def build_global_residue_index(
            self,
            batch: Tensor,
            atom_to_token_idx: Tensor,
            num_tokens: Tensor,
    ):
        """Convert per-graph token IDs into batch-global residue IDs."""
        if batch is None:
            batch = torch.zeros(
                atom_to_token_idx.numel(),
                dtype=torch.long,
                device=atom_to_token_idx.device,
            )

        atom_to_token_idx = atom_to_token_idx.long().view(-1)
        num_tokens = num_tokens.long().view(-1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        if atom_to_token_idx.numel() != batch.numel():
            raise ValueError(
                "atom_to_token_idx and batch must have the same number of atoms"
            )
        if num_tokens.numel() != num_graphs:
            raise ValueError("num_tokens must contain one value per graph")

        token_offset = torch.cat(
            [
                num_tokens.new_zeros(1),
                num_tokens.cumsum(dim=0)[:-1],
            ],
            dim=0,
        )
        global_residue_index = (
            token_offset[batch] + atom_to_token_idx
        )
        total_residues = int(num_tokens.sum().item())

        if (
                global_residue_index.numel() > 0
                and (
                global_residue_index.min() < 0
                or global_residue_index.max() >= total_residues
        )
        ):
            raise ValueError("atom_to_token_idx contains an invalid residue id")

        return global_residue_index, total_residues

    # Modify_5
    def augment_residual_source(
            self,
            x0_centered: Tensor,
            x1_centered: Tensor,
            batch: Tensor,
    ):
        """Apply graph-level identity/near-native training augmentation."""
        num_graphs = int(batch.max().item()) + 1
        random_value = torch.rand(
            num_graphs,
            device=x0_centered.device,
        )
        identity_graph_mask = (
            random_value < self.identity_pair_probability
        )
        near_native_graph_mask = (
            (random_value >= self.identity_pair_probability)
            & (
                random_value
                < self.identity_pair_probability
                + self.near_native_pair_probability
            )
        )

        alpha = torch.empty(
            num_graphs,
            1,
            dtype=x0_centered.dtype,
            device=x0_centered.device,
        ).uniform_(
            self.near_native_min_alpha,
            self.near_native_max_alpha,
        )
        near_native_source = (
            x1_centered
            + alpha[batch] * (x0_centered - x1_centered)
        )
        identity_atom_mask = identity_graph_mask[batch].unsqueeze(-1)
        near_native_atom_mask = near_native_graph_mask[batch].unsqueeze(-1)

        augmented_source = torch.where(
            identity_atom_mask,
            x1_centered,
            x0_centered,
        )
        augmented_source = torch.where(
            near_native_atom_mask,
            near_native_source,
            augmented_source,
        )
        return (
            augmented_source,
            identity_graph_mask.float().mean(),
            near_native_graph_mask.float().mean(),
        )

    # Modify_5
    @staticmethod
    def residue_rms_error(
            prediction: Tensor,
            target: Tensor,
            global_residue_index: Tensor,
            total_residues: int,
    ) -> Tensor:
        atom_square_error = (
            prediction - target
        ).square().sum(dim=-1)
        return torch.sqrt(
            scatter(
                atom_square_error,
                global_residue_index,
                dim=0,
                dim_size=total_residues,
                reduce="mean",
            ).clamp_min(1.0e-8)
        )

    # Modify_5
    def apply_residue_mobility_gate(
            self,
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
        required_values = {
            "atom_plddt": atom_plddt,
            "atom_mobility_attr": atom_mobility_attr,
            "atom_to_token_idx": atom_to_token_idx,
            "num_tokens": num_tokens,
            "geometry_bond_index": geometry_bond_index,
            "ideal_bond_length": ideal_bond_length,
            "clash_exclusion_index": clash_exclusion_index,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            raise ValueError(
                "use_mobility_v1=True requires: " + ", ".join(missing)
            )

        atom_plddt = atom_plddt.to(
            dtype=hidden.dtype,
            device=hidden.device,
        ).view(-1, 1)
        atom_mobility_attr = atom_mobility_attr.to(
            dtype=hidden.dtype,
            device=hidden.device,
        )
        if atom_mobility_attr.shape != (hidden.size(0), 4):
            raise ValueError("atom_mobility_attr must have shape [num_atoms, 4]")
        if atom_plddt.size(0) != hidden.size(0):
            raise ValueError("atom_plddt must contain one value per atom")

        source_bond_error = atomwise_bond_error(
            prediction=pos_source,
            geometry_bond_index=geometry_bond_index,
            ideal_bond_length=ideal_bond_length,
        ).to(dtype=hidden.dtype, device=hidden.device)
        source_clash_score = atomwise_steric_clash_score(
            prediction=pos_source,
            atomic_numbers=z,
            batch=batch,
            geometry_bond_index=geometry_bond_index,
            vdw_radius_table=self.vdw_radius_table,
            clash_exclusion_index=clash_exclusion_index,
        ).to(dtype=hidden.dtype, device=hidden.device)
        atom_gate_input = torch.cat(
            [
                hidden,
                atom_plddt.clamp(0.0, 1.0),
                atom_mobility_attr,
                source_bond_error,
                source_clash_score,
            ],
            dim=-1,
        )
        global_residue_index, total_residues = (
            self.build_global_residue_index(
                batch=batch,
                atom_to_token_idx=atom_to_token_idx,
                num_tokens=num_tokens,
            )
        )
        residue_gate_input = scatter(
            atom_gate_input,
            global_residue_index,
            dim=0,
            dim_size=total_residues,
            reduce="mean",
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

    def forward(
            self,
            z: Tensor,
            t: Tensor,
            pos: Tensor,
            bond_index: Tensor,
            pos_source: Tensor,
            edge_attr: Optional[Tensor] = None,
            node_attr: Optional[Tensor] = None,
            batch: Optional[Tensor] = None,
            # Modify_4
            atom_plddt: Optional[Tensor] = None,
            atom_to_token_idx: Optional[Tensor] = None,
            token_pair_confidence: Optional[Tensor] = None,
            num_tokens: Optional[Tensor] = None,
            # Modify_5
            atom_mobility_attr: Optional[Tensor] = None,
            geometry_bond_index: Optional[Tensor] = None,
            ideal_bond_length: Optional[Tensor] = None,
            clash_exclusion_index: Optional[Tensor] = None,
            return_aux: bool = False,
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
        # Modify_4
        if self.confidence_conditioning:
            if node_attr is None:
                raise ValueError(
                    "confidence_conditioning=True requires node_attr"
                )

            if atom_plddt is None:
                raise ValueError(
                    "confidence_conditioning=True requires atom_plddt"
                )

            atom_plddt = atom_plddt.to(
                dtype=node_attr.dtype,
                device=node_attr.device,
            ).view(-1, 1)

            if atom_plddt.size(0) != node_attr.size(0):
                raise ValueError(
                    "atom_plddt and node_attr must have the "
                    "same number of atoms"
                )

            node_attr = torch.cat(
                [
                    node_attr,
                    atom_plddt.clamp(0.0, 1.0),
                ],
                dim=-1,
            )
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
        # Modify_4
        if self.confidence_conditioning:
            if edge_type is None:
                raise ValueError(
                    "confidence_conditioning=True requires edge_attr"
                )

            confidence_edge_attr = self.build_confidence_edge_attr(
                edge_index=edge_index,
                batch=batch,
                atom_to_token_idx=atom_to_token_idx,
                token_pair_confidence=token_pair_confidence,
                num_tokens=num_tokens,
            )

            confidence_edge_attr = confidence_edge_attr.to(
                dtype=edge_type.dtype,
                device=edge_type.device,
            )

            edge_type = torch.cat(
                [
                    edge_type,
                    confidence_edge_attr,
                ],
                dim=-1,
            )
        network_kwargs = {
            "z": z,
            "t": t[batch],
            "pos": pos,
            "pos_source": pos_source,
            "edge_index": edge_index,
            "edge_attr": edge_type,
            "node_attr": node_attr,
            "batch": batch,
        }
        if self.use_mobility_v1:
            raw_velocity, hidden = self.network(
                **network_kwargs,
                return_hidden=True,
            )
            v_t, mobility_aux = self.apply_residue_mobility_gate(
                raw_velocity=raw_velocity,
                hidden=hidden,
                pos_source=pos_source,
                z=z,
                batch=batch,
                atom_plddt=atom_plddt,
                atom_mobility_attr=atom_mobility_attr,
                atom_to_token_idx=atom_to_token_idx,
                num_tokens=num_tokens,
                geometry_bond_index=geometry_bond_index,
                ideal_bond_length=ideal_bond_length,
                clash_exclusion_index=clash_exclusion_index,
            )
            if return_aux:
                return v_t, mobility_aux
            return v_t

        v_t = self.network(**network_kwargs)
        if return_aux:
            return v_t, {}
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
        # Modify_4
        atom_plddt = batched_data.get(
            "atom_plddt",
            None,
        )
        atom_to_token_idx = batched_data.get(
            "atom_to_token_idx",
            None,
        )
        token_pair_confidence = batched_data.get(
            "token_pair_confidence",
            None,
        )
        num_tokens = batched_data.get(
            "num_tokens",
            None,
        )
        # Modify_5
        atom_mobility_attr = batched_data.get(
            "atom_mobility_attr",
            None,
        )
        batch_size = int(batch.max().item()) + 1 if batch is not None else 1

        # 【核心修改】：流匹配的起点不再是噪声，而是预测结构
        x0 = pos_pred

        # Modify_3
        x0_centered = center_of_mass(x0, batch=batch)
        x1_centered = center_of_mass(pos, batch=batch)

        identity_pair_fraction = x0_centered.new_zeros(())
        near_native_pair_fraction = x0_centered.new_zeros(())
        if self.use_mobility_v1 and stage == "train":
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
            )
            # Keep source conditioning and residual targets on the same
            # augmented, already-centered source structure.
            x0 = x0_centered

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
        forward_result = self(
            z=z,
            t=t,
            pos=x_t,
            pos_source=x0,
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

        # Modify_5: first-version mobility/no-regret objectives.
        if self.use_mobility_v1:
            global_residue_index = mobility_aux["global_residue_index"]
            total_residues = mobility_aux["total_residues"]
            raw_residue_error = self.residue_rms_error(
                prediction=x0_centered,
                target=x1_centered,
                global_residue_index=global_residue_index,
                total_residues=total_residues,
            )
            refined_residue_error = self.residue_rms_error(
                prediction=pos_estimate,
                target=x1_centered,
                global_residue_index=global_residue_index,
                total_residues=total_residues,
            )
            mobility_target = (
                (raw_residue_error - self.mobility_good_error)
                / (self.mobility_move_error - self.mobility_good_error)
            ).clamp(0.0, 1.0)
            mobility_residue = mobility_aux["mobility_residue"].view(-1)
            mobility_loss = F.smooth_l1_loss(
                mobility_residue,
                mobility_target,
            )
            no_regret_loss = torch.relu(
                refined_residue_error - raw_residue_error
            ).mean()

            correct_residue_mask = (
                raw_residue_error < self.protect_error_threshold
            )
            protect_atom_mask = correct_residue_mask[
                global_residue_index
            ].to(dtype=pos_estimate.dtype)
            atom_movement = (
                pos_estimate - x0_centered
            ).square().sum(dim=-1)
            protect_loss = (
                atom_movement * protect_atom_mask
            ).sum() / protect_atom_mask.sum().clamp_min(1.0)

            raw_velocity_square = mobility_aux[
                "raw_velocity"
            ].square().sum(dim=-1, keepdim=True)
            velocity_budget_loss = (
                (1.0 - mobility_aux["mobility_atom"].detach())
                * raw_velocity_square
            ).mean()

            loss = (
                loss
                + self.mobility_gate_loss_weight * mobility_loss
                + self.no_regret_loss_weight * no_regret_loss
                + self.protect_loss_weight * protect_loss
                + self.velocity_budget_loss_weight * velocity_budget_loss
            )

        if not torch.isfinite(loss):
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
        if self.use_mobility_v1:
            for metric_name, metric_value in {
                "mobility_loss": mobility_loss,
                "no_regret_loss": no_regret_loss,
                "protect_loss": protect_loss,
                "velocity_budget_loss": velocity_budget_loss,
                "mobility_mean": mobility_residue.mean(),
                "mobility_target_mean": mobility_target.mean(),
                "identity_pair_fraction": identity_pair_fraction,
                "near_native_pair_fraction": near_native_pair_fraction,
            }.items():
                self.log_helper(
                    f"{stage}/{metric_name}",
                    metric_value,
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
                atom_plddt=atom_plddt,
                atom_to_token_idx=atom_to_token_idx,
                token_pair_confidence=token_pair_confidence,
                num_tokens=num_tokens,
                atom_mobility_attr=atom_mobility_attr,
                geometry_bond_index=geometry_bond_index,
                ideal_bond_length=ideal_bond_length,
                clash_exclusion_index=clash_exclusion_index,
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
            #Modify_4
            v_t = self(
                z=z,
                t=t,
                pos=x,
                pos_source=source,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
                atom_plddt=atom_plddt,
                atom_to_token_idx=atom_to_token_idx,
                token_pair_confidence=token_pair_confidence,
                num_tokens=num_tokens,
                atom_mobility_attr=atom_mobility_attr,
                geometry_bond_index=geometry_bond_index,
                ideal_bond_length=ideal_bond_length,
                clash_exclusion_index=clash_exclusion_index,
            )
            # 沿着向量场前进一小步
            x = x + delta_t * v_t

        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
