from typing import Optional
import torch
from etflow.models.base import BaseModel
from etflow.data.constants import BASE_ATOM_NAME_IDS


def initialize(
        self,
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
        use_source_augmentation: bool = True,
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
        representation_type: str = "torchmd",
        output_head_type: str = "gated",
        representation_args: Optional[dict] = None,
        confidence_strength: float = 1.0,
        validate_inputs: bool = True,
        **kwargs,
):
    BaseModel.__init__(self, **kwargs)
    if dynamic_graph and edge_attr_dim != num_edge_types:
        raise ValueError(
            "dynamic_graph=True requires edge_attr_dim "
            f"({edge_attr_dim}) to equal num_edge_types ({num_edge_types})"
        )
    if training_objective not in {"flow", "residual"}:
        raise ValueError(f"Unknown training_objective: {training_objective}")
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
            "near-native alpha bounds must satisfy 0 <= min <= max <= 1"
        )
    if not 0.0 <= mobility_good_error < mobility_move_error:
        raise ValueError(
            "mobility error bounds must satisfy 0 <= good_error < move_error"
        )
    if protect_error_threshold < 0.0:
        raise ValueError("protect_error_threshold must be non-negative")
    if flow_path not in {"deterministic", "stochastic"}:
        raise ValueError(f"Unknown flow_path: {flow_path}")
    if flow_path == "stochastic" and sigma <= 0:
        raise ValueError("stochastic flow requires sigma > 0")
    if dynamic_radius_cutoff <= 0:
        raise ValueError("dynamic_radius_cutoff must be positive")
    if dynamic_radius_cutoff > cutoff_upper:
        raise ValueError("dynamic_radius_cutoff should not exceed cutoff_upper")

    weights = (bond_loss_weight, clash_loss_weight, plane_loss_weight,
               mobility_gate_loss_weight, no_regret_loss_weight,
               protect_loss_weight, velocity_budget_loss_weight)
    import math
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("loss weights must be finite and non-negative")
    if not 0 <= dynamic_edge_type < num_edge_types:
        raise ValueError("dynamic_edge_type outside edge type range")
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

    self.confidence_conditioning = confidence_conditioning
    self.confidence_node_attr_dim = 1
    self.confidence_edge_attr_dim = 4

    network_node_attr_dim = node_attr_dim
    network_edge_attr_dim = edge_attr_dim

    if self.confidence_conditioning:
        network_node_attr_dim += self.confidence_node_attr_dim
        network_edge_attr_dim += self.confidence_edge_attr_dim

    from etflow.networks.factory import build_network
    self.network = build_network(
        network_type=network_type, representation_type=representation_type,
        output_head_type=output_head_type, representation_args=representation_args,
        hidden_channels=hidden_channels, num_layers=num_layers, num_rbf=num_rbf,
        rbf_type=rbf_type, trainable_rbf=trainable_rbf, activation=activation,
        neighbor_embedding=neighbor_embedding, cutoff_lower=cutoff_lower,
        cutoff_upper=cutoff_upper, max_z=max_z, node_attr_dim=network_node_attr_dim,
        edge_attr_dim=network_edge_attr_dim, attn_activation=attn_activation,
        num_heads=num_heads, distance_influence=distance_influence, reduce_op=reduce_op,
        qk_norm=qk_norm, output_layer_norm=output_layer_norm,
        clip_during_norm=clip_during_norm, so3_equivariant=so3_equivariant,
        source_conditioning=source_conditioning)
    if not 0 <= confidence_strength <= 1:
        raise ValueError("confidence_strength must be in [0, 1]")
    self.confidence_strength = confidence_strength
    self.validate_inputs = validate_inputs
    # Modify_5: first-version residue mobility refinement.
    self.use_mobility_v1 = use_mobility_v1
    self.use_source_augmentation = use_source_augmentation
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

