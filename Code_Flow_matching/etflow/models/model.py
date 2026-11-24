"""Public Lightning entry point. A inputs -> B representation -> C head -> D gate -> E output."""
from typing import Any, Dict, Optional, TypeVar
import torch
from torch import Tensor
from etflow.models.base import BaseModel
from etflow.models.construction import initialize
from etflow.models.conditioning.inputs import prepare_network_inputs
from etflow.models.conditioning.confidence import ConfidenceMixin
from etflow.models.conditioning.mobility import MobilityMixin
from etflow.models.training.paths import FlowPathMixin
from etflow.models.training.augmentation import SourceAugmentationMixin
from etflow.models.training.step import TrainingMixin
from etflow.models.training.metrics import MetricsMixin
from etflow.models.losses.composition import LossMixin
from etflow.models.sampling.integrators import SamplingMixin

Config = TypeVar("Config", str, Dict[str, Any])
__all__ = ["BaseFlow"]

class BaseFlow(TrainingMixin, MetricsMixin, LossMixin, FlowPathMixin,
               SourceAugmentationMixin, ConfidenceMixin, MobilityMixin,
               SamplingMixin, BaseModel):
    __init__ = initialize

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        if isinstance(cfg, str):
            with open(cfg) as stream:
                cfg = yaml.safe_load(stream)
        if isinstance(cfg, dict):
            return cls(**cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")


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
        # A: validated graph + current/source + optional confidence -> network inputs
        inputs = prepare_network_inputs(
            self, z, t, pos, bond_index, pos_source, edge_attr, node_attr, batch,
            atom_plddt, atom_to_token_idx, token_pair_confidence, num_tokens)

        # B + C: representation -> scalar/vector latents -> vector output head
        velocity, hidden = self.network(**inputs, return_hidden=True)
        aux = {}

        # D: atom vectors + residue features -> gated velocity and gate supervision
        if self.use_mobility_v1:
            velocity, aux = self.apply_residue_mobility_gate(
                velocity, hidden, inputs["pos_source"], z, inputs["batch"], atom_plddt,
                atom_mobility_attr, atom_to_token_idx, num_tokens,
                geometry_bond_index, ideal_bond_length, clash_exclusion_index)

        # E: velocity [N,3], optional auxiliary residue outputs
        return (velocity, aux) if return_aux else velocity


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
