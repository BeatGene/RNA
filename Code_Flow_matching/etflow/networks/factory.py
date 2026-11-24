"""Independent representation/head selection. Optional e3nn is imported lazily."""
from torch import nn
from etflow.commons.reductions import center_by_graph
from etflow.networks.torchmd_net import TorchMDDynamics
from etflow.networks.torchmd_net.modules import EquivariantVectorOutput
from etflow.networks.heads.linear import LinearVectorHead


class ModularDynamics(nn.Module):
    def __init__(self, representation, head, head_type):
        super().__init__()
        self.representation_model = representation
        self.output_model = head
        self.head_type = head_type

    def forward(self, return_hidden=False, **inputs):
        result = self.representation_model(**inputs)
        x, vector, z, pos, batch = result[:5]
        if self.head_type == "abrefine":
            velocity = self.output_model(result[5], z)
        elif self.head_type == "gated":
            _, velocity = self.output_model.pre_reduce(x, vector, z, pos, batch)
        else:
            velocity = self.output_model(x, vector)
        velocity = center_by_graph(velocity, batch=batch)
        return (velocity, x) if return_hidden else velocity


def build_network(network_type, representation_type, output_head_type, representation_args=None, **kwargs):
    if network_type != "TorchMDDynamics":
        raise ValueError("Use representation_type to select the backend; network_type must be TorchMDDynamics")
    if output_head_type not in {"gated", "linear", "abrefine"}:
        raise ValueError(f"Unknown output_head_type: {output_head_type}")
    if representation_type == "torchmd":
        if representation_args:
            raise ValueError("representation_args is only supported for abrefine")
        if output_head_type == "abrefine":
            raise ValueError("The ABRefine attention head requires spherical ABRefine latents")
        network = TorchMDDynamics(**kwargs)
        if output_head_type == "gated":
            return network  # preserve TorchMD state_dict names for existing checkpoints
        return ModularDynamics(network.representation_model,
                               LinearVectorHead(kwargs["hidden_channels"]), "linear")


    if representation_type != "abrefine":
        raise ValueError(f"Unknown representation_type: {representation_type}")
    try:
        from etflow.networks.representations.abrefine import ABRefineRepresentation
        from etflow.networks.heads.abrefine import ABRefineVectorHead
    except ImportError as error:
        raise ImportError("ABRefine backend requires e3nn (tested with 0.5.1)") from error
    names = ("hidden_channels", "num_layers", "num_rbf", "num_heads", "cutoff_upper",
             "max_z", "node_attr_dim", "edge_attr_dim", "source_conditioning")
    representation = ABRefineRepresentation(**{k: kwargs[k] for k in names}, **(representation_args or {}))
    attention = representation.core.equivariant_block
    del representation.core.equivariant_block  # head parameters are owned only by output_model
    if output_head_type == "abrefine":
        head = ABRefineVectorHead(attention)
    elif output_head_type == "linear":
        head = LinearVectorHead(kwargs["hidden_channels"])
    else:
        head = EquivariantVectorOutput(kwargs["hidden_channels"], activation=kwargs["activation"],
                                       layer_norm=kwargs["output_layer_norm"])
    return ModularDynamics(representation, head, output_head_type)
