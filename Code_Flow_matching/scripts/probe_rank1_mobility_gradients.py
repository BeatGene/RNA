"""Read-only checkpoint probe: actual mobility gates, raw outputs and loss gradients.

Run on the laboratory server. Uses generic_step's actual losses and weights;
no optimizer step is performed. Gradients diagnose this checkpoint on these
samples, not causal effects of removing a loss during training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.data import Batch

import evaluate_refinement as base
from etflow.data.dataset import EuclideanDataset
from etflow.models.utils import center_of_mass


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def norm(tensor):
    return 0.0 if tensor is None else float(tensor.float().norm())


def get_target_residue_errors(model, batch, aux):
    return model.residue_rms_error(
        prediction=center_of_mass(batch.pos_pred, batch=batch.batch),
        target=center_of_mass(batch.pos, batch=batch.batch),
        global_residue_index=aux["global_residue_index"],
        total_residues=aux["total_residues"],
        atom_mask=batch.target_mask,
    )


def gate_numerical_diagnostics(raw, velocity, aux):
    # Each probe batch contains exactly one graph. This FP32 mean deliberately
    # bypasses the production scatter reduction to check the RMS contraction.
    raw32 = raw.detach().float()
    velocity32 = velocity.detach().float()
    weighted32 = aux["mobility_atom"].detach().float() * raw32
    reference32 = weighted32 - weighted32.mean(dim=0, keepdim=True)
    rms = lambda value: float(value.square().sum(dim=-1).mean().sqrt())
    return {
        "raw_dtype": str(raw.dtype), "gate_dtype": str(aux["mobility_atom"].dtype),
        "velocity_dtype": str(velocity.dtype),
        "fp32_reference_gated_rms_a": rms(reference32),
        "production_vs_fp32_gate_rms_difference_a": rms(velocity32 - reference32),
        "raw_centroid_norm_a": float(raw32.mean(dim=0).norm()),
        "production_velocity_centroid_norm_a": float(velocity32.mean(dim=0).norm()),
    }


def main():
    root = Path.home() / "Code_Flow_matching"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/RNA_train_residual_v3.yaml")
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt")
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Data_PT_V2")
    parser.add_argument("--pdb", nargs="+", default=["8JHP", "9IOS", "7POF", "7ZJ4", "9C6J"])
    parser.add_argument("--output", type=Path, default=root / "evaluation/rank1_mobility_probe")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--parameter-gradients", action="store_true", help="also compare weighted parameter gradient norms/cosines")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a fresh --output: {args.output}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = base.load_model(args.config, args.checkpoint, device)
    if model.training_objective != "residual" or not model.use_mobility_v1:
        raise ValueError("requires residual objective with mobility")
    weights = {
        "flow_matching_loss": 1.0, "bond_loss": model.bond_loss_weight,
        "clash_loss": model.clash_loss_weight, "plane_loss": model.plane_loss_weight,
        "mobility_loss": model.mobility_gate_loss_weight, "no_regret_loss": model.no_regret_loss_weight,
        "protect_loss": model.protect_loss_weight, "velocity_budget_loss": model.velocity_budget_loss_weight,
    }
    requested = {pdb.upper() for pdb in args.pdb}
    found, gate_rows, gradient_rows = set(), [], []
    parameters = tuple(p for p in model.parameters() if p.requires_grad) if args.parameter_gradients else ()
    for split in ("val", "test"):
        dataset = EuclideanDataset(args.data_root, split=split, include_metadata=True)
        for index, path in enumerate(dataset.data_files):
            pdb = path.relative_to(args.data_root / split).parts[0].upper()
            if pdb not in requested:
                continue
            if pdb in found:
                raise ValueError(f"multiple PTs for requested PDB: {pdb}")
            found.add(pdb)
            batch = Batch.from_data_list([dataset.get(index)]).to(device)
            logged, captured = {}, {}
            original_forward = model.forward

            def capture_forward(*a, **kw):
                result = original_forward(*a, **kw)
                captured["velocity"], captured["aux"] = result
                return result

            model.forward = capture_forward
            model.log_helper = lambda key, value, batch_size: logged.__setitem__(key.split("/", 1)[1], value)
            with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                     enabled=device.type == "cuda" and args.precision == "bf16"):
                model.generic_step(batch, 0, "val")
            model.forward = original_forward
            aux = captured["aux"]
            raw, gates = aux["raw_velocity"], aux["mobility_residue"]
            velocity = captured["velocity"]
            target_error, valid = get_target_residue_errors(model, batch, aux)
            targets = ((target_error - model.mobility_good_error) /
                       (model.mobility_move_error - model.mobility_good_error)).clamp(0, 1)
            gate_values = gates.detach().float().view(-1)
            raw_rms = float(raw.detach().float().square().sum(-1).mean().sqrt())
            gated_rms = float(velocity.detach().float().square().sum(-1).mean().sqrt())
            gate_rows.append({
                "split": split, "pdb_id": pdb, "raw_velocity_rms_a": raw_rms,
                "gated_velocity_rms_a": gated_rms, "output_to_raw_ratio": gated_rms / max(raw_rms, 1e-20),
                "gate_mean": float(gate_values.mean()), "gate_median": float(gate_values.median()),
                "gate_min": float(gate_values.min()), "gate_max": float(gate_values.max()),
                "gate_fraction_lt_0.01": float((gate_values < 0.01).float().mean()),
                "gate_target_mean": float(targets[valid].mean()),
                "protect_fraction": float((target_error[valid] < model.protect_error_threshold).float().mean()),
                **gate_numerical_diagnostics(raw, velocity, aux),
            })
            baseline_raw = None
            baseline_parameters = None
            for name, weight in weights.items():
                grads = torch.autograd.grad(weight * logged[name], (raw, gates, *parameters),
                                            allow_unused=True, retain_graph=True)
                raw_grad, gate_grad = grads[:2]
                if name == "flow_matching_loss":
                    baseline_raw = raw_grad.detach().float() if raw_grad is not None else None
                    baseline_parameters = [None if g is None else g.detach().float() for g in grads[2:]]
                raw_cosine = ""
                if raw_grad is not None and baseline_raw is not None and norm(raw_grad) * norm(baseline_raw) > 1e-20:
                    raw_cosine = float((raw_grad.float() * baseline_raw).sum()) / (norm(raw_grad) * norm(baseline_raw))
                parameter_norm, parameter_cosine = "", ""
                if parameters:
                    square = sum(float(g.float().square().sum()) for g in grads[2:] if g is not None)
                    parameter_norm = square ** 0.5
                    base_square = sum(float(g.square().sum()) for g in baseline_parameters if g is not None)
                    dot = sum(float(g.float().mul(b).sum()) for g, b in zip(grads[2:], baseline_parameters)
                              if g is not None and b is not None)
                    if parameter_norm * base_square ** 0.5 > 1e-20:
                        parameter_cosine = dot / (parameter_norm * base_square ** 0.5)
                gradient_rows.append({
                    "split": split, "pdb_id": pdb, "loss": name, "weight": weight,
                    "unweighted_value": float(logged[name].detach()),
                    "weighted_value": float(weight * logged[name].detach()),
                    "weighted_raw_output_gradient_norm": norm(raw_grad),
                    "weighted_gate_gradient_norm": norm(gate_grad), "raw_gradient_cosine_vs_coordinate": raw_cosine,
                    "weighted_parameter_gradient_norm": parameter_norm,
                    "parameter_gradient_cosine_vs_coordinate": parameter_cosine,
                })
            print("PROBED", pdb, gate_rows[-1], flush=True)
            del batch, logged, captured, aux, raw, gates, velocity, grads, baseline_raw, baseline_parameters
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if found != requested:
        raise ValueError(f"requested PDBs missing: {sorted(requested - found)}")
    args.output.mkdir(parents=True)
    write_rows(args.output / "gates.tsv", gate_rows)
    write_rows(args.output / "loss_gradients.tsv", gradient_rows)
    print("MOBILITY_PROBE_COMPLETE", args.output)


if __name__ == "__main__":
    main()
