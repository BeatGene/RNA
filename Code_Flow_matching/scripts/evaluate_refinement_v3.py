"""Evaluate V3 RNA refinement with optional per-candidate physical geometry.

The V2 evaluator remains unchanged for reproducibility.  This entry point
shares its RMSD definitions and aggregation code, and adds physical losses
computed with the exact functions used during V3 training.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch_geometric.data import Batch

import evaluate_refinement as base
from etflow.data.dataset import EuclideanDataset
from etflow.data.constants import ATOM_NAME_TO_ID
from etflow.models.loss import bond_length_loss, steric_clash_loss
from etflow.models.utils import center_of_mass


PHYSICS_FIELDS = [
    "input_bond_loss_a2", "refined_bond_loss_a2",
    "input_clash_loss_a2", "refined_clash_loss_a2",
    "input_plane_loss_a2", "refined_plane_loss_a2",
]
BACKBONE_ATOM_NAMES = (
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'",
)
BACKBONE_ATOM_IDS = tuple(ATOM_NAME_TO_ID[name] for name in BACKBONE_ATOM_NAMES)
BACKBONE_FIELDS = [
    "observed_backbone_atom_count",
    "input_backbone_aligned_rmsd",
    "refined_backbone_aligned_rmsd",
    "backbone_aligned_improvement",
]
SAMPLE_FIELDS = base.SAMPLE_FIELDS + BACKBONE_FIELDS + PHYSICS_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--length-cutoffs", type=int, nargs=3, default=(50, 100, 200))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--physics", action="store_true", help="Calculate all three V3 physical losses before and after refinement")
    return parser.parse_args()


def plane_loss_vectorized(
    pos: torch.Tensor, residue_index: torch.Tensor,
    atom_name_id: torch.Tensor, base_atom_name_ids: torch.Tensor,
) -> torch.Tensor:
    """The base_plane_loss formula, batched over residues for evaluation speed."""
    mask = torch.isin(atom_name_id.view(-1), base_atom_name_ids)
    if not bool(mask.any()):
        return pos.sum() * 0.0
    positions = pos[mask].float()
    _, group, count = torch.unique(
        residue_index[mask].view(-1), sorted=True,
        return_inverse=True, return_counts=True,
    )
    valid = count >= 4
    if not bool(valid.any()):
        return pos.sum() * 0.0
    means = positions.new_zeros((len(count), 3))
    means.index_add_(0, group, positions)
    means = means / count[:, None]
    centered = positions - means[group]
    outer = (centered[:, :, None] * centered[:, None, :]).reshape(-1, 9)
    covariance = positions.new_zeros((len(count), 9))
    covariance.index_add_(0, group, outer)
    covariance = (covariance / count[:, None]).reshape(-1, 3, 3)
    return torch.linalg.eigvalsh(covariance[valid])[:, 0].clamp_min(0).mean()


def physical_losses(pos: torch.Tensor, graph, model: torch.nn.Module) -> tuple[float, float, float]:
    """Return unweighted training bond/clash losses and equivalent plane loss."""
    device = pos.device
    geometry_bond_index = graph.geometry_bond_index.to(device)
    ideal_bond_length = graph.ideal_bond_length.to(device)
    clash_exclusion_index = graph.clash_exclusion_index.to(device)
    atom_count = pos.shape[0]
    graph_batch = torch.zeros(atom_count, dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        pos = pos.float()
        bond = bond_length_loss(pos, geometry_bond_index, ideal_bond_length)
        clash = steric_clash_loss(
            pos, graph.atomic_numbers.to(device), graph_batch,
            geometry_bond_index, model.vdw_radius_table,
            clash_exclusion_index=clash_exclusion_index,
        )
        plane = plane_loss_vectorized(
            pos, graph.residue_index.to(device), graph.atom_name_id.to(device),
            model.base_atom_name_ids,
        )
    return float(bond), float(clash), float(plane)


def evaluate_rank(args: argparse.Namespace, rank: int, world_size: int, local_rank: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("Multi-process evaluation requires CUDA")
        device = torch.device("cpu")

    dataset = EuclideanDataset(args.data_dir, split=args.split, include_metadata=True)
    indices = list(range(len(dataset)))
    if args.limit is not None:
        indices = indices[:args.limit]
    indices = indices[rank::world_size]
    model = base.load_model(args.config, args.checkpoint, device)
    backbone_ids = torch.tensor(BACKBONE_ATOM_IDS, dtype=torch.long, device=device)
    shard_path = args.output_dir / f"samples.rank{rank:03d}.tsv"
    started = time.perf_counter()
    processed = 0

    with shard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, delimiter="\t")
        writer.writeheader()
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset:offset + args.batch_size]
            data_list = [dataset.get(index) for index in selected]
            batch = Batch.from_data_list(data_list).to(device)
            graph_count = len(data_list)
            amp_enabled = device.type == "cuda" and args.precision == "bf16"
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled,
            ):
                prediction = model.sample(
                    z=batch.atomic_numbers,
                    pos_pred=batch.pos_pred,
                    bond_index=batch.edge_index,
                    batch=batch.batch,
                    node_attr=batch.node_attr,
                    edge_attr=batch.edge_attr,
                    atom_plddt=batch.atom_plddt,
                    atom_to_token_idx=batch.atom_to_token_idx,
                    token_pair_confidence=batch.token_pair_confidence,
                    num_tokens=batch.num_tokens,
                    atom_mobility_attr=batch.atom_mobility_attr,
                    geometry_bond_index=batch.geometry_bond_index,
                    ideal_bond_length=batch.ideal_bond_length,
                    clash_exclusion_index=batch.clash_exclusion_index,
                    n_timesteps=args.num_timesteps,
                )

            source_centered = center_of_mass(batch.pos_pred, batch=batch.batch)
            target_centered = center_of_mass(batch.pos, batch=batch.batch)
            prediction_centered = center_of_mass(prediction, batch=batch.batch)
            metadata = {
                "sample_ids": base._as_list(batch.sample_id, graph_count),
                "pdb_ids": base._as_list(batch.native_structure_id, graph_count),
                "native_chains": base._as_list(batch.native_chain_id, graph_count),
                "predicted_chains": base._as_list(batch.predicted_chain_id, graph_count),
                "sample_paths": base._as_list(batch.sample_path, graph_count),
                "seeds": base._as_list(batch.protenix_seed, graph_count),
                "sample_numbers": base._as_list(batch.protenix_sample, graph_count),
                "lengths": base._as_list(batch.sequence_length, graph_count),
                "stored_rmsds": base._as_list(batch.stored_input_rmsd, graph_count),
            }

            for graph_index, graph in enumerate(data_list):
                atom_selector = batch.batch == graph_index
                observed_selector = atom_selector & batch.target_mask
                backbone_selector = observed_selector & torch.isin(batch.atom_name_id, backbone_ids)
                source_observed = batch.pos_pred[observed_selector]
                target_observed = batch.pos[observed_selector]
                refined_observed = prediction[observed_selector]
                input_frame = base.rmsd(
                    source_centered[observed_selector], target_centered[observed_selector],
                )
                refined_frame = base.rmsd(
                    prediction_centered[observed_selector], target_centered[observed_selector],
                )
                input_aligned = base.aligned_rmsd(source_observed, target_observed)
                refined_aligned = base.aligned_rmsd(refined_observed, target_observed)
                improvement = input_aligned - refined_aligned
                input_backbone = base.aligned_rmsd(
                    batch.pos_pred[backbone_selector], batch.pos[backbone_selector],
                )
                refined_backbone = base.aligned_rmsd(
                    prediction[backbone_selector], batch.pos[backbone_selector],
                )
                atom_count = int(atom_selector.sum())
                observed_count = int(observed_selector.sum())
                row = {
                    "split": args.split,
                    "sample_id": metadata["sample_ids"][graph_index],
                    "pdb_id": str(metadata["pdb_ids"][graph_index]).upper(),
                    "native_chain_id": metadata["native_chains"][graph_index],
                    "predicted_chain_id": metadata["predicted_chains"][graph_index],
                    "protenix_seed": int(metadata["seeds"][graph_index]),
                    "protenix_sample": int(metadata["sample_numbers"][graph_index]),
                    "length": int(metadata["lengths"][graph_index]),
                    "atom_count": atom_count,
                    "observed_atom_count": observed_count,
                    "observed_atom_fraction": observed_count / max(atom_count, 1),
                    "mean_plddt": float(batch.atom_plddt[atom_selector].float().mean()),
                    "input_frame_rmsd": input_frame,
                    "refined_frame_rmsd": refined_frame,
                    "frame_improvement": input_frame - refined_frame,
                    "input_aligned_rmsd": input_aligned,
                    "refined_aligned_rmsd": refined_aligned,
                    "aligned_improvement": improvement,
                    "stored_input_rmsd": float(metadata["stored_rmsds"][graph_index]),
                    "observed_backbone_atom_count": int(backbone_selector.sum()),
                    "input_backbone_aligned_rmsd": input_backbone,
                    "refined_backbone_aligned_rmsd": refined_backbone,
                    "backbone_aligned_improvement": input_backbone - refined_backbone,
                    "improved": int(improvement > args.tie_tolerance),
                    "worsened": int(improvement < -args.tie_tolerance),
                    "sample_path": metadata["sample_paths"][graph_index],
                }
                if args.physics:
                    source = batch.pos_pred[atom_selector].float()
                    refined = prediction[atom_selector].float()
                    input_losses = physical_losses(source, graph, model)
                    refined_losses = physical_losses(refined, graph, model)
                    for name, before, after in zip(
                        ("bond", "clash", "plane"), input_losses, refined_losses,
                    ):
                        row[f"input_{name}_loss_a2"] = before
                        row[f"refined_{name}_loss_a2"] = after
                else:
                    row.update({field: float("nan") for field in PHYSICS_FIELDS})
                writer.writerow({key: base.fmt(row[key]) for key in SAMPLE_FIELDS})

            processed += graph_count
            if rank == 0 and args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rank=0 processed={processed}/{len(indices)} "
                    f"samples_per_second={processed / max(elapsed, 1e-9):.3f}",
                    flush=True,
                )


def aggregate_outputs(args: argparse.Namespace, world_size: int) -> None:
    # Preserve all V2-style files and definitions.  Each shard has extra
    # columns; the V2 reader ignores those while computing its summaries.
    base.aggregate_outputs(args, world_size)
    rows: list[dict] = []
    for rank in range(world_size):
        with (args.output_dir / f"samples.rank{rank:03d}.tsv").open(
            "r", encoding="utf-8", newline="",
        ) as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    rows.sort(key=lambda row: (
        row["pdb_id"], row["native_chain_id"], int(row["protenix_seed"]),
        int(row["protenix_sample"]), row["sample_id"],
    ))
    with (args.output_dir / "samples.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_timesteps <= 0:
        raise SystemExit("batch-size and num-timesteps must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    if sorted(args.length_cutoffs) != list(args.length_cutoffs):
        raise SystemExit("length cutoffs must be increasing")
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"file does not exist: {path}")
    if not (args.data_dir / args.split).is_dir():
        raise SystemExit(f"split directory does not exist: {args.data_dir / args.split}")
    rank, world_size, local_rank = base.distributed_context()
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit(f"output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    base.barrier(world_size)
    evaluate_rank(args, rank, world_size, local_rank)
    base.barrier(world_size)
    if rank == 0:
        aggregate_outputs(args, world_size)
        print(f"EVALUATION_COMPLETE output_dir={args.output_dir}", flush=True)
    base.barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
