"""Evaluate a refinement checkpoint at candidate and PDB-chain level.

The 200 Protenix candidates belonging to one experimental RNA are correlated
replicates.  This script therefore writes both candidate-level results and a
PDB-chain macro summary; scientific conclusions should use the latter as the
primary unit of analysis.

The script can run directly on one GPU or under ``torchrun``.  Multi-process
execution uses one independent model per GPU and Gloo only for synchronization;
model inference itself does not require DDP/NCCL collectives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import yaml
from torch_geometric.data import Batch

from etflow.data.dataset import EuclideanDataset
from etflow.models.model import BaseFlow
from etflow.models.utils import center_of_mass


SAMPLE_FIELDS = [
    "split",
    "sample_id",
    "pdb_id",
    "native_chain_id",
    "predicted_chain_id",
    "protenix_seed",
    "protenix_sample",
    "length",
    "atom_count",
    "observed_atom_count",
    "observed_atom_fraction",
    "mean_plddt",
    "input_frame_rmsd",
    "refined_frame_rmsd",
    "frame_improvement",
    "input_aligned_rmsd",
    "refined_aligned_rmsd",
    "aligned_improvement",
    "stored_input_rmsd",
    "improved",
    "worsened",
    "sample_path",
]

CHAIN_FIELDS = [
    "pdb_id",
    "native_chain_id",
    "predicted_chain_id",
    "sample_count",
    "length",
    "mean_plddt",
    "input_rmsd_mean",
    "input_rmsd_median",
    "refined_rmsd_mean",
    "refined_rmsd_median",
    "improvement_mean",
    "improvement_median",
    "improvement_p10",
    "improvement_p90",
    "candidate_win_rate",
    "candidate_worsen_rate",
    "improvement_ge_0.5_rate",
    "damage_ge_0.5_rate",
    "outcome_by_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--tie-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--length-cutoffs",
        type=int,
        nargs=3,
        default=(50, 100, 200),
        metavar=("SHORT", "MEDIUM", "LONG"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # Evaluation performs no tensor collectives.  Gloo avoids depending on
        # NVLS/NCCL merely for two file-coordination barriers.
        dist.init_process_group(backend="gloo")
    return rank, world_size, local_rank


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> BaseFlow:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = BaseFlow(**config["model_args"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _as_list(value, count: int) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return float("nan")
    return float(torch.sqrt((a.float() - b.float()).square().sum(dim=-1).mean()))


def aligned_rmsd(mobile: torch.Tensor, target: torch.Tensor) -> float:
    """Proper-rotation Kabsch RMSD for one observed atom set."""
    if mobile.size(0) < 3:
        return float("nan")
    a = mobile.double()
    b = target.double()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(a.T @ b)
    v = vh.T
    rotation = v @ u.T
    if torch.det(rotation) < 0:
        v = v.clone()
        v[:, -1] *= -1
        rotation = v @ u.T
    aligned = (rotation @ a.T).T
    return float(torch.sqrt((aligned - b).square().sum(dim=-1).mean()))


def fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.8f}"
    return str(value)


def write_rows(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in fields})


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    integer_fields = {
        "protenix_seed", "protenix_sample", "length", "atom_count",
        "observed_atom_count", "improved", "worsened",
    }
    float_fields = {
        "observed_atom_fraction", "mean_plddt", "input_frame_rmsd",
        "refined_frame_rmsd", "frame_improvement", "input_aligned_rmsd",
        "refined_aligned_rmsd", "aligned_improvement", "stored_input_rmsd",
    }
    for row in rows:
        for key in integer_fields:
            row[key] = int(row[key])
        for key in float_fields:
            row[key] = float(row[key])
    return rows


def evaluate_rank(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    local_rank: int,
) -> Path:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("multi-process evaluation requires CUDA devices")
        device = torch.device("cpu")

    dataset = EuclideanDataset(
        args.data_dir, split=args.split, include_metadata=True
    )
    all_indices = list(range(len(dataset)))
    if args.limit is not None:
        all_indices = all_indices[: args.limit]
    indices = all_indices[rank::world_size]
    model = load_model(args.config, args.checkpoint, device)

    shard_path = args.output_dir / f"samples.rank{rank:03d}.tsv"
    processed = 0
    started = time.perf_counter()
    with shard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, delimiter="\t")
        writer.writeheader()
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset : offset + args.batch_size]
            data_list = [dataset.get(index) for index in selected]
            batch = Batch.from_data_list(data_list).to(device)
            graph_count = len(data_list)

            amp_enabled = device.type == "cuda" and args.precision == "bf16"
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
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

            sample_ids = _as_list(batch.sample_id, graph_count)
            pdb_ids = _as_list(batch.native_structure_id, graph_count)
            native_chains = _as_list(batch.native_chain_id, graph_count)
            predicted_chains = _as_list(batch.predicted_chain_id, graph_count)
            sample_paths = _as_list(batch.sample_path, graph_count)
            seeds = _as_list(batch.protenix_seed, graph_count)
            sample_numbers = _as_list(batch.protenix_sample, graph_count)
            lengths = _as_list(batch.sequence_length, graph_count)
            stored_rmsds = _as_list(batch.stored_input_rmsd, graph_count)

            for graph_index in range(graph_count):
                atom_selector = batch.batch == graph_index
                observed_selector = atom_selector & batch.target_mask
                source_observed = batch.pos_pred[observed_selector]
                target_observed = batch.pos[observed_selector]
                refined_observed = prediction[observed_selector]

                input_frame = rmsd(
                    source_centered[observed_selector],
                    target_centered[observed_selector],
                )
                refined_frame = rmsd(
                    prediction_centered[observed_selector],
                    target_centered[observed_selector],
                )
                input_aligned = aligned_rmsd(source_observed, target_observed)
                refined_aligned = aligned_rmsd(refined_observed, target_observed)
                improvement = input_aligned - refined_aligned
                atom_count = int(atom_selector.sum())
                observed_count = int(observed_selector.sum())
                mean_plddt = float(batch.atom_plddt[atom_selector].float().mean())

                row = {
                    "split": args.split,
                    "sample_id": sample_ids[graph_index],
                    "pdb_id": str(pdb_ids[graph_index]).upper(),
                    "native_chain_id": native_chains[graph_index],
                    "predicted_chain_id": predicted_chains[graph_index],
                    "protenix_seed": int(seeds[graph_index]),
                    "protenix_sample": int(sample_numbers[graph_index]),
                    "length": int(lengths[graph_index]),
                    "atom_count": atom_count,
                    "observed_atom_count": observed_count,
                    "observed_atom_fraction": observed_count / max(atom_count, 1),
                    "mean_plddt": mean_plddt,
                    "input_frame_rmsd": input_frame,
                    "refined_frame_rmsd": refined_frame,
                    "frame_improvement": input_frame - refined_frame,
                    "input_aligned_rmsd": input_aligned,
                    "refined_aligned_rmsd": refined_aligned,
                    "aligned_improvement": improvement,
                    "stored_input_rmsd": float(stored_rmsds[graph_index]),
                    "improved": int(improvement > args.tie_tolerance),
                    "worsened": int(improvement < -args.tie_tolerance),
                    "sample_path": sample_paths[graph_index],
                }
                writer.writerow({key: fmt(row[key]) for key in SAMPLE_FIELDS})

            processed += graph_count
            if rank == 0 and args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rank=0 processed={processed}/{len(indices)} "
                    f"samples_per_second={processed / max(elapsed, 1e-9):.3f}",
                    flush=True,
                )
    return shard_path


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank_value = (start + stop - 1) / 2 + 1
        for position in range(start, stop):
            ranks[ordered[position]] = rank_value
        start = stop
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator > 0 else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def bootstrap_mean_ci(
    values: Sequence[float], replicates: int, seed: int = 42
) -> tuple[float, float]:
    if not values or replicates <= 0:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    estimates = [
        mean([values[generator.randrange(len(values))] for _ in values])
        for _ in range(replicates)
    ]
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def aggregate_chains(rows: Sequence[dict], tie_tolerance: float) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pdb_id"], row["native_chain_id"])].append(row)

    results = []
    for (pdb_id, native_chain), group in grouped.items():
        improvements = [row["aligned_improvement"] for row in group]
        input_rmsds = [row["input_aligned_rmsd"] for row in group]
        refined_rmsds = [row["refined_aligned_rmsd"] for row in group]
        improvement_mean = mean(improvements)
        if improvement_mean > tie_tolerance:
            outcome = "improved"
        elif improvement_mean < -tie_tolerance:
            outcome = "worsened"
        else:
            outcome = "tie"
        results.append({
            "pdb_id": pdb_id,
            "native_chain_id": native_chain,
            "predicted_chain_id": group[0]["predicted_chain_id"],
            "sample_count": len(group),
            "length": round(median([row["length"] for row in group])),
            "mean_plddt": mean([row["mean_plddt"] for row in group]),
            "input_rmsd_mean": mean(input_rmsds),
            "input_rmsd_median": median(input_rmsds),
            "refined_rmsd_mean": mean(refined_rmsds),
            "refined_rmsd_median": median(refined_rmsds),
            "improvement_mean": improvement_mean,
            "improvement_median": median(improvements),
            "improvement_p10": quantile(improvements, 0.10),
            "improvement_p90": quantile(improvements, 0.90),
            "candidate_win_rate": mean([float(value > tie_tolerance) for value in improvements]),
            "candidate_worsen_rate": mean([float(value < -tie_tolerance) for value in improvements]),
            "improvement_ge_0.5_rate": mean([float(value >= 0.5) for value in improvements]),
            "damage_ge_0.5_rate": mean([float(value <= -0.5) for value in improvements]),
            "outcome_by_mean": outcome,
        })
    return sorted(results, key=lambda row: row["improvement_mean"], reverse=True)


def summarize_strata(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for label, group in groups.items():
        deltas = [row["improvement_mean"] for row in group]
        rows.append({
            "stratum": label,
            "pdb_chain_count": len(group),
            "candidate_count": sum(row["sample_count"] for row in group),
            "input_rmsd_mean": mean([row["input_rmsd_mean"] for row in group]),
            "refined_rmsd_mean": mean([row["refined_rmsd_mean"] for row in group]),
            "improvement_mean": mean(deltas),
            "improvement_median": median(deltas),
            "pdb_chain_win_rate": mean([float(delta > 0) for delta in deltas]),
            "candidate_win_rate_mean": mean([row["candidate_win_rate"] for row in group]),
        })
    return rows


def length_strata(chain_rows: Sequence[dict], cutoffs: Sequence[int]) -> list[dict]:
    first, second, third = cutoffs
    groups = {
        f"<= {first}": [],
        f"{first + 1}-{second}": [],
        f"{second + 1}-{third}": [],
        f"> {third}": [],
    }
    for row in chain_rows:
        length = row["length"]
        if length <= first:
            label = f"<= {first}"
        elif length <= second:
            label = f"{first + 1}-{second}"
        elif length <= third:
            label = f"{second + 1}-{third}"
        else:
            label = f"> {third}"
        groups[label].append(row)
    return summarize_strata(groups)


def rmsd_strata(chain_rows: Sequence[dict]) -> list[dict]:
    groups = {"<= 2": [], "2-5": [], "5-10": [], "> 10": []}
    for row in chain_rows:
        value = row["input_rmsd_mean"]
        if value <= 2:
            label = "<= 2"
        elif value <= 5:
            label = "2-5"
        elif value <= 10:
            label = "5-10"
        else:
            label = "> 10"
        groups[label].append(row)
    return summarize_strata(groups)


def markdown_table(rows: Sequence[dict], fields: Sequence[str]) -> list[str]:
    if not rows:
        return ["（无）"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    summary: dict,
    chain_rows: Sequence[dict],
    length_rows: Sequence[dict],
    rmsd_rows: Sequence[dict],
) -> None:
    top_fields = [
        "pdb_id", "native_chain_id", "length", "sample_count",
        "input_rmsd_mean", "refined_rmsd_mean", "improvement_mean",
        "candidate_win_rate",
    ]
    stratum_fields = [
        "stratum", "pdb_chain_count", "candidate_count", "input_rmsd_mean",
        "refined_rmsd_mean", "improvement_mean", "pdb_chain_win_rate",
        "candidate_win_rate_mean",
    ]
    best = list(chain_rows[:15])
    worst = list(reversed(chain_rows[-15:]))
    lines = [
        f"# {summary['split']} refinement evaluation",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- candidates: {summary['candidate_count']}",
        f"- PDB chains: {summary['pdb_chain_count']}",
        f"- candidate-level mean RMSD: {summary['candidate_input_rmsd_mean']:.4f} → {summary['candidate_refined_rmsd_mean']:.4f} Å",
        f"- candidate win/worsen rate: {summary['candidate_win_rate']:.2%} / {summary['candidate_worsen_rate']:.2%}",
        f"- PDB-chain macro mean improvement: {summary['pdb_chain_macro_improvement_mean']:.4f} Å "
        f"(95% bootstrap CI {summary['pdb_chain_macro_improvement_ci95'][0]:.4f}, "
        f"{summary['pdb_chain_macro_improvement_ci95'][1]:.4f})",
        f"- PDB-chain win/worsen rate: {summary['pdb_chain_win_rate']:.2%} / {summary['pdb_chain_worsen_rate']:.2%}",
        f"- Spearman(length, improvement): {summary['spearman_length_vs_improvement']:.4f}",
        f"- Spearman(input RMSD, improvement): {summary['spearman_input_rmsd_vs_improvement']:.4f}",
        f"- Spearman(mean pLDDT, improvement): {summary['spearman_plddt_vs_improvement']:.4f}",
        "",
        "正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。",
        "所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。",
        "",
        "## 按长度分层（PDB-chain macro）",
        "",
        *markdown_table(length_rows, stratum_fields),
        "",
        "## 按输入 RMSD 分层（PDB-chain macro）",
        "",
        *markdown_table(rmsd_rows, stratum_fields),
        "",
        "## 改善最大的 PDB chains",
        "",
        *markdown_table(best, top_fields),
        "",
        "## 变差最大的 PDB chains",
        "",
        *markdown_table(worst, top_fields),
        "",
        "## 解释限制",
        "",
        "- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。",
        "- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。",
        "- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_outputs(args: argparse.Namespace, world_size: int) -> None:
    rows = []
    for rank in range(world_size):
        rows.extend(read_rows(args.output_dir / f"samples.rank{rank:03d}.tsv"))
    rows.sort(key=lambda row: (
        row["pdb_id"], row["native_chain_id"], row["protenix_seed"],
        row["protenix_sample"], row["sample_id"],
    ))
    write_rows(args.output_dir / "samples.tsv", rows, SAMPLE_FIELDS)

    chain_rows = aggregate_chains(rows, args.tie_tolerance)
    write_rows(args.output_dir / "pdb_chain_summary.tsv", chain_rows, CHAIN_FIELDS)

    length_rows = length_strata(chain_rows, args.length_cutoffs)
    rmsd_rows = rmsd_strata(chain_rows)
    stratum_fields = list(length_rows[0].keys()) if length_rows else []
    write_rows(args.output_dir / "length_summary.tsv", length_rows, stratum_fields)
    write_rows(args.output_dir / "input_rmsd_summary.tsv", rmsd_rows, stratum_fields)

    candidate_improvements = [row["aligned_improvement"] for row in rows]
    chain_improvements = [row["improvement_mean"] for row in chain_rows]
    ci_low, ci_high = bootstrap_mean_ci(
        chain_improvements, args.bootstrap_replicates
    )
    stored_differences = [
        abs(row["input_aligned_rmsd"] - row["stored_input_rmsd"])
        for row in rows
        if math.isfinite(row["stored_input_rmsd"])
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "world_size": world_size,
        "candidate_count": len(rows),
        "pdb_chain_count": len(chain_rows),
        "candidate_input_rmsd_mean": mean([row["input_aligned_rmsd"] for row in rows]),
        "candidate_input_rmsd_median": median([row["input_aligned_rmsd"] for row in rows]),
        "candidate_refined_rmsd_mean": mean([row["refined_aligned_rmsd"] for row in rows]),
        "candidate_refined_rmsd_median": median([row["refined_aligned_rmsd"] for row in rows]),
        "candidate_improvement_mean": mean(candidate_improvements),
        "candidate_improvement_median": median(candidate_improvements),
        "candidate_win_rate": mean([float(row["improved"]) for row in rows]),
        "candidate_worsen_rate": mean([float(row["worsened"]) for row in rows]),
        "pdb_chain_macro_input_rmsd_mean": mean([row["input_rmsd_mean"] for row in chain_rows]),
        "pdb_chain_macro_refined_rmsd_mean": mean([row["refined_rmsd_mean"] for row in chain_rows]),
        "pdb_chain_macro_improvement_mean": mean(chain_improvements),
        "pdb_chain_macro_improvement_median": median(chain_improvements),
        "pdb_chain_macro_improvement_ci95": [ci_low, ci_high],
        "pdb_chain_win_rate": mean([float(value > args.tie_tolerance) for value in chain_improvements]),
        "pdb_chain_worsen_rate": mean([float(value < -args.tie_tolerance) for value in chain_improvements]),
        "spearman_length_vs_improvement": spearman(
            [row["length"] for row in chain_rows], chain_improvements
        ),
        "spearman_input_rmsd_vs_improvement": spearman(
            [row["input_rmsd_mean"] for row in chain_rows], chain_improvements
        ),
        "spearman_plddt_vs_improvement": spearman(
            [row["mean_plddt"] for row in chain_rows], chain_improvements
        ),
        "max_abs_recomputed_vs_stored_input_rmsd": max(stored_differences, default=float("nan")),
        "primary_metric": "observed-atom proper-rotation Kabsch RMSD; improvement=input-refined",
        "inference_note": "Each .pt candidate is evaluated; PDB-chain macro is the primary independent-target summary.",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
    write_report(
        args.output_dir / "report.md", summary, chain_rows, length_rows, rmsd_rows
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if sorted(args.length_cutoffs) != list(args.length_cutoffs):
        raise SystemExit("--length-cutoffs must be increasing")
    if not args.config.is_file():
        raise SystemExit(f"config does not exist: {args.config}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if not (args.data_dir / args.split).is_dir():
        raise SystemExit(f"split directory does not exist: {args.data_dir / args.split}")

    rank, world_size, local_rank = distributed_context()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    evaluate_rank(args, rank, world_size, local_rank)
    barrier(world_size)
    if rank == 0:
        aggregate_outputs(args, world_size)
        print(f"EVALUATION_COMPLETE output_dir={args.output_dir}", flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
