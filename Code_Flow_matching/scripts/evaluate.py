import argparse
import csv
import os
import sys
import traceback
import statistics
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch_geometric.utils import unbatch

from etflow.data.datamodule import BaseDataModule
from etflow.models.model import BaseFlow
from etflow.models.utils import find_rigid_alignment


def compute_rmsd(pos_a, pos_b):
    """Kabsch-aligned RMSD between two coordinate sets (in Angstroms)."""
    R, t = find_rigid_alignment(pos_a, pos_b)
    aligned = (R @ pos_a.T).T + t
    return torch.sqrt(((aligned - pos_b) ** 2).sum(dim=-1).mean()).item()


@torch.no_grad()
def evaluate_rmsd(
        config_path: str,
        ckpt_path: str,
        data_dir: str,
        split: str = "test",
        sample_files: str = None,
        num_samples: int = None,
        n_timesteps: int = 50,
        batch_size: int = 16,
        num_workers: int = 4,
        device: str = "cuda",
        csv_path: str = "evaluation/rmsd_summary.csv",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. 解析配置并加载模型 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n[1/3] Building model and loading checkpoint...")
    model = BaseFlow(**cfg["model_args"]).to(device)
    model.eval()

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Check if 'state_dict' is inside a PyTorch Lightning checkpoint wrapper
    state_dict = raw.get("state_dict", raw)

    # Handle the case where the keys have an extra "model." prefix
    # (common in Lightning if you wrapped your model)
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    del raw

    # ---- 2. 加载 DataLoader ----
    # 覆盖 yaml 里的 data_dir，如果传入了自定义路径
    final_data_dir = data_dir or cfg.get("datamodule_args", {}).get("data_dir")
    print(f"\n[2/3] Loading Dataloader from: {final_data_dir} (Split: {split})")

    dm = BaseDataModule(
        data_dir=Path(final_data_dir) if final_data_dir else None,
        dataloader_args={"batch_size": batch_size, "num_workers": num_workers},
        sample_files=sample_files,
    )

    # Load the requested split
    dm.setup(stage="fit" if split in ("train", "val") else "test")

    if sample_files:
        dataset = dm.train_dataset
    elif split == "train":
        dataset = dm.train_dataset
    elif split == "val":
        dataset = dm.val_dataset
    else:
        dataset = dm.test_dataset

    # --- Subsample if requested ---
    if num_samples is not None and num_samples < len(dataset):
        print(f"  Randomly selecting {num_samples} samples from {len(dataset)} total {split} samples...")
        # Get random indices
        indices = random.sample(range(len(dataset)), num_samples)
        # Subset the dataset
        dataset = torch.utils.data.Subset(dataset, indices)

    # Create the dataloader manually since we might be using a subset
    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"  Evaluating on {len(dataset)} samples. Batch size: {batch_size}")

    # ---- 3. 批量推理与 RMSD 计算 ----
    print(f"\n[3/3] Running ODE sampling (timesteps={n_timesteps})...")
    results = []
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)

        try:
            # 批量前向传播 (ODE 积分)
            pos_refined = model.sample(
                z=batch.z,
                pos_pred=batch.pos_pred,
                edge_index=batch.edge_index,
                batch=batch.batch,
                n_timesteps=n_timesteps,
            )

            # 使用 unbatch 将连在一起的超大图拆分成单个分子的坐标列表
            true_list = unbatch(batch.pos, batch.batch)
            pred_list = unbatch(batch.pos_pred, batch.batch)
            refined_list = unbatch(pos_refined, batch.batch)

            # 从 batch 中提取 pdb_id
            pdb_ids = batch.pdb_id if hasattr(batch, 'pdb_id') else [f"unknown_{i}" for i in range(len(true_list))]

            # 逐个分子计算 RMSD
            for i in range(len(true_list)):
                rmsd_before = compute_rmsd(pred_list[i], true_list[i])
                rmsd_after = compute_rmsd(refined_list[i], true_list[i])
                imp = rmsd_before - rmsd_after
                imp_pct = (imp / rmsd_before * 100) if rmsd_before > 0 else 0.0

                results.append({
                    "pdb_id": pdb_ids[i],
                    "num_atoms": true_list[i].size(0),
                    "rmsd_before": round(rmsd_before, 4),
                    "rmsd_after": round(rmsd_after, 4),
                    "improvement": round(imp, 4),
                    "improvement_pct": round(imp_pct, 2),
                })

        except Exception:
            tqdm.write(f"[Batch Error]: {traceback.format_exc().strip().split(chr(10))[-1]}")

    if not results:
        print("\nAll samples failed.")
        return

    # ---- 4. 统计与输出 ----
    def _stats(arr):
        return {
            "mean": sum(arr) / len(arr),
            "median": statistics.median(arr),
            "min": min(arr),
            "max": max(arr)
        }

    r_before = [r["rmsd_before"] for r in results]
    r_after = [r["rmsd_after"] for r in results]
    r_imp = [r["improvement"] for r in results]
    n_better = sum(1 for v in r_imp if v > 0)

    print("\n" + "=" * 50)
    print(" 📊 EVALUATION SUMMARY (RMSD)")
    print("=" * 50)
    print(f"Total evaluated : {len(results)} ({split} set)")
    print(
        f"RMSD Before     : Mean {sum(r_before) / len(r_before):.4f} Å  |  Median {statistics.median(r_before):.4f} Å")
    print(f"RMSD After      : Mean {sum(r_after) / len(r_after):.4f} Å  |  Median {statistics.median(r_after):.4f} Å")
    print("-" * 50)
    print(f"Improved Ratio  : {n_better}/{len(results)} ({n_better / len(results) * 100:.1f}%)")
    print("=" * 50)

    # 写入 CSV
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\n✅ Detailed results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RMSD for RNA Flow Matching model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")

    # 新增参数
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--sample_files", "-sf", type=str, default=None,
                        help="Path to sample list file (one .pt per line). Overrides split-based loading.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample this many molecules. If None, evaluate all.")

    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--n_timesteps", type=int, default=50, help="ODE steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv_path", type=str, default="evaluation/rmsd_summary.csv")

    args = parser.parse_args()

    evaluate_rmsd(
        config_path=args.config,
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        split=args.split,
        sample_files=args.sample_files,
        num_samples=args.num_samples,
        n_timesteps=args.n_timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        csv_path=args.csv_path,
    )