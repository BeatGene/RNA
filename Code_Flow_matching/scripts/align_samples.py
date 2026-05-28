"""
对 sample_10_files.txt 中列出的 .pt 文件做 Kabsch 对齐：
将 pos_pred 刚性对齐到 pos，然后覆写 pos_pred 和 edge_index。

用法 (在服务器上):
  python scripts/align_samples.py \
      --sample_list config/sample_10_files.txt \
      --output_dir /remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA_10sample
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# 导入 Kabsch 对齐函数
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etflow.models.utils import find_rigid_alignment


def native_radius_graph(x, r):
    """原生 PyTorch 实现的 radius_graph，同 pt.py 里的逻辑"""
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(float("inf"))
    mask = dist < r
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


def align_and_save(pt_path, output_dir):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    pos = data["pos"]  # [N, 3] 真实坐标
    pos_pred = data["pos_pred"]  # [N, 3] Protenix 预测坐标

    # Kabsch 对齐: pos_pred → pos
    R, t = find_rigid_alignment(pos_pred, pos)
    pos_pred_aligned = (R @ pos_pred.T).T + t

    # 确认对齐效果
    diff_before = torch.norm(pos - pos_pred, dim=-1).mean().item()
    diff_after = torch.norm(pos - pos_pred_aligned, dim=-1).mean().item()
    print(f"  mean per-atom diff: {diff_before:.1f} -> {diff_after:.1f} Å")

    # 重新计算 edge_index（基于对齐后的 pos_pred）
    edge_index = native_radius_graph(pos_pred_aligned, r=4.5)

    # 构建新的 data dict
    new_data = {
        "pos": pos,
        "pos_pred": pos_pred_aligned,
        "atomic_numbers": data["atomic_numbers"],
        "sequence": data.get("sequence", ""),
        "edge_index": edge_index,
        "pdb_id": data.get("pdb_id", "unknown"),
        "seed": data.get("seed", -1),
        "pred_filename": data.get("pred_filename", ""),
        "alignment_ratio": data.get("alignment_ratio", 1.0),
    }

    # 保存
    save_path = Path(output_dir) / Path(pt_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_data, save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_list", type=str, required=True,
                        help="Path to sample_10_files.txt (one .pt path per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for aligned .pt files")
    args = parser.parse_args()

    with open(args.sample_list, "r") as f:
        pt_files = [line.strip() for line in f if line.strip()]

    print(f"Aligning {len(pt_files)} files...")
    aligned_paths = []
    for pt_path in tqdm(pt_files):
        if not os.path.exists(pt_path):
            print(f"  SKIP (not found): {pt_path}")
            continue
        print(f"\n[{Path(pt_path).stem}]")
        saved = align_and_save(pt_path, args.output_dir)
        aligned_paths.append(str(saved))

    # 写入新的 sample list
    new_list_path = Path(args.output_dir) / "sample_10_files.txt"
    with open(new_list_path, "w") as f:
        for p in aligned_paths:
            f.write(f"{p}\n")

    print(f"\nDone! {len(aligned_paths)} files saved to {args.output_dir}")
    print(f"New sample list: {new_list_path}")


if __name__ == "__main__":
    main()
