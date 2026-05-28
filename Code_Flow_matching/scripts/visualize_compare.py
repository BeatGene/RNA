"""
随机挑选 RNA 结构，将 真实 / refine前 / refine后 三组坐标写为 PDB 文件，
并生成 PyMOL 脚本用于可视化对比。

用法:
  python scripts/visualize_compare.py \
      --config config/RNA_test.yaml \
      --ckpt logs/RNA_test_01/runs/2026-05-11_15-34-06/checkpoints/rna_refinement_v1/last.ckpt \
      --data_dir /path/to/RNA/Data/RNA \
      --num_samples 5 \
      --output_dir visualization_output
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from etflow.models.model import BaseFlow
from etflow.models.utils import find_rigid_alignment, center_of_mass

Z_TO_ELEMENT = {6: 'C', 7: 'N', 8: 'O', 15: 'P', 16: 'S', 12: 'MG', 19: 'K'}


def write_pdb(coords, atomic_numbers, output_path, pdb_id="model"):
    """将坐标写为 PDB 格式 (PyMOL 可读)"""
    with open(output_path, 'w') as f:
        f.write(f"MODEL     {pdb_id}\n")
        for i, (pos, z) in enumerate(zip(coords, atomic_numbers)):
            atom_num = i + 1
            element = Z_TO_ELEMENT.get(int(z.item()), 'X')
            atom_name = f"{element}{atom_num:<3}"[:4]
            residue_name = "  A"
            residue_num = atom_num
            x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
            f.write(
                f"ATOM  {atom_num:5d} {atom_name:4s} {residue_name:3s}  "
                f"{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
        f.write("ENDMDL\n")
    print(f"  -> saved: {output_path}")


@torch.no_grad()
def run_visualization(
    config_path: str,
    ckpt_path: str,
    data_dir: str,
    num_samples: int = 5,
    output_dir: str = "visualization_output",
    n_timesteps: int = 50,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. 加载模型 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Loading model...")
    model = BaseFlow(**cfg["model_args"]).to(device)
    model.eval()

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw.get("state_dict", raw)
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    del raw
    print("Model loaded.")

    # ---- 2. 收集所有 .pt 文件 ----
    data_root = Path(data_dir)
    all_pt_files = list(data_root.rglob("*.pt"))
    print(f"Found {len(all_pt_files)} .pt files in {data_dir}")

    if len(all_pt_files) == 0:
        print("ERROR: No .pt files found!")
        return

    # 随机挑选
    selected = random.sample(all_pt_files, min(num_samples, len(all_pt_files)))
    print(f"Randomly selected {len(selected)} files:")
    for f in selected:
        print(f"  - {f}")

    # ---- 3. 创建输出目录 ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- 会话级别的 PyMOL 脚本收集器 ----
    pml_lines = [
        "# PyMOL 可视化脚本",
        "# 用法: pymol visualization_output/compare.pml",
        "",
        "bg_color white",
        "set grid_mode, 0",
        f"cd {os.path.abspath(output_dir)}",
        "",
    ]
    colors = {"gt": "green", "pre": "red", "post": "blue"}
    labels = {"gt": "Ground Truth (crystal)", "pre": "Before refinement (Protenix)", "post": "After refinement (Flow Matching)"}

    all_pdb_ids = []

    # ---- 4. 逐个处理 ----
    for pt_path in tqdm(selected, desc="Processing"):
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        pdb_id = data.get('pdb_id', pt_path.stem)
        seed = data.get('seed', '??')

        pos_gt = data['pos']          # [N, 3] 真实坐标
        pos_pre = data['pos_pred']    # [N, 3] refine前 (Protenix)
        atomic_numbers = data['atomic_numbers']
        edge_index = data['edge_index']

        # 构造 batch=0 (单分子)
        batch = torch.zeros(pos_gt.size(0), dtype=torch.long, device=device)

        pos_gt_dev = pos_gt.to(device)
        pos_pre_dev = pos_pre.to(device)
        edge_index_dev = edge_index.to(device)
        z_dev = atomic_numbers.to(device)

        # 模型推理: pos_pre -> pos_post (refine后)
        print(f"\n  [{pdb_id}] Running ODE sampling ({n_timesteps} steps)...")
        pos_post = model.sample(
            z=z_dev,
            pos_pred=pos_pre_dev,
            edge_index=edge_index_dev,
            batch=batch,
            n_timesteps=n_timesteps,
        ).cpu()

        # ---- 5. 将 pre 和 post 都 Kabsch-align 到真实坐标 ----
        pos_gt_cpu = pos_gt.cpu()
        pos_pre_cpu = pos_pre.cpu()

        # Align pre -> gt
        R_pre, t_pre = find_rigid_alignment(pos_pre_cpu, pos_gt_cpu)
        pos_pre_aligned = (R_pre @ pos_pre_cpu.T).T + t_pre

        # Align post -> gt
        R_post, t_post = find_rigid_alignment(pos_post, pos_gt_cpu)
        pos_post_aligned = (R_post @ pos_post.T).T + t_post

        # ---- 6. 写 PDB 文件 ----
        safe_name = f"{pdb_id}_seed{seed}"
        gt_pdb = os.path.join(output_dir, f"{safe_name}_gt.pdb")
        pre_pdb = os.path.join(output_dir, f"{safe_name}_pre.pdb")
        post_pdb = os.path.join(output_dir, f"{safe_name}_post.pdb")

        write_pdb(pos_gt_cpu, atomic_numbers, gt_pdb, f"{pdb_id}_gt")
        write_pdb(pos_pre_aligned, atomic_numbers, pre_pdb, f"{pdb_id}_pre")
        write_pdb(pos_post_aligned, atomic_numbers, post_pdb, f"{pdb_id}_post")

        all_pdb_ids.append(safe_name)

        # 添加到 PyMOL 脚本
        pml_lines.append(f"# --- {pdb_id} (seed={seed}) ---")
        pml_lines.append(f"load {safe_name}_gt.pdb, {safe_name}_gt")
        pml_lines.append(f"load {safe_name}_pre.pdb, {safe_name}_pre")
        pml_lines.append(f"load {safe_name}_post.pdb, {safe_name}_post")
        pml_lines.append(f"color {colors['gt']}, {safe_name}_gt")
        pml_lines.append(f"color {colors['pre']}, {safe_name}_pre")
        pml_lines.append(f"color {colors['post']}, {safe_name}_post")
        pml_lines.append(f"show cartoon, {safe_name}_gt")
        pml_lines.append(f"show cartoon, {safe_name}_pre")
        pml_lines.append(f"show cartoon, {safe_name}_post")
        pml_lines.append("")

    # ---- 7. 写整体 PyMOL 脚本 ----
    pml_lines.append("# 每组单独对齐并在同一视图中展示")
    pml_lines.append("set cartoon_fancy_helices, 1")
    pml_lines.append("set cartoon_smooth_loops, 1")
    pml_lines.append("")

    for name in all_pdb_ids:
        pml_lines.append(f"# 对齐 {name}: pre -> gt, post -> gt")
        pml_lines.append(f"align {name}_pre, {name}_gt")
        pml_lines.append(f"align {name}_post, {name}_gt")
        pml_lines.append("")

    pml_lines.append("# 禁用所有对象，方便逐个查看")
    pml_lines.append(f"disable all")
    pml_lines.append("")

    # 逐个 group 展示
    for i, name in enumerate(all_pdb_ids):
        pml_lines.append(f"# Group {i+1}: {name}")
        pml_lines.append(f"group group_{i+1}, {name}_gt {name}_pre {name}_post")
        pml_lines.append(f"# 双击或 enable group_{i+1} 来查看这一组")

    pml_lines.append("")
    pml_lines.append("zoom all")
    pml_lines.append("")
    pml_lines.append("# 图例说明:")
    for key in ["gt", "pre", "post"]:
        pml_lines.append(f"#  {colors[key]:6s} = {labels[key]}")

    pml_path = os.path.join(output_dir, "compare.pml")
    with open(pml_path, 'w') as f:
        f.write('\n'.join(pml_lines))

    print(f"\n{'='*60}")
    print(f"Done! {len(selected)} structures processed.")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"PyMOL script:      {pml_path}")
    print(f"\nTo visualize, run:")
    print(f"  pymol {os.path.abspath(pml_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select RNA structures and generate PDBs + PyMOL script for comparison"
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing .pt files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random structures to pick")
    parser.add_argument("--output_dir", type=str, default="visualization_output", help="Output directory")
    parser.add_argument("--n_timesteps", type=int, default=50, help="ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_visualization(
        config_path=args.config,
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        n_timesteps=args.n_timesteps,
        device=args.device,
    )
