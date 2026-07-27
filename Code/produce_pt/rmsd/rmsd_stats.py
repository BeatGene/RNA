import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
SPLITS = ['train', 'val', 'test']
OUTPUT_CSV = "rmsd_all.csv"


def compute_rmsd(pt_path):
    """加载单个 .pt 文件，返回 rmsd 值和原子数。失败返回 None。"""
    try:
        data = torch.load(pt_path, weights_only=True)
        pos = data['pos']          # (N, 3)
        pos_pred = data['pos_pred']
        diff = pos_pred - pos
        rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()
        return rmsd, pos.shape[0]
    except Exception:
        return None


def extract_info(filename):
    """从文件名提取 PDB ID 和 seed。9cso_s45_9cso_sample_9.pt → ('9cso', '45')"""
    pdb_id = filename[:4]
    seed = filename.split('_s')[1].split('_')[0]
    return pdb_id, seed


def main():
    records = []

    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            continue

        pt_files = list(Path(split_dir).rglob("*.pt"))
        print(f"{split}: {len(pt_files)} 个文件")

        for pt_path in tqdm(pt_files, desc=split):
            result = compute_rmsd(str(pt_path))
            if result is None:
                continue
            rmsd, n_atoms = result
            pdb_id, seed = extract_info(os.path.basename(pt_path))
            records.append({
                'split': split,
                'pdb_id': pdb_id,
                'seed': seed,
                'n_atoms': n_atoms,
                'rmsd': round(rmsd, 4),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n保存至 {OUTPUT_CSV}，共 {len(df)} 条记录")

    # ---- 统计 ----
    rmsd = df['rmsd']
    print(f"\n{'='*50}")
    print(f"RMSD 分布统计 (Å):")
    print(f"  count : {len(rmsd)}")
    print(f"  mean  : {rmsd.mean():.4f}")
    print(f"  std   : {rmsd.std():.4f}")
    print(f"  min   : {rmsd.min():.4f}")
    print(f"  Q1    : {rmsd.quantile(0.25):.4f}")
    print(f"  median: {rmsd.quantile(0.50):.4f}")
    print(f"  Q3    : {rmsd.quantile(0.75):.4f}")
    print(f"  max   : {rmsd.max():.4f}")
    print(f"\n  RMSD < 1Å : {(rmsd < 1).mean():.2%}")
    print(f"  RMSD < 2Å : {(rmsd < 2).mean():.2%}")
    print(f"  RMSD < 5Å : {(rmsd < 5).mean():.2%}")

    # ---- 直方图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 全量
    axes[0].hist(np.clip(rmsd, 0, 20), bins=100, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(rmsd.median(), color='red', linestyle='--', label=f'Median: {rmsd.median():.2f} Å')
    axes[0].set_xlabel('RMSD (Å)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('All RMSD (clipped at 20 Å)')
    axes[0].legend()

    # 聚焦 0-10 Å
    axes[1].hist(np.clip(rmsd, 0, 10), bins=80, color='darkorange', edgecolor='white', alpha=0.85)
    axes[1].axvline(rmsd.median(), color='red', linestyle='--', label=f'Median: {rmsd.median():.2f} Å')
    axes[1].set_xlabel('RMSD (Å)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('RMSD 0-10 Å (zoomed)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('rmsd_distribution.png', dpi=200, bbox_inches='tight')
    print(f"图片保存至 rmsd_distribution.png")

    # ---- 按 split 分组 ----
    print(f"\n--- 按 split 分组 ---")
    for split in SPLITS:
        sub = df[df['split'] == split]['rmsd']
        if len(sub) > 0:
            print(f"  {split}: mean={sub.mean():.3f}, median={sub.median():.3f}, n={len(sub)}")

    # ---- RMSD 最大的前 20 ----
    print(f"\n--- RMSD 最大的 20 个 ---")
    top = df.nlargest(20, 'rmsd')
    print(top[['pdb_id', 'seed', 'split', 'rmsd']].to_string(index=False))


if __name__ == "__main__":
    main()
