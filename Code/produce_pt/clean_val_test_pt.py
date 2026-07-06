"""
清理 val/test 目录下的 .pt 文件：每个 PDB 只保留 ranking_score 最高的那一个。

逻辑：
1. 对每个 PDB，遍历4个 seed (42, 43, 44, 45)
2. 读取每个 seed 的 summary_confidence_sample_0.json，获取 ranking_score
3. 跨 seed 比较，只保留 ranking_score 最高的 seed 下的 sample_0.pt
4. 删除该 PDB 下所有其他 .pt 文件

用法：
  python clean_val_test_pt.py --dry_run    # 仅预览，不实际删除
  python clean_val_test_pt.py --execute     # 实际执行删除
"""

import os
import json
import argparse
import sys

PRED_BASE = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
RNA_BASE = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
SPLITS = ["val", "test"]
SEEDS = [42, 43, 44, 45]


def get_ranking_score(pdb_id, seed):
    """读取某个 seed 下 sample_0 的 ranking_score，失败返回 None"""
    json_path = os.path.join(
        PRED_BASE,
        f"pred_output_{pdb_id}_seed_{seed}",
        pdb_id,
        f"seed_{seed}",
        "predictions",
        f"{pdb_id}_summary_confidence_sample_0.json",
    )
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("ranking_score", None)
    except Exception:
        return None


def process_pdb(pdb_id, pt_dir, execute=False):
    """
    对一个 PDB 目录进行处理。
    返回 (kept_file, deleted_count, error_msg)
    """
    # 收集所有 .pt 文件
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    if not pt_files:
        return None, 0, "no .pt files"

    # 获取每个 seed 的 ranking_score
    seed_scores = {}
    for seed in SEEDS:
        score = get_ranking_score(pdb_id, seed)
        if score is not None:
            seed_scores[seed] = score

    if not seed_scores:
        return None, 0, "no summary_confidence JSON found for any seed"

    # 按 ranking_score 降序排列 seed，找到第一个 sample_0.pt 实际存在的
    sorted_seeds = sorted(seed_scores, key=seed_scores.get, reverse=True)
    best_seed = None
    best_pt = None
    for seed in sorted_seeds:
        candidate = f"{pdb_id}_s{seed}_{pdb_id}_sample_0.pt"
        if candidate in pt_files:
            best_seed = seed
            best_pt = candidate
            break

    if best_seed is None:
        return None, 0, f"no sample_0.pt found for any seed in {pt_dir} (seeds with JSON: {sorted_seeds})"

    # 统计
    to_delete = [f for f in pt_files if f != best_pt]

    print(f"  PDB: {pdb_id}")
    print(f"    各 seed ranking_score: { {s: f'{v:.4f}' for s, v in seed_scores.items()} }")
    if best_seed != sorted_seeds[0]:
        print(f"    ⚠ 最佳 seed={sorted_seeds[0]} 的 sample_0.pt 缺失，回退到 seed={best_seed}")
    print(f"    最佳: seed={best_seed}, ranking_score={seed_scores[best_seed]:.4f}")
    print(f"    保留: {best_pt}")
    print(f"    删除: {len(to_delete)} 个文件")

    if execute and to_delete:
        for fname in to_delete:
            fpath = os.path.join(pt_dir, fname)
            try:
                os.remove(fpath)
            except OSError as e:
                print(f"    删除失败: {fname} - {e}", file=sys.stderr)
        print(f"    已删除 {len(to_delete)} 个文件")

    return best_pt, len(to_delete), None


def main():
    parser = argparse.ArgumentParser(description="清理 val/test .pt 文件，每 PDB 只保留最佳")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry_run", action="store_true", help="仅预览，不删除")
    group.add_argument("--execute", action="store_true", help="实际执行删除")
    args = parser.parse_args()

    execute = args.execute
    mode = "执行删除" if execute else "预览（不删除）"
    print(f"模式: {mode}\n")

    total_kept = 0
    total_deleted = 0
    errors = []

    for split in SPLITS:
        split_dir = os.path.join(RNA_BASE, split)
        if not os.path.exists(split_dir):
            print(f"⚠ {split} 目录不存在: {split_dir}")
            continue

        pdb_dirs = sorted(
            [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        )
        print(f"{'='*60}")
        print(f"处理 {split} 集 ({len(pdb_dirs)} 个 PDB 目录)")
        print(f"{'='*60}")

        for pdb_id in pdb_dirs:
            pt_dir = os.path.join(split_dir, pdb_id)
            kept, deleted, err = process_pdb(pdb_id, pt_dir, execute=execute)

            if err:
                if err != "no .pt files":
                    print(f"  ⚠ {pdb_id}: {err}")
                    errors.append(f"{split}/{pdb_id}: {err}")
            else:
                total_kept += 1
                total_deleted += deleted

    print(f"\n{'='*60}")
    print(f"汇总: 保留 {total_kept} 个 .pt, 删除 {total_deleted} 个 .pt")
    if errors:
        print(f"异常 ({len(errors)} 个):")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
