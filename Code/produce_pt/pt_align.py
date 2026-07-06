"""
基于序列比对的 .pt 文件生成脚本。

与 pt.py 的区别：
  - pt.py 的软对齐按残基顺序硬匹配（global_res_idx），缺口后全部错位
  - 本脚本用 Needleman-Wunsch 序列比对建立 GT↔Pred 残基映射，正确处理 gap

输出：
  - .pt 文件保存到 RNA_aligned/ （不覆盖原有 RNA/ 下的数据）
  - 运行结束后生成 alignment_report.csv

用法：
  python pt_align.py --all
"""

import os
import re
import csv
import sys
import argparse
import torch
from tqdm import tqdm
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionException
warnings.simplefilter('ignore', PDBConstructionException)

ELEMENT_TO_Z = {'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'MG': 12, 'K': 19}

CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
RNA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
RNA_ALIGNED_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA_aligned"
PRED_BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"

ALL_SEEDS = [42, 43, 44, 45]


# ─── CIF 解析 ───────────────────────────────────────────

def extract_residue_data(cif_path, structure_id="id"):
    """
    返回 (sequence_str, residues_list)
      sequence_str: "ACGUA..."
      residues[i]:  list[dict], 每个原子 {'name', 'element', 'pos'}
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_id, cif_path)
    except Exception:
        return None, None

    sequence = []
    residues = []

    model = structure[0]
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            res_name = res.get_resname().strip()
            sequence.append(res_name[0] if res_name else 'N')
            atoms = []
            for atom in res:
                element = atom.element.upper()
                if element == 'H':
                    continue
                atoms.append({
                    'name': atom.get_name().strip(),
                    'element': element,
                    'pos': atom.get_coord(),
                })
            residues.append(atoms)

    if not residues:
        return None, None
    return "".join(sequence), residues


# ─── 序列比对 ────────────────────────────────────────────

def align_sequences(gt_seq, pred_seq):
    """
    Needleman-Wunsch 全局比对。
    返回 (residue_map, identity)
      residue_map: [(gt_idx, pred_idx), ...]
      identity: 匹配位点数 / max(len(gt), len(pred))
    """
    if not gt_seq or not pred_seq:
        return [], 0.0

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -1

    alignments = aligner.align(gt_seq, pred_seq)
    if not alignments:
        return [], 0.0

    best = alignments[0]
    aligned_gt, aligned_pred = best[0], best[1]

    gt_idx = 0
    pred_idx = 0
    residue_map = []
    matches = 0
    total = max(len(gt_seq), len(pred_seq))

    for gt_char, pred_char in zip(aligned_gt, aligned_pred):
        if gt_char != '-' and pred_char != '-':
            residue_map.append((gt_idx, pred_idx))
            if gt_char == pred_char:
                matches += 1
        if gt_char != '-':
            gt_idx += 1
        if pred_char != '-':
            pred_idx += 1

    identity = matches / total if total > 0 else 0.0
    return residue_map, identity


# ─── 原子匹配 & .pt 构建 ─────────────────────────────────

def native_radius_graph(x, r):
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(float('inf'))
    mask = dist < r
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


def build_pt_data(gt_residues, pred_residues, residue_map,
                  pdb_id, seed, pred_filename):
    aligned_gt_pos = []
    aligned_pred_pos = []
    atomic_numbers = []

    for gt_idx, pred_idx in residue_map:
        gt_atoms = gt_residues[gt_idx]
        pred_atoms = pred_residues[pred_idx]
        pred_atom_by_name = {a['name']: a for a in pred_atoms}

        for gt_atom in gt_atoms:
            name = gt_atom['name']
            if name in pred_atom_by_name:
                element = gt_atom['element']
                z = ELEMENT_TO_Z.get(element, 0)
                if z == 0:
                    continue
                aligned_gt_pos.append(gt_atom['pos'])
                aligned_pred_pos.append(pred_atom_by_name[name]['pos'])
                atomic_numbers.append(z)

    if len(aligned_gt_pos) < 10:
        return None

    total_gt_atoms = sum(len(r) for r in gt_residues)
    alignment_ratio = len(aligned_gt_pos) / total_gt_atoms if total_gt_atoms > 0 else 0

    pos_tensor = torch.tensor(aligned_gt_pos, dtype=torch.float)
    pos_pred_tensor = torch.tensor(aligned_pred_pos, dtype=torch.float)
    z_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
    edge_index = native_radius_graph(pos_pred_tensor, r=4.5)

    return {
        'pos': pos_tensor,
        'pos_pred': pos_pred_tensor,
        'atomic_numbers': z_tensor,
        'edge_index': edge_index,
        'pdb_id': pdb_id,
        'seed': seed,
        'pred_filename': pred_filename,
        'alignment_ratio': alignment_ratio,
    }


# ─── 从现有 .pt 文件名解析 seed 和 pred_cif ──────────────

def parse_existing_pt(filename):
    """
    输入: "157d_s45_157d_sample_0.pt"
    返回: (seed, pred_cif)  如 (45, "157d_sample_0.cif")
    格式: {pdb_id}_s{seed}_{rest}.pt  →  seed 在 _s 和下一个 _ 之间
    """
    m = re.search(r'_s(\d+)_(.+)\.pt$', filename)
    if not m:
        return None, None
    seed = int(m.group(1))
    pred_cif = m.group(2) + ".cif"
    return seed, pred_cif


def get_val_test_targets(split, pdb_id):
    """
    读取 RNA/<split>/<pdb_id>/ 下已有的 .pt 文件，
    返回 [(seed, pred_cif), ...] —— 每个 PDB 只有最好的那一个。
    """
    pdb_dir = os.path.join(RNA_DIR, split, pdb_id)
    if not os.path.isdir(pdb_dir):
        return []
    targets = []
    for f in os.listdir(pdb_dir):
        if f.endswith('.pt'):
            seed, pred_cif = parse_existing_pt(f)
            if seed is not None:
                targets.append((seed, pred_cif))
    return targets


# ─── 单 PDB 处理 ─────────────────────────────────────────

def process_one_sample(pdb_id, seed, pred_cif, save_dir):
    """
    处理单个 (pdb_id, seed, pred_cif)。
    返回 (saved: bool, csv_row: dict)
    """
    gt_cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
    if not os.path.exists(gt_cif_path):
        return False, {'pdb_id': pdb_id, 'seed': seed, 'pred_cif': pred_cif,
                       'gt_seq': '', 'pred_seq': '', 'gt_len': 0, 'pred_len': 0,
                       'alignment_identity': 0.0, 'n_matched_residues': 0,
                       'note': 'gt_cif_missing'}

    gt_seq, gt_residues = extract_residue_data(gt_cif_path, pdb_id)
    if gt_seq is None:
        return False, {'pdb_id': pdb_id, 'seed': seed, 'pred_cif': pred_cif,
                       'gt_seq': '', 'pred_seq': '', 'gt_len': 0, 'pred_len': 0,
                       'alignment_identity': 0.0, 'n_matched_residues': 0,
                       'note': 'gt_cif_parse_failed'}

    pred_cif_path = os.path.join(
        PRED_BASE_DIR,
        f"pred_output_{pdb_id}_seed_{seed}",
        pdb_id, f"seed_{seed}", "predictions",
        pred_cif
    )
    if not os.path.exists(pred_cif_path):
        return False, {'pdb_id': pdb_id, 'seed': seed, 'pred_cif': pred_cif,
                       'gt_seq': gt_seq, 'pred_seq': '',
                       'gt_len': len(gt_seq), 'pred_len': 0,
                       'alignment_identity': 0.0, 'n_matched_residues': 0,
                       'note': 'pred_cif_missing'}

    pred_seq, pred_residues = extract_residue_data(pred_cif_path, f"{pdb_id}_pred")
    if pred_seq is None:
        return False, {'pdb_id': pdb_id, 'seed': seed, 'pred_cif': pred_cif,
                       'gt_seq': gt_seq, 'pred_seq': '',
                       'gt_len': len(gt_seq), 'pred_len': 0,
                       'alignment_identity': 0.0, 'n_matched_residues': 0,
                       'note': 'pred_cif_parse_failed'}

    residue_map, identity = align_sequences(gt_seq, pred_seq)

    seq_note = 'ok' if gt_seq == pred_seq else 'seq_mismatch'
    row = {
        'pdb_id': pdb_id, 'seed': seed, 'pred_cif': pred_cif,
        'gt_seq': gt_seq, 'pred_seq': pred_seq,
        'gt_len': len(gt_seq), 'pred_len': len(pred_seq),
        'alignment_identity': round(identity, 4),
        'n_matched_residues': len(residue_map),
        'note': seq_note,
    }

    if not residue_map:
        return False, row

    pt_data = build_pt_data(gt_residues, pred_residues, residue_map,
                            pdb_id, seed, pred_cif)
    if pt_data is None:
        return False, row

    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{pdb_id}_s{seed}_{pred_cif.replace('.cif', '.pt')}"
    torch.save(pt_data, os.path.join(save_dir, save_name))
    return True, row


def process_pdb(pdb_id, split):
    """
    处理一个 PDB。train 全量，val/test 只处理已有 .pt 对应的最佳 sample。
    返回 (pt_count, csv_rows)
    """
    save_dir = os.path.join(RNA_ALIGNED_DIR, split, pdb_id)
    csv_rows = []
    pt_count = 0

    if split in ("val", "test"):
        targets = get_val_test_targets(split, pdb_id)
    else:
        # train: 全部 seed × 全部 sample
        targets = []
        for seed in ALL_SEEDS:
            pred_folder = os.path.join(
                PRED_BASE_DIR,
                f"pred_output_{pdb_id}_seed_{seed}",
                pdb_id, f"seed_{seed}", "predictions"
            )
            if os.path.isdir(pred_folder):
                for f in os.listdir(pred_folder):
                    if f.endswith('.cif'):
                        targets.append((seed, f))

    for seed, pred_cif in targets:
        saved, row = process_one_sample(pdb_id, seed, pred_cif, save_dir)
        csv_rows.append(row)
        if saved:
            pt_count += 1

    return pt_count, csv_rows


# ─── 主流程 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    elif args.all:
        splits = ["train", "val", "test"]
    else:
        print("请指定 --all 或 --split")
        sys.exit(1)

    total_pt = 0
    all_rows = []

    for split in splits:
        split_src = os.path.join(RNA_DIR, split)
        if not os.path.isdir(split_src):
            print(f"⚠ {split} 源目录不存在: {split_src}")
            continue

        pdb_folders = sorted(
            d for d in os.listdir(split_src)
            if os.path.isdir(os.path.join(split_src, d))
        )
        print(f"\n处理 {split} 集 ({len(pdb_folders)} 个 PDB)")

        for pdb_id in tqdm(pdb_folders, desc=f"Processing {split}"):
            count, rows = process_pdb(pdb_id, split)
            total_pt += count
            all_rows.extend(rows)

    # ─── 写入 CSV ──────────────────────────────────────
    csv_path = os.path.join(RNA_ALIGNED_DIR, "alignment_report.csv")
    fieldnames = ['pdb_id', 'seed', 'pred_cif', 'gt_seq', 'pred_seq',
                  'gt_len', 'pred_len', 'alignment_identity', 'n_matched_residues', 'note']
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ─── 统计 ──────────────────────────────────────────
    valid = [r for r in all_rows if r['pred_cif']]
    mismatched = [r for r in valid if r['note'] == 'seq_mismatch']
    matched = [r for r in valid if r['note'] == 'ok']
    n_valid = len(valid)
    mismatched_pdbs = set(r['pdb_id'] for r in mismatched)

    print(f"\n{'='*60}")
    print("完成")
    print(f"{'='*60}")
    print(f"  生成 .pt: {total_pt} 个")
    print(f"  对齐报告: {csv_path}")
    if n_valid > 0:
        print(f"  序列完全一致: {len(matched)} 个 ({len(matched)/n_valid*100:.1f}%)")
        print(f"  序列有差异:   {len(mismatched)} 个 ({len(mismatched)/n_valid*100:.1f}%)")
    print(f"  受影响的 PDB 数: {len(mismatched_pdbs)}")
    if mismatched_pdbs:
        print(f"    示例: {sorted(mismatched_pdbs)[:20]}")


if __name__ == "__main__":
    main()
