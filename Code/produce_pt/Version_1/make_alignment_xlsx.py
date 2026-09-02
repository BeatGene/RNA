"""
生成 RNA 序列对齐报告 .xlsx。

对 1979 个 PDB 各取一条代表性 (seed, pred_cif)，提取 GT/Pred 核苷酸序列并做比对。
已有 alignment_report.csv 的数据直接复用，skipped 的补全，无 pred 数据的标记原因。

输出: Code/produce_pt/alignment_summary.xlsx

用法:
  python make_alignment_xlsx.py
"""

import os
import re
import sys
import argparse
from tqdm import tqdm

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionException
warnings.simplefilter('ignore', PDBConstructionException)

CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
RNA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
RNA_ALIGNED_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA_aligned"
PRED_BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
REPORT_CSV = os.path.join(RNA_ALIGNED_DIR, "alignment_report.csv")
OUTPUT_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Code/produce_pt"

ALL_SEEDS = [42, 43, 44, 45]


# ─── CIF 序列提取 (轻量，不取原子坐标) ──────────────

def extract_sequence_from_cif(cif_path, structure_id="id"):
    """只提取核苷酸序列字符串，不做原子解析。"""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_id, cif_path)
    except Exception:
        return None

    seq = []
    model = structure[0]
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            res_name = res.get_resname().strip()
            seq.append(res_name[0] if res_name else 'N')

    return "".join(seq) if seq else None


# ─── 序列比对 ────────────────────────────────────────────

def align_sequences(gt_seq, pred_seq):
    if not gt_seq or not pred_seq:
        return 0.0, 0

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -1

    alignments = aligner.align(gt_seq, pred_seq)
    if not alignments:
        return 0.0, 0

    best = alignments[0]
    aligned_gt, aligned_pred = best[0], best[1]

    gt_idx = 0
    pred_idx = 0
    matched_residues = 0
    matches = 0
    total = max(len(gt_seq), len(pred_seq))

    for gt_char, pred_char in zip(aligned_gt, aligned_pred):
        if gt_char != '-' and pred_char != '-':
            matched_residues += 1
            if gt_char == pred_char:
                matches += 1
        if gt_char != '-':
            gt_idx += 1
        if pred_char != '-':
            pred_idx += 1

    identity = matches / total if total > 0 else 0.0
    return round(identity, 4), matched_residues


# ─── 文件名解析 ──────────────────────────────────────────

def parse_seed_from_pt_filename(filename):
    m = re.search(r'_s(\d+)_', filename)
    return int(m.group(1)) if m else None


def parse_pred_cif_from_pt_filename(filename):
    """从 .pt 文件名提取 pred_cif。格式: ..._sNN_<pred_cif_without_extension>.pt"""
    m = re.search(r'_s\d+_(.+)\.pt$', filename)
    return m.group(1) + ".cif" if m else None


# ─── 找一个代表性的 (seed, pred_cif) ────────────────────

def find_representative_sample(split, pdb_id):
    """返回 (seed, pred_cif) 或 (None, None)"""
    # 优先从 RNA_aligned 找
    for base in [RNA_ALIGNED_DIR, RNA_DIR]:
        pdb_dir = os.path.join(base, split, pdb_id)
        if not os.path.isdir(pdb_dir):
            continue
        for f in sorted(os.listdir(pdb_dir)):
            if f.endswith('.pt'):
                seed = parse_seed_from_pt_filename(f)
                pred_cif = parse_pred_cif_from_pt_filename(f)
                if seed is not None and pred_cif is not None:
                    return seed, pred_cif

    # 没有 .pt 文件，去 pred 文件夹找任意 sample_0
    for seed in ALL_SEEDS:
        pred_folder = os.path.join(
            PRED_BASE_DIR,
            f"pred_output_{pdb_id}_seed_{seed}",
            pdb_id, f"seed_{seed}", "predictions"
        )
        if not os.path.isdir(pred_folder):
            continue
        # 优先 sample_0
        for f in sorted(os.listdir(pred_folder)):
            if f.endswith('.cif'):
                return seed, f

    return None, None


# ─── 处理单个 PDB ────────────────────────────────────────

def process_one_pdb(pdb_id, split, known_rows):
    """
    known_rows: alignment_report.csv 中该 PDB 的所有行（可能为空，可能全是 skipped）
    返回 dict: {pdb_id, split, gt_seq, pred_seq, gt_len, pred_len,
                alignment_identity, n_matched_residues, note}
    """
    # 1. 如果有非 skipped 的行，直接用
    for row in known_rows:
        if row['note'] != 'skipped_existing':
            return {
                'pdb_id': pdb_id,
                'split': split,
                'gt_seq': row['gt_seq'],
                'pred_seq': row['pred_seq'],
                'gt_len': len(row['gt_seq']),
                'pred_len': len(row['pred_seq']),
                'alignment_identity': row['alignment_identity'],
                'n_matched_residues': row['n_matched_residues'],
                'note': row['note'],
            }

    # 2. GT CIF 序列
    gt_cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
    if not os.path.exists(gt_cif_path):
        return {'pdb_id': pdb_id, 'split': split,
                'gt_seq': '', 'pred_seq': '', 'gt_len': 0, 'pred_len': 0,
                'alignment_identity': 0.0, 'n_matched_residues': 0,
                'note': 'gt_cif_missing'}

    gt_seq = extract_sequence_from_cif(gt_cif_path, pdb_id)
    if gt_seq is None:
        return {'pdb_id': pdb_id, 'split': split,
                'gt_seq': '', 'pred_seq': '', 'gt_len': 0, 'pred_len': 0,
                'alignment_identity': 0.0, 'n_matched_residues': 0,
                'note': 'gt_cif_parse_failed'}

    # 3. 找代表性 pred
    seed, pred_cif = find_representative_sample(split, pdb_id)
    if seed is None:
        return {'pdb_id': pdb_id, 'split': split,
                'gt_seq': gt_seq, 'pred_seq': '', 'gt_len': len(gt_seq), 'pred_len': 0,
                'alignment_identity': 0.0, 'n_matched_residues': 0,
                'note': 'no_pred_available'}

    pred_cif_path = os.path.join(
        PRED_BASE_DIR,
        f"pred_output_{pdb_id}_seed_{seed}",
        pdb_id, f"seed_{seed}", "predictions", pred_cif
    )
    if not os.path.exists(pred_cif_path):
        return {'pdb_id': pdb_id, 'split': split,
                'gt_seq': gt_seq, 'pred_seq': '', 'gt_len': len(gt_seq), 'pred_len': 0,
                'alignment_identity': 0.0, 'n_matched_residues': 0,
                'note': 'pred_cif_missing'}

    pred_seq = extract_sequence_from_cif(pred_cif_path, f"{pdb_id}_pred")
    if pred_seq is None:
        return {'pdb_id': pdb_id, 'split': split,
                'gt_seq': gt_seq, 'pred_seq': '', 'gt_len': len(gt_seq), 'pred_len': 0,
                'alignment_identity': 0.0, 'n_matched_residues': 0,
                'note': 'pred_cif_parse_failed'}

    # 4. 比对
    identity, n_matched = align_sequences(gt_seq, pred_seq)
    seq_note = 'ok' if gt_seq == pred_seq else 'seq_mismatch'

    return {'pdb_id': pdb_id, 'split': split,
            'gt_seq': gt_seq, 'pred_seq': pred_seq,
            'gt_len': len(gt_seq), 'pred_len': len(pred_seq),
            'alignment_identity': identity, 'n_matched_residues': n_matched,
            'note': seq_note}


# ─── 主流程 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="输出 .xlsx 路径 (默认 Code/produce_pt/alignment_summary.xlsx)")
    args = parser.parse_args()

    # ── 读取已有 CSV ──────────────────────────────────
    print("读取 alignment_report.csv ...")
    known_by_pdb = {}
    if os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdb = row['pdb_id']
                known_by_pdb.setdefault(pdb, []).append(row)
        print(f"  已加载 {len(known_by_pdb)} 个 PDB 的记录")
    else:
        print("  ⚠ 未找到 alignment_report.csv，将全部重新处理")

    # ── 收集全部 PDB ──────────────────────────────────
    all_pdbs = []  # [(pdb_id, split), ...]
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(RNA_DIR, split)
        if os.path.isdir(split_dir):
            for d in sorted(os.listdir(split_dir)):
                if os.path.isdir(os.path.join(split_dir, d)):
                    all_pdbs.append((d, split))

    print(f"共 {len(all_pdbs)} 个 PDB (train/val/test)")

    # ── 逐个处理 ──────────────────────────────────────
    results = []
    for pdb_id, split in tqdm(all_pdbs, desc="处理 PDB"):
        known = known_by_pdb.get(pdb_id, [])
        row = process_one_pdb(pdb_id, split, known)
        results.append(row)

    # ── 写入 .xlsx ────────────────────────────────────
    output_path = args.output or os.path.join(OUTPUT_DIR, "alignment_summary.xlsx")
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "alignment_summary"

    headers = ["pdb_id", "split", "gt_seq", "pred_seq", "gt_len", "pred_len",
               "alignment_identity", "n_matched_residues", "note"]
    ws.append(headers)

    for r in results:
        ws.append([r[h] for h in headers])

    # 冻结首行
    ws.freeze_panes = "A2"

    wb.save(output_path)

    # ── 统计 ──────────────────────────────────────────
    ok_count = sum(1 for r in results if r['note'] == 'ok')
    mismatch_count = sum(1 for r in results if r['note'] == 'seq_mismatch')
    other_count = len(results) - ok_count - mismatch_count

    print(f"\n{'='*60}")
    print("完成")
    print(f"{'='*60}")
    print(f"  总计: {len(results)} 个 PDB")
    print(f"  序列一致 (ok):            {ok_count}")
    print(f"  序列有差异 (seq_mismatch): {mismatch_count}")
    print(f"  异常/无数据:               {other_count}")
    print(f"  输出: {output_path}")

    # 异常统计
    if other_count > 0:
        notes = {}
        for r in results:
            if r['note'] not in ('ok', 'seq_mismatch'):
                notes[r['note']] = notes.get(r['note'], 0) + 1
        print(f"\n  异常详情:")
        for note, cnt in sorted(notes.items()):
            print(f"    {note}: {cnt}")


if __name__ == "__main__":
    import csv
    main()
