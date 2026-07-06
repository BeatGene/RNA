"""
反推 RNA 链聚类划分参数：
1. 从 CIF 文件中提取所有链的序列
2. 用不同 identity 阈值跑 cd-hit-est 聚类
3. 检查同簇的链是否落在同一个 train/val/test 集合中
4. 找出最匹配的阈值

用法：
  python recover_split_params.py --output_dir ./split_analysis
"""

import os
import subprocess
import argparse
import json
from collections import defaultdict
from pathlib import Path

from Bio.PDB import MMCIFParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionException
warnings.simplefilter('ignore', PDBConstructionException)


CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
RNA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
SPLITS = ["train", "val", "test"]

# 常见的 cd-hit-est 阈值 (RNA 常用 80%-95%)
THRESHOLDS_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00]


def extract_chain_sequences():
    """从 CIF 文件中提取每一条链的序列，返回 {pdb_id: {chain_id: sequence}}"""
    parser = MMCIFParser(QUIET=True)
    pdb_chains = defaultdict(dict)

    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    print(f"解析 {len(cif_files)} 个 CIF 文件...")

    for cif_name in cif_files:
        pdb_id = cif_name.replace(".cif", "")
        cif_path = os.path.join(CIF_DIR, cif_name)
        try:
            structure = parser.get_structure(pdb_id, cif_path)
            model = structure[0]
            for chain in model:
                seq = ""
                for res in chain:
                    if res.id[0] != " ":
                        continue
                    res_name = res.get_resname().strip()
                    seq += res_name[0] if res_name else 'N'
                if len(seq) >= 5:  # 过滤过短的链
                    pdb_chains[pdb_id][chain.id] = seq
        except Exception:
            pass

    total_chains = sum(len(chains) for chains in pdb_chains.values())
    print(f"提取了 {len(pdb_chains)} 个 PDB, {total_chains} 条链")
    return pdb_chains


def get_split_map():
    """返回 {pdb_id: split_label}"""
    split_map = {}
    for split in SPLITS:
        split_dir = os.path.join(RNA_DIR, split)
        if not os.path.exists(split_dir):
            continue
        for pdb_id in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, pdb_id)):
                split_map[pdb_id] = split
    return split_map


def write_fasta(pdb_chains, split_map, fasta_path):
    """写出 FASTA 文件，每条链一行，header 格式: >pdb_id|chain_id|split"""
    with open(fasta_path, "w") as f:
        for pdb_id, chains in pdb_chains.items():
            split_label = split_map.get(pdb_id, "unknown")
            for chain_id, seq in chains.items():
                # cd-hit 要求 header 在第一个空格前结束
                header = f">{pdb_id}_{chain_id}_{split_label}"
                f.write(f"{header}\n{seq}\n")
    print(f"FASTA 写入: {fasta_path}")


def run_cdhit(fasta_path, threshold, output_dir):
    """运行 cd-hit-est，返回 {cluster_id: [header1, header2, ...]}"""
    output_prefix = os.path.join(output_dir, f"clusters_{threshold:.2f}")
    cmd = [
        "cd-hit-est",
        "-i", fasta_path,
        "-o", output_prefix,
        "-c", str(threshold),
        "-n", "8",          # 高精度 (对短序列推荐 8)
        "-d", "0",          # 用全长序列描述
        "-M", "16000",      # 内存 (MB)
        "-T", "8",          # 线程
        "-g", "1",          # 精确模式
    ]
    print(f"  运行 cd-hit-est -c {threshold:.2f} ...")
    subprocess.run(cmd, capture_output=True, text=True)

    # 读取 .clstr 文件
    clstr_path = output_prefix + ".clstr"
    if not os.path.exists(clstr_path):
        print(f"  ⚠ 输出文件不存在: {clstr_path}")
        return None

    clusters = defaultdict(list)
    current_cluster = None
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line and current_cluster is not None:
                # 格式: 0\t1234aa, >pdb_id|chain_id|split... *
                #       at +/XX.XX%
                parts = line.split(">")
                if len(parts) >= 2:
                    header = parts[1].split("...")[0].strip()
                    clusters[current_cluster].append(header)

    return dict(clusters)


def evaluate_clusters(clusters, split_map):
    """
    评估聚类质量：
    - purity: 同簇内所有链是否来自同一个 split
    - 返回 (pure_clusters, mixed_clusters, purity_ratio)
    """
    pure = 0
    mixed = 0
    mixed_details = []

    for cluster_id, headers in clusters.items():
        splits_in_cluster = set()
        for header in headers:
            parts = header.split("_")
            # header 格式: pdb_id_chain_id_split
            # 注意 pdb_id 可能包含下划线，所以取最后两个
            split_label = parts[-1]
            splits_in_cluster.add(split_label)

        if len(splits_in_cluster) <= 1:
            pure += 1
        else:
            mixed += 1
            mixed_details.append({
                "cluster_id": cluster_id,
                "splits": list(splits_in_cluster),
                "chains": headers[:10],  # 只记录前10个
                "total_chains": len(headers),
            })

    total = pure + mixed
    purity = pure / total if total > 0 else 0
    return pure, mixed, purity, mixed_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./split_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 提取链序列
    print("=" * 60)
    print("[1/4] 提取链序列")
    print("=" * 60)
    pdb_chains = extract_chain_sequences()

    # 2. 获取当前划分
    print("\n" + "=" * 60)
    print("[2/4] 读取当前 train/val/test 划分")
    print("=" * 60)
    split_map = get_split_map()
    for split in SPLITS:
        count = sum(1 for v in split_map.values() if v == split)
        print(f"  {split}: {count} 个 PDB")

    # 3. 写 FASTA
    print("\n" + "=" * 60)
    print("[3/4] 写 FASTA 文件")
    print("=" * 60)
    fasta_path = os.path.join(args.output_dir, "all_rna_chains.fasta")
    write_fasta(pdb_chains, split_map, fasta_path)

    # 4. 跑不同阈值
    print("\n" + "=" * 60)
    print("[4/4] 用不同阈值测试 cd-hit-est")
    print("=" * 60)

    results = []
    for threshold in THRESHOLDS_TO_TEST:
        clusters = run_cdhit(fasta_path, threshold, args.output_dir)
        if clusters is None:
            continue

        pure, mixed, purity, mixed_details = evaluate_clusters(clusters, split_map)
        n_chains = sum(len(v) for v in clusters.values())

        print(f"  -c {threshold:.2f}: {len(clusters)} 个簇, "
              f"pure={pure}, mixed={mixed}, "
              f"purity={purity:.4f} ({purity*100:.1f}%)")

        results.append({
            "threshold": threshold,
            "n_clusters": len(clusters),
            "n_chains": n_chains,
            "pure_clusters": pure,
            "mixed_clusters": mixed,
            "purity": round(purity, 4),
            "mixed_details": mixed_details[:5],  # 只保存前5个
        })

    # 5. 报告
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)

    # 找最匹配的阈值 (purity 最高但簇数合理的那几个)
    results.sort(key=lambda x: (-x["purity"], x["n_clusters"]))

    print(f"\n{'阈值':<10} {'簇数':<8} {'纯度':<10} {'Mixed簇':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['threshold']:<10.2f} {r['n_clusters']:<8} {r['purity']:<10.4f} {r['mixed_clusters']:<10}")

    # 最佳推测
    best = results[0]
    print(f"\n推测: 使用了 cd-hit-est -c {best['threshold']:.2f} (核苷酸序列相似度阈值 {best['threshold']*100:.0f}%)")
    print(f"该阈值下簇划分纯度: {best['purity']*100:.1f}%")

    # 保存结果
    result_path = os.path.join(args.output_dir, "analysis.json")
    with open(result_path, "w") as f:
        json.dump(results, indent=2, default=str, fp=f)
    print(f"\n详细结果保存至: {result_path}")


if __name__ == "__main__":
    main()
