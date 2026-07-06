"""
对测试集按序列聚类，结合 refinement 评估结果，按簇分析提升效果。

流程：
  1. 从 FASTA 中提取每个 PDB 的代表链（最长链）
  2. 对 test + train 的代表链跑 cd-hit-est 聚类
  3. 对每个含测试集 PDB 的簇，统计：
     - 簇内 train PDB 数（训练覆盖度）
     - 提升比例、平均提升、平均 rmsd_before
  4. 按提升比例排序输出，找出"难簇"

用法：
  # 仅聚类（还没有评估结果时）
  python analyze_test_clusters.py --output_dir ./split_analysis

  # 结合评估结果
  python analyze_test_clusters.py --output_dir ./split_analysis --eval_csv ./split_analysis/test_rmsd.csv
"""

import os
import csv
import json
import argparse
import subprocess
import statistics
from collections import defaultdict


def parse_fasta_header(header):
    """从 FASTA header 解析 pdb_id, chain_id, split_label
    header 格式: pdb_id_chain_id_split (pdb_id 可能含下划线)"""
    parts = header.split("_")
    split_label = parts[-1]
    chain_id = parts[-2]
    pdb_id = "_".join(parts[:-2])
    return pdb_id, chain_id, split_label


def load_fasta(fasta_path):
    """返回 {header: sequence}"""
    sequences = {}
    with open(fasta_path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences[header] = "".join(seq_lines)
                header = line[1:]  # 去掉 >
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            sequences[header] = "".join(seq_lines)
    return sequences


def pick_representative_chains(fasta_path):
    """每个 PDB 选最长链，返回 {pdb_id: (header, sequence)}"""
    sequences = load_fasta(fasta_path)
    pdb_best = {}  # pdb_id -> (header, sequence, length)

    for header, seq in sequences.items():
        pdb_id, chain_id, split_label = parse_fasta_header(header)
        if pdb_id not in pdb_best or len(seq) > len(pdb_best[pdb_id][1]):
            pdb_best[pdb_id] = (header, seq)

    return pdb_best


def write_representative_fasta(pdb_best, output_path):
    """写代表链 FASTA"""
    with open(output_path, "w") as f:
        for pdb_id, (header, seq) in sorted(pdb_best.items()):
            f.write(f">{header}\n{seq}\n")
    print(f"代表链 FASTA 写入: {output_path} ({len(pdb_best)} 条)")


def run_cdhit(fasta_path, threshold, output_dir):
    """跑 cd-hit-est，解析 .clstr 返回 {cluster_id: [header, ...]}"""
    output_prefix = os.path.join(output_dir, f"rep_clusters_{threshold:.2f}")
    clstr_path = output_prefix + ".clstr"

    if os.path.exists(clstr_path):
        print(f"  复用已有 .clstr: {clstr_path}")
    else:
        cmd = [
            "cd-hit-est",
            "-i", fasta_path,
            "-o", output_prefix,
            "-c", str(threshold),
            "-n", "8",
            "-d", "0",
            "-M", "16000",
            "-T", "8",
            "-g", "1",
        ]
        print(f"  运行 cd-hit-est -c {threshold:.2f} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  cd-hit-est 错误: {result.stderr[:500]}")
            return None

    clusters = defaultdict(list)
    current_cluster = None
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line and current_cluster is not None:
                parts = line.split(">")
                if len(parts) >= 2:
                    header = parts[1].split("...")[0].strip()
                    clusters[current_cluster].append(header)

    return dict(clusters)


def add_missing_singletons(clusters, pdb_best, extra_test_pdbs=None):
    """cd-hit-est 会静默丢弃短序列 (<~16 nt for -n 8)。
    把没有出现在任何簇中的 PDB 每人补一个独立簇，确保总数不漏。
    extra_test_pdbs: 因所有链 < 5 nt 根本没进 FASTA 的测试集 PDB。"""
    all_clustered = set()
    for headers in clusters.values():
        for h in headers:
            all_clustered.add(h)

    max_cid = max(clusters.keys()) if clusters else 0

    # 1. 进了 FASTA 但被 cd-hit 丢弃的（6-10 nt 短链）
    for pdb_id, (header, seq) in pdb_best.items():
        if header not in all_clustered:
            max_cid += 1
            clusters[max_cid] = [header]

    # 2. 根本没进 FASTA 的（所有链 < 5 nt）
    if extra_test_pdbs:
        for pdb_id in sorted(extra_test_pdbs):
            max_cid += 1
            # 构造合成 header，chain_id 用 ? 表示未知
            clusters[max_cid] = [f"{pdb_id}_?_test"]

    return clusters


def load_eval_csv(csv_path):
    """返回 {pdb_id: {rmsd_before, rmsd_after, improvement, improvement_pct}}"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    results = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            results[row["pdb_id"]] = {
                "rmsd_before": float(row["rmsd_before"]),
                "rmsd_after": float(row["rmsd_after"]),
                "improvement": float(row["improvement"]),
                "improvement_pct": float(row["improvement_pct"]),
                "num_atoms": int(row["num_atoms"]),
            }
    return results


def analyze(clusters, eval_results):
    """按簇统计，返回分析结果列表"""
    cluster_stats = []

    for cid, headers in clusters.items():
        pdb_splits = defaultdict(set)  # split -> set of pdb_ids
        for header in headers:
            pdb_id, chain_id, split_label = parse_fasta_header(header)
            pdb_splits[split_label].add(pdb_id)

        test_pdbs = pdb_splits.get("test", set())
        if not test_pdbs:
            continue  # 不含测试集 PDB 的簇，跳过

        train_pdbs = pdb_splits.get("train", set())
        val_pdbs = pdb_splits.get("val", set())

        # 统计测试集 PDB 的 refinement 效果
        improvements = []
        rmsd_befores = []
        num_atoms_list = []
        improved_count = 0
        missing_eval = []

        for pdb_id in sorted(test_pdbs):
            if pdb_id in eval_results:
                imp = eval_results[pdb_id]["improvement"]
                rmsd_b = eval_results[pdb_id]["rmsd_before"]
                improvements.append(imp)
                rmsd_befores.append(rmsd_b)
                if imp > 0:
                    improved_count += 1
                if "num_atoms" in eval_results[pdb_id]:
                    num_atoms_list.append(eval_results[pdb_id]["num_atoms"])
            else:
                missing_eval.append(pdb_id)

        n_evaluated = len(improvements)
        stat = {
            "cluster_id": cid,
            "n_train": len(train_pdbs),
            "n_val": len(val_pdbs),
            "n_test": len(test_pdbs),
            "n_evaluated": n_evaluated,
            "improved_ratio": round(improved_count / n_evaluated, 3) if n_evaluated > 0 else None,
            "avg_improvement": round(statistics.mean(improvements), 4) if improvements else None,
            "median_improvement": round(statistics.median(improvements), 4) if improvements else None,
            "avg_rmsd_before": round(statistics.mean(rmsd_befores), 4) if rmsd_befores else None,
            "avg_num_atoms": round(statistics.mean(num_atoms_list), 1) if num_atoms_list else None,
            "test_pdbs": sorted(test_pdbs),
            "train_pdbs": sorted(train_pdbs)[:10],  # 只记前10个
            "missing_eval": missing_eval,
        }
        cluster_stats.append(stat)

    return cluster_stats


def print_summary(cluster_stats, has_eval):
    """打印分析摘要"""
    if not cluster_stats:
        print("\n没有找到含测试集 PDB 的簇")
        return

    # 按提升比例排序（差的在前）
    if has_eval:
        cluster_stats.sort(key=lambda x: x["improved_ratio"] if x["improved_ratio"] is not None else -1)
    else:
        cluster_stats.sort(key=lambda x: -x["n_train"])  # 按训练覆盖度排序

    print(f"\n{'='*80}")
    print(f"测试集簇分析 ({len(cluster_stats)} 个簇含测试集 PDB)")
    print(f"{'='*80}")

    if has_eval:
        print(f"{'簇ID':<8} {'test数':<8} {'train数':<8} {'eval数':<8} {'提升比':<10} {'平均提升(Å)':<14} {'平均rmsd_before':<16}")
        print("-" * 80)
        for s in cluster_stats:
            imp_str = f"{s['avg_improvement']:+.4f}" if s["avg_improvement"] is not None else "N/A"
            ratio_str = f"{s['improved_ratio']*100:.0f}%" if s["improved_ratio"] is not None else "N/A"
            rmsd_str = f"{s['avg_rmsd_before']:.4f}" if s["avg_rmsd_before"] is not None else "N/A"
            print(f"{s['cluster_id']:<8} {s['n_test']:<8} {s['n_train']:<8} {s['n_evaluated']:<8} "
                  f"{ratio_str:<10} {imp_str:<14} {rmsd_str:<16}")
    else:
        print(f"{'簇ID':<8} {'test数':<8} {'train数':<8} {'val数':<8}")
        print("-" * 50)
        for s in cluster_stats:
            print(f"{s['cluster_id']:<8} {s['n_test']:<8} {s['n_train']:<8} {s['n_val']:<8}")

    # 全局统计
    total_test = sum(s["n_test"] for s in cluster_stats)
    total_eval = sum(s["n_evaluated"] for s in cluster_stats)
    total_improved = sum(
        int(s["improved_ratio"] * s["n_evaluated"]) if s["improved_ratio"] is not None else 0
        for s in cluster_stats
    )
    all_imps = []
    for s in cluster_stats:
        if s["avg_improvement"] is not None and s["n_evaluated"] > 0:
            all_imps.extend([s["avg_improvement"]] * s["n_evaluated"])  # 近似

    print(f"\n{'='*80}")
    print("全局统计")
    print(f"{'='*80}")
    print(f"  测试集 PDB 总数: {total_test}")
    if has_eval:
        print(f"  已评估: {total_eval}")
        print(f"  整体提升比例: {total_improved}/{total_eval} "
              f"({total_improved/total_eval*100:.1f}%)" if total_eval > 0 else "")

    # 标记难簇和好簇
    if has_eval and cluster_stats:
        worst = [s for s in cluster_stats if s["improved_ratio"] is not None and s["improved_ratio"] < 0.3]
        best = [s for s in cluster_stats if s["improved_ratio"] is not None and s["improved_ratio"] > 0.8]
        if worst:
            print(f"\n  难簇 (提升比例 < 30%): {len(worst)} 个")
            for s in worst[:5]:
                train_info = f"train覆盖={s['n_train']}" if s['n_train'] > 0 else "train无覆盖!"
                print(f"    簇{s['cluster_id']}: test={s['n_test']}, {train_info}, "
                      f"平均提升={s['avg_improvement']:+.4f}Å, 平均初始RMSD={s['avg_rmsd_before']:.4f}Å")
        if best:
            print(f"\n  好簇 (提升比例 > 80%): {len(best)} 个")
            for s in best[:5]:
                train_info = f"train覆盖={s['n_train']}" if s['n_train'] > 0 else "train无覆盖!"
                print(f"    簇{s['cluster_id']}: test={s['n_test']}, {train_info}, "
                      f"平均提升={s['avg_improvement']:+.4f}Å, 平均初始RMSD={s['avg_rmsd_before']:.4f}Å")

        # ---- 对比表 1: Train 覆盖 vs 提升效果 ----
        clusters_with_eval = [s for s in cluster_stats if s["improved_ratio"] is not None]
        train_zero = [s for s in clusters_with_eval if s["n_train"] == 0]
        train_pos = [s for s in clusters_with_eval if s["n_train"] > 0]

        def _weighted_avg(stats_list, key, weight_key="n_evaluated"):
            total_w = sum(s[weight_key] for s in stats_list)
            if total_w == 0:
                return 0.0
            return sum(s[key] * s[weight_key] for s in stats_list if s[key] is not None) / total_w

        print(f"\n{'='*80}")
        print("对比分析 1: Train 覆盖度 vs 提升效果 (按评估数加权)")
        print(f"{'='*80}")
        print(f"{'':<20} {'簇数':<8} {'评估数':<8} {'平均提升比':<12} {'平均提升(Å)':<14} {'平均rmsd_before':<16} {'平均原子数':<12}")
        print("-" * 100)
        for label, subset in [("train=0 的簇", train_zero), ("train>0 的簇", train_pos)]:
            if not subset:
                continue
            n_clusters = len(subset)
            n_eval = sum(s["n_evaluated"] for s in subset)
            avg_ratio = _weighted_avg(subset, "improved_ratio")
            avg_imp = _weighted_avg(subset, "avg_improvement")
            avg_rmsd = _weighted_avg(subset, "avg_rmsd_before")
            avg_atoms = _weighted_avg(subset, "avg_num_atoms")
            print(f"{label:<20} {n_clusters:<8} {n_eval:<8} {avg_ratio*100:.1f}%        {avg_imp:+.4f}         {avg_rmsd:.4f}           {avg_atoms:.0f}")

        # ---- 对比表 2: 难簇 vs 好簇画像 ----
        worst_eval = [s for s in worst if s["improved_ratio"] is not None]
        best_eval = [s for s in best if s["improved_ratio"] is not None]

        print(f"\n{'='*80}")
        print("对比分析 2: 难簇 vs 好簇 特征画像 (按评估数加权)")
        print(f"{'='*80}")
        print(f"{'':<20} {'簇数':<8} {'评估数':<8} {'平均提升比':<12} {'平均rmsd_before':<16} {'平均原子数':<12} {'train=0占比':<12}")
        print("-" * 100)
        for label, subset in [("难簇 (<30%)", worst_eval), ("好簇 (>80%)", best_eval)]:
            if not subset:
                continue
            n_clusters = len(subset)
            n_eval = sum(s["n_evaluated"] for s in subset)
            avg_ratio = _weighted_avg(subset, "improved_ratio")
            avg_rmsd = _weighted_avg(subset, "avg_rmsd_before")
            avg_atoms = _weighted_avg(subset, "avg_num_atoms")
            train_zero_frac = sum(1 for s in subset if s["n_train"] == 0) / n_clusters if n_clusters > 0 else 0
            print(f"{label:<20} {n_clusters:<8} {n_eval:<8} {avg_ratio*100:.1f}%        {avg_rmsd:.4f}           {avg_atoms:.0f}            {train_zero_frac*100:.0f}%")



def main():
    parser = argparse.ArgumentParser(description="测试集按簇 refinement 效果分析")
    parser.add_argument("--output_dir", default="./split_analysis")
    parser.add_argument("--fasta", default=None,
                        help="已有 FASTA 文件路径 (默认使用 output_dir/all_rna_chains.fasta)")
    parser.add_argument("--eval_csv", default=None,
                        help="评估 CSV 路径 (evaluate.py 的输出)")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="cd-hit 聚类阈值 (默认 0.90)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="多个阈值，逗号分隔，如 '0.80,0.85,0.90,0.95'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定阈值列表
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = [args.threshold]

    # 1. 加载评估结果
    eval_results = load_eval_csv(args.eval_csv)
    has_eval = len(eval_results) > 0
    if has_eval:
        print(f"加载评估结果: {len(eval_results)} 个 PDB")
    else:
        print("未提供评估 CSV，仅输出聚类分配 (后续可用 --eval_csv 补充)")

    # 2. 提取代表链
    fasta_path = args.fasta or os.path.join(args.output_dir, "all_rna_chains.fasta")
    if not os.path.exists(fasta_path):
        print(f"错误: FASTA 文件不存在: {fasta_path}")
        print("请先运行 recover_split_params.py 生成 FASTA")
        return

    print(f"加载 FASTA: {fasta_path}")
    pdb_best = pick_representative_chains(fasta_path)
    print(f"  选了 {len(pdb_best)} 个 PDB 的代表链 (最长链)")

    # 找出测试集目录中有但 FASTA 没有的 PDB（链长全部 < 5 nt，被过滤掉了）
    test_dir = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA/test"
    test_pdbs_in_dir = set(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )
    extra_test_pdbs = test_pdbs_in_dir - set(pdb_best.keys())
    if extra_test_pdbs:
        print(f"  测试集中有 {len(extra_test_pdbs)} 个 PDB 无代表链 (链长全部 < 5 nt)，将作为独立簇处理")
        print(f"    示例: {sorted(extra_test_pdbs)[:10]}")

    rep_fasta = os.path.join(args.output_dir, "representative_chains.fasta")
    write_representative_fasta(pdb_best, rep_fasta)

    # 3. 对每个阈值跑聚类和分析
    for threshold in thresholds:
        clusters = run_cdhit(rep_fasta, threshold, args.output_dir)
        if clusters is None:
            continue

        clusters = add_missing_singletons(clusters, pdb_best, extra_test_pdbs)

        n_test_clusters = sum(
            1 for headers in clusters.values()
            if any(parse_fasta_header(h)[2] == "test" for h in headers)
        )
        print(f"  -c {threshold:.2f}: {len(clusters)} 个簇, 其中 {n_test_clusters} 个含测试集")

        stats = analyze(clusters, eval_results)

        # 保存
        result_path = os.path.join(args.output_dir, f"test_cluster_analysis_{threshold:.2f}.json")
        with open(result_path, "w") as f:
            json.dump(stats, indent=2, default=str, fp=f)
        print(f"  结果保存至: {result_path}")

        print_summary(stats, has_eval)


if __name__ == "__main__":
    main()
