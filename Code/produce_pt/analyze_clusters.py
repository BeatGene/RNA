"""
修复 purity 计算：忽略 unknown split，只检查 train/val/test 之间的交叉污染。
同时统计测试集链在每个簇中的分布情况。

用法：
  python analyze_clusters.py --output_dir ./split_analysis
"""

import os
import json
import argparse
from collections import defaultdict

REAL_SPLITS = {"train", "val", "test"}


def analyze_threshold(output_dir, threshold):
    clstr_path = os.path.join(output_dir, f"clusters_{threshold:.2f}.clstr")
    if not os.path.exists(clstr_path):
        print(f"  ⚠ {clstr_path} 不存在，跳过")
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
    clusters = dict(clusters)

    pure = 0
    mixed_unknown = 0
    mixed_real = 0
    mixed_details = []

    test_cluster_ids = set()
    test_total = 0

    for cid, headers in clusters.items():
        real_splits = set()
        has_unknown = False
        for header in headers:
            split_label = header.split("_")[-1]
            if split_label in REAL_SPLITS:
                real_splits.add(split_label)
                if split_label == "test":
                    test_total += 1
                    test_cluster_ids.add(cid)
            else:
                has_unknown = True

        if len(real_splits) <= 1:
            if has_unknown:
                mixed_unknown += 1
            else:
                pure += 1
        else:
            mixed_real += 1
            mixed_details.append({
                "cluster_id": cid,
                "splits": list(real_splits),
                "n_chains": len(headers),
                "chains": headers[:10],
            })

    total = pure + mixed_unknown + mixed_real
    purity_strict = pure / total if total > 0 else 0
    purity_relaxed = (pure + mixed_unknown) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  阈值 -c {threshold:.2f}")
    print(f"{'='*60}")
    print(f"  总簇数: {total}")
    print(f"  纯簇 (同split):          {pure}")
    print(f"  仅含unknown的簇:          {mixed_unknown}")
    print(f"  真正交叉污染簇:           {mixed_real}")
    print(f"  严格 purity:              {purity_strict:.4f} ({purity_strict*100:.1f}%)")
    print(f"  宽松 purity (忽略unknown): {purity_relaxed:.4f} ({purity_relaxed*100:.1f}%)")
    print(f"  测试集链数: {test_total}, 分布在 {len(test_cluster_ids)} 个簇中")

    if mixed_details:
        print(f"\n  交叉污染示例 (前5个):")
        for d in mixed_details[:5]:
            print(f"    簇{d['cluster_id']}: splits={d['splits']}, {d['n_chains']}条链")
            for ch in d["chains"][:3]:
                print(f"      {ch}")

    return {
        "threshold": threshold,
        "total_clusters": total,
        "pure": pure,
        "mixed_unknown": mixed_unknown,
        "mixed_real": mixed_real,
        "purity_strict": round(purity_strict, 4),
        "purity_relaxed": round(purity_relaxed, 4),
        "test_chains": test_total,
        "test_clusters": len(test_cluster_ids),
        "test_cluster_ids": sorted(test_cluster_ids),
        "mixed_details": mixed_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./split_analysis")
    args = parser.parse_args()

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00]
    all_results = []

    for t in thresholds:
        result = analyze_threshold(args.output_dir, t)
        if result:
            all_results.append(result)

    # 找最佳阈值：pure 最高且 mixed_real 最少
    best = min(all_results, key=lambda x: (-x["purity_relaxed"], x["mixed_real"]))

    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    print(f"最佳推测阈值: cd-hit-est -c {best['threshold']:.2f}")
    print(f"宽松 purity: {best['purity_relaxed']*100:.1f}%")
    print(f"真正交叉污染簇: {best['mixed_real']} 个")
    print(f"测试集 {best['test_chains']} 条链分布在 {best['test_clusters']} 个簇中")

    # 保存
    result_path = os.path.join(args.output_dir, "cluster_analysis_fixed.json")
    with open(result_path, "w") as f:
        json.dump(all_results, indent=2, default=str, fp=f)
    print(f"\n详细结果保存至: {result_path}")


if __name__ == "__main__":
    main()
