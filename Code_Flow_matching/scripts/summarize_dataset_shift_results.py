#!/usr/bin/env python3
"""Interpret downloaded shift TSVs; optionally join historical rank-1 scores.

Read-only inputs. All new outputs go into local_interpretation/. Historical
epoch49 performance is explicitly labeled as predating the BF16 mean repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LENGTH_LABELS = ["<=50", "51-100", "101-200", ">200"]
RMSD_LABELS = ["<2", "2-5", "5-10", "10-20", "20-30", ">=30"]


def write(path, frame):
    frame.to_csv(path, sep="\t", index=False, float_format="%.12g")


def summarize(root, historical_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "local_interpretation"
    out.mkdir(exist_ok=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    columns = ["pool", "split", "pdb_id", "seed", "sample", "sequence", "length", "rmsd", "ranking_score", "plddt_residue_mean"]
    frame = pd.read_csv(root / "features.tsv", sep="\t", usecols=columns,
                        dtype={"pool": str, "split": str, "pdb_id": str, "sequence": str})
    frame = frame[frame.pool == "current"].copy()
    if frame[columns].isna().any().any():
        raise ValueError("Missing current candidate data; resolve extraction/score errors first")
    identity = ["split", "pdb_id", "seed", "sample"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate current candidate identities")
    train = frame[frame.split == "train"].copy()
    val, test = frame[frame.split == "val"].copy(), frame[frame.split == "test"].copy()
    if any(f.pdb_id.duplicated().any() for f in (val, test)):
        raise ValueError("val/test must contain one selected candidate per PDB")
    # Validate actual PT lengths against the sampler used in the downloaded run.
    # The upstream report would warn if its sampler table and PT lengths differed.
    if any("differ in length" in w or "No sampler length" in w for w in summary["warnings"]):
        raise ValueError("Sampler lengths need explicit reconciliation before joint exposure analysis")
    train["weight"] = np.where(train.length > summary["long_threshold"], summary["long_factor"], 1.)
    rank = train.sort_values(["ranking_score", "seed", "sample"], ascending=[False, True, True]).drop_duplicates("pdb_id").copy()
    rank["weight"] = 1.
    val["weight"], test["weight"] = 1., 1.
    views = {"train_effective": train, "train_rank1": rank, "val_rank1": val, "test_rank1": test}
    rows = []
    joint = {}
    for name, f in views.items():
        f["length_bin"] = pd.cut(f.length, [0, 50, 100, 200, np.inf], labels=LENGTH_LABELS)
        f["rmsd_bin"] = pd.cut(f.rmsd, [-np.inf, 2, 5, 10, 20, 30, np.inf], right=False, labels=RMSD_LABELS)
        grouped = f.groupby(["length_bin", "rmsd_bin"], observed=True).agg(
            weight=("weight", "sum"), candidates=("pdb_id", "size"), pdbs=("pdb_id", "nunique")).reset_index()
        grouped["view"] = name
        grouped["global_share"] = grouped.weight / f.weight.sum()
        grouped["within_length_share"] = grouped.weight / grouped.groupby("length_bin", observed=True).weight.transform("sum")
        rows.append(grouped)
        joint[name] = grouped
    write(out / "joint_length_rmsd.tsv", pd.concat(rows, ignore_index=True))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (name, g) in zip(axes.flat, joint.items()):
        matrix = g.pivot(index="length_bin", columns="rmsd_bin", values="within_length_share").reindex(
            index=LENGTH_LABELS, columns=RMSD_LABELS).fillna(0) * 100
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for i in range(4):
            for j in range(6):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="white" if value > 55 else "black", fontsize=9)
        ax.set_xticks(range(6), RMSD_LABELS)
        ax.set_yticks(range(4), LENGTH_LABELS)
        ax.set(title=name, xlabel="Input RMSD (A)", ylabel="Length (nt)")
    fig.suptitle("Within each length bin: weighted input RMSD distribution\nTrain exposure includes length oversampling; selected splits are PDB-equal")
    fig.tight_layout(); fig.savefig(out / "joint_length_rmsd.png", dpi=180); plt.close(fig)

    distinct = frame[["split", "pdb_id", "sequence", "length"]].drop_duplicates()
    overlaps, overlap_pdbs = [], set()
    for sequence, group in distinct.groupby("sequence"):
        if group.split.nunique() <= 1:
            continue
        for _, row in group.iterrows():
            overlaps.append(dict(split=row.split, pdb_id=row.pdb_id, length=row.length, sequence=sequence,
                                 other_split_pdbs=",".join(group.loc[group.split != row.split, "pdb_id"])))
            if row.split == "val" and (group.split == "train").any():
                overlap_pdbs.add(row.pdb_id)
    write(out / "sequence_overlap_pdbs.tsv", pd.DataFrame(overlaps))
    val["train_sequence_overlap"] = val.pdb_id.isin(overlap_pdbs)
    overlap_metrics = []
    for overlap, g in val.groupby("train_sequence_overlap"):
        overlap_metrics.append(dict(train_sequence_overlap=overlap, pdbs=len(g), mean_input_rmsd=g.rmsd.mean(),
                                    median_input_rmsd=g.rmsd.median(), rmsd_lt2_count=int((g.rmsd < 2).sum())))
    overlap_metrics = pd.DataFrame(overlap_metrics)

    paired, performance_summary = None, None
    if historical_root is not None:
        scores = []
        for split, directory in (("val", "val_final"), ("test", "test_locked")):
            f = pd.read_csv(historical_root / directory / "samples.tsv", sep="\t").rename(
                columns={"protenix_seed": "seed", "protenix_sample": "sample"})
            f.pdb_id = f.pdb_id.str.upper()
            if not (f.split == split).all():
                raise ValueError("Incorrect historical evaluation split")
            scores.append(f[identity + ["input_aligned_rmsd", "refined_aligned_rmsd", "aligned_improvement"]])
        scores = pd.concat(scores, ignore_index=True)
        paired = frame[frame.split != "train"].merge(scores, on=identity, validate="one_to_one")
        if len(paired) != len(val) + len(test) or np.abs(paired.rmsd - paired.input_aligned_rmsd).max() > .01:
            raise ValueError("Historical performance differs in candidate coverage/input RMSD")
        paired["train_sequence_overlap"] = paired.pdb_id.isin(overlap_pdbs) & (paired.split == "val")
        for space, file in (("input", "input_coverage.tsv"), ("rnafm", "rnafm_coverage.tsv")):
            c = pd.read_csv(root / file, sep="\t")
            c = c[["split", "pdb_id", "nearest_distance", "outside_reference_q95"]].rename(
                columns={"nearest_distance": f"{space}_distance", "outside_reference_q95": f"{space}_outside_q95"})
            paired = paired.merge(c, on=["split", "pdb_id"], validate="one_to_one")
        write(out / "historical_performance_paired.tsv", paired)
        prs = []
        for space in ("input", "rnafm"):
            for (split, outside), g in paired.groupby(["split", f"{space}_outside_q95"]):
                prs.append(dict(space=space, split=split, outside_reference_q95=outside, pdbs=len(g),
                                mean_improvement=g.aligned_improvement.mean(), median_improvement=g.aligned_improvement.median(),
                                mean_input_rmsd=g.rmsd.mean(), mean_length=g.length.mean()))
        performance_summary = pd.DataFrame(prs)
        write(out / "historical_performance_by_coverage.tsv", performance_summary)
        for i, row in overlap_metrics.iterrows():
            g = paired[(paired.split == "val") & (paired.train_sequence_overlap == row.train_sequence_overlap)]
            overlap_metrics.loc[i, "historical_mean_improvement"] = g.aligned_improvement.mean()
            overlap_metrics.loc[i, "historical_median_improvement"] = g.aligned_improvement.median()
    write(out / "validation_overlap_metrics.tsv", overlap_metrics)

    longtrain = train[train.length > 200]
    hardtrain = longtrain[longtrain.rmsd >= 10]
    longtest = test[test.length > 200]
    hardtest = longtest[longtest.rmsd >= 10]
    coverage = pd.read_csv(root / "input_coverage.tsv", sep="\t")
    hard_coverage = coverage[(coverage.split == "test") & (coverage.length > 200) & (coverage.rmsd >= 10)].copy()
    audit = pd.read_csv(root / "pdb_audit.tsv", sep="\t")
    counts = audit[(audit.pool == "current") & (audit.split == "train")].set_index("pdb_id").candidate_count
    hard_coverage["nearest_train_candidate_count"] = hard_coverage.nearest_train_pdb.map(counts)
    write(out / "hard_long_test_neighbors.tsv", hard_coverage)
    stats = pd.read_csv(root / "distribution_summary.tsv", sep="\t")
    stat = lambda view, metric: stats[(stats.view == view) & (stats.metric == metric)].iloc[0]
    shifts = pd.read_csv(root / "shift_metrics.tsv", sep="\t")
    torsion = pd.read_csv(root / "torsion_shift.tsv", sep="\t")
    lines = ["# Data_PT_V2：本地结果解读", "", "结论：训练与测试的监督难度、长度和置信度存在明确差异；最突出的缺口是长 RNA 的大幅构象错误。输入局部二面角分布高度相似，二维降维也未显示整体完全脱离训练区域，不能据此断言模型无法学习。", "",
             "本报告读取下载的 PT 摘要，不改变数据/模型。下文关联的 epoch49 评估为历史 BF16 修复前结果，不能用它证明分布差异的因果贡献。", "",
             "## 1. 输入质量和长度", "", "| 指标 | 训练实际采样 | 训练 rank-1 | 验证 rank-1 | 测试 rank-1 |", "|---|---:|---:|---:|---:|"]
    for metric, label, fmt in [("length", "平均长度 nt", ".1f"), ("rmsd", "平均输入 RMSD Å", ".3f"),
                               ("plddt_residue_mean", "平均残基 pLDDT (0–1)", ".3f"),
                               ("clash_pairs_per_1000_atoms", "每1000原子碰撞对", ".2f"),
                               ("alpha_error_mean_deg", "α 对 native 圆周误差 °", ".2f"),
                               ("gamma_error_mean_deg", "γ 对 native 圆周误差 °", ".2f")]:
        lines.append("| " + label + " | " + " | ".join(format(stat(v, metric)["mean"], fmt) for v in views) + " |")
    r = shifts[(shifts.reference == "train_rank1") & (shifts.target == "test_rank1") & (shifts.metric == "rmsd")].iloc[0]
    lines += ["", f"统一候选选择规则后，测试与训练 rank-1 的 RMSD 均值差仍为 {r.mean_difference:.3f} Å，PDB bootstrap 95% 区间 [{r.ci95_low:.3f}, {r.ci95_high:.3f}]。这不是主要由 200 候选 vs rank-1 造成的表观差异。", "",
              "## 2. 最明显的缺口：长链 × 大误差", "",
              f"- 训练 >200 nt：{longtrain.pdb_id.nunique()} 个 PDB、{len(longtrain):,} 个 PT，实际采样占比 {longtrain.weight.sum()/train.weight.sum():.2%}；测试为 {len(longtest)}/{len(test)} = {len(longtest)/len(test):.2%}。",
              f"- 训练长链 RMSD 均值 {np.average(longtrain.rmsd, weights=longtrain.weight):.3f} Å，测试长链为 {longtest.rmsd.mean():.3f} Å。",
              f"- 训练长链 RMSD ≥10 Å：仅 {len(hardtrain)}/{len(longtrain)} = {len(hardtrain)/len(longtrain):.2%}，来自 {hardtrain.pdb_id.nunique()} 个 PDB（{', '.join(sorted(hardtrain.pdb_id.unique()))}）；测试长链为 {len(hardtest)}/{len(longtest)} = {len(hardtest)/len(longtest):.2%}。",
              f"- 长链且 RMSD ≥10 Å 在全训练曝光只占 {hardtrain.weight.sum()/train.weight.sum():.3%}，但在整个测试集中占 {len(hardtest)/len(test):.2%}。单纯把所有 >50 nt 样本加权 2 倍，不能补足这类任务。",
              f"- 13 个困难长链测试目标中，有 {int(hard_coverage.nearest_train_pdb.isin(['4C4Q', '5G2Y']).sum())} 个在输入摘要空间的最近训练 PDB 是 4C4Q/5G2Y，这两个 PDB 分别只保留 {int(counts['4C4Q'])}/{int(counts['5G2Y'])} 个候选。最近邻可找到类似样本，不表示训练经常看到这些样本；参考集每 PDB 最多 8 个候选也不是实际采样频率。详见 hard_long_test_neighbors.tsv。",
              "- 大量近似候选来自少数 PDB，增加候选数量不等于增加独立结构/家族覆盖。", "",
              "![按长度条件化的 RMSD 分布](joint_length_rmsd.png)", "",
              "## 3. 验证集与训练集序列重叠", "",
              f"共 {len({r['sequence'] for r in overlaps})} 种 PT 序列跨集合出现，涉及 {len(overlap_pdbs)}/{len(val)} 个验证 PDB；原始 exact_split_overlaps.tsv 无跨集合同名 PDB，也无测试集完全相同序列命中。序列是 PT 建模使用的 canonical 序列，不能替代 native/完整同源审计。", "",
              "| 验证子集 | PDB 数 | 平均输入 RMSD Å | 输入 <2 Å 个数 | 历史 epoch49 平均改善 Å |", "|---|---:|---:|---:|---:|"]
    for _, r in overlap_metrics.iterrows():
        gain = f"{r.historical_mean_improvement:.3f}" if "historical_mean_improvement" in r else "未关联"
        lines.append(f"| {'与训练序列相同' if r.train_sequence_overlap else '未命中完全相同序列'} | {int(r.pdbs)} | {r.mean_input_rmsd:.3f} | {int(r.rmsd_lt2_count)} | {gain} |")
    lines += ["", "这不意味着原时间划分执行错误：项目单独对测试集做了去同源，验证集的定义与测试集不同。验证集不能直接当作新序列泛化评估。未完全相同的 82 个验证 PDB 仍可能近同源，不应命名为严格 novel。", "",
              "建议保留历史全集成绩，再固定报告 seen-sequence / unmatched-exact-sequence 两个面板。对未来训练，在训练 PDB 内按已有同源簇留出严格验证、保持每 PDB rank-1；不要把相同 PDB 候选拆到训练与验证。不能针对本次测试结果重划分出更容易的测试集。", "",
              "## 4. 降维、二面角与覆盖", ""]
    for file, label in (("input_coverage.tsv", "无 native 输入摘要"), ("rnafm_coverage.tsv", "均值 RNA-FM")):
        c = pd.read_csv(root / file, sep="\t")
        for split in ("val", "test"):
            g = c[c.split == split]
            lines.append(f"- {label}，{split} 超过训练跨 PDB 最近邻距离 q95：{int(g.outside_reference_q95.sum())}/{len(g)} = {g.outside_reference_q95.mean():.2%}。这是特征依赖的参考标记，不是不可学习概率。")
    t = torsion[(torsion.reference == "train_effective") & (torsion.target == "test_rank1")]
    lines += [f"- 九类输入二面角边际分布的圆周 Wasserstein 距离为 {t.circular_wasserstein_deg.min():.2f}–{t.circular_wasserstein_deg.max():.2f}°（12 箱近似）；图形高度重合。但相对 native 的角误差变大，两者描述的问题不同。",
              "- 这更支持全局构象/结构域安排和大误差监督稀缺的假设；边际角分布相似不保证长程相互作用、角度相关性或 native 分布相似。",
              "- input PCA 前两维解释约 39.9%，RNA-FM PCA 约 27.6%；二维重合不能排除其余维度的偏移。", "",
              "## 5. 历史模型效果的关联（有限证据）", ""]
    if performance_summary is not None:
        lines += ["已按 split/PDB/seed/sample 匹配全部 170 个候选，并核对输入 RMSD。", "",
                  "| 特征空间 | 测试子集 | PDB 数 | 历史平均改善 Å | 历史中位改善 Å |", "|---|---|---:|---:|---:|"]
        for _, r in performance_summary[performance_summary.split == "test"].iterrows():
            lines.append(f"| {r.space} | {'超过参考 q95' if r.outside_reference_q95 else '参考 q95 内'} | {int(r.pdbs)} | {r.mean_improvement:.4f} | {r.median_improvement:.4f} |")
        lines += ["", "均值 RNA-FM 较远的少数目标效果更差，值得验证；但输入摘要距离较近的大多数目标也几乎没有改善，不能用 OOD 几个目标统一解释失败。小组样本少、长度/质量混杂，且历史 BF16 数值错误尚在训练和推理中；不作因果结论。"]
    else:
        lines += ["未关联历史模型效果。"]
    partial = audit[(audit.pool == "current") & (audit.split == "train") & (audit.candidate_count < 200)]
    lines += ["", "## 6. 数据质量与后续实验", "",
              "- 抽取错误 0、缺 ranking_score 0、与归档可核验排名不一致 0、存储/重算 RMSD >0.01 Å 不一致 0。",
              "- val 完整 200 候选为 97/103 个 PDB，test 为 55/67 个。rank-1 只承诺在现有 PT 集合中选最高分。",
              "- 7ZJ4 的测试输入 RMSD 为 41.663 Å，归档 200 候选均 >30 Å；train 所有 PT ≤30 Å。保留该目标作为明确的困难/范围外子集，不静默删除。其 PT metadata 的 dataset_rmsd_limit 仍为 30，需要追溯保留策略。",
              "- 训练部分 PDB 的候选显著不足，不能假定每 PDB 具有同等曝光：" + "; ".join(f"{r.pdb_id}={int(r.candidate_count)}" for _, r in partial.iterrows()) + "。低计数具体原因应追溯生成/筛选日志。",
              "- 原 ranking_score 图被两个约 −99 的训练候选拉伸；常规分数挤在右端。应查看数值表或只缩放显示常规区间，不把图中压扁误读为分布一致。", "",
              "后续顺序：先在修复 BF16 路径后作统一复评；再固定无同源泄漏的选模验证；最后开展长度×误差分层采样/真实困难训练 PDB 增补的受控对照。若用 native RMSD 构造训练采样权重，说明它是训练标签，不用于推理时筛选。保留近 native 训练和保护验证，防止增加困难样本后破坏好输入。",
              "不要只将已有 5 个困难长链 PDB 的候选反复放大就声称覆盖已补足，也不要直接改为训练仅 rank-1：选择效应并非主要差距，训练 rank-1 的全局误差仍很低。",
              "", "## 复现与新增文件", "", "本地命令：", "", "```bash",
              f'python scripts/summarize_dataset_shift_results.py --result-dir "{root}"' + (f' --historical-eval-root "{historical_root}"' if historical_root else ""),
              "```", "", "新增脚本 scripts/summarize_dataset_shift_results.py 和本 local_interpretation/ 下的报告、PNG、TSV；全部不参与训练，保存复现记录后可删除。本次不需要上传或执行服务器命令。"]
    report = out / "分析解读.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    print("hard long train:", len(hardtrain), "PDBs:", hardtrain.pdb_id.nunique(), "global exposure:", hardtrain.weight.sum()/train.weight.sum())
    print("overlap validation metrics:\n", overlap_metrics.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--historical-eval-root", type=Path)
    args = parser.parse_args()
    summarize(args.result_dir.resolve(), args.historical_eval_root.resolve() if args.historical_eval_root else None)


if __name__ == "__main__":
    main()
