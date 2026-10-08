# 9 个 FoldBench RNA 单体目标：精修前后指标对照

> 固定权重：`epoch=49-step=157550.ckpt`。链长和“本项目 RMSD”来自本次 `v3_rank1_epoch49_20261008/test_locked/samples.tsv`；FoldBench RMSD 与 lDDT 来自此前 `v3_20261008/foldbench_9targets` 对同一目标、同一 seed/sample 的 ProtenixV3Input / ProtenixV3Refined 评分。两次精修导出不是同一个结果文件，因此下表并列展示两次评估的各自口径，不将数值直接相减跨口径比较。

| FoldBench 目标 / 本项目 native 链 | 链长 (nt) | 本项目对齐 RMSD 前 (Å) | 本项目对齐 RMSD 后 (Å) | FoldBench RMSD 前 (Å) | FoldBench RMSD 后 (Å) | FoldBench lDDT 前 | FoldBench lDDT 后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7SXP / A | 22 | 4.1838 | 4.5097 | 2.576 | 2.817 | 0.780 | 0.000 |
| 7WIA / V | 50 | 9.1782 | 9.1542 | 9.908 | 9.878 | 0.714 | 0.041 |
| 7WII / V | 50 | 4.4622 | 4.2132 | 3.853 | 3.694 | 0.777 | 0.013 |
| 7ZJ4 / E | 374 | 41.6628 | 41.6590 | 42.167 | 42.164 | 0.626 | 0.523 |
| 8HB8 / A | 55 | 6.6685 | 6.6409 | 5.996 | 5.977 | 0.372 | 0.045 |
| 8ITS / A | 46 | 13.0042 | 12.9918 | 14.016 | 14.064 | 0.608 | 0.002 |
| 8UPT / A | 71 | 4.0872 | 4.1135 | 4.008 | 4.028 | 0.738 | 0.346 |
| 8V1H / A | 75 | 2.5710 | 2.5754 | 2.490 | 2.516 | 0.872 | 0.463 |
| 9G7C / A | 224 | 29.7998 | 29.8034 | 30.446 | 30.446 | 0.516 | 0.443 |
| **9 个目标算术平均** | — | **12.8464** | **12.8512** | **12.829** | **12.843** | **0.667** | **0.208** |

## 读表要点

- **本项目对齐 RMSD**：在 native 可观测原子上对每个候选重新 Kabsch 对齐后计算。9 个目标中 5 个降低、4 个升高；平均 **12.8464 → 12.8512 Å**，略有升高。
- **FoldBench RMSD**：来自 FoldBench/OpenStructure 原始评分 CSV 的 `rmsd` 列。9 个目标中 4 个降低、4 个升高、1 个在 CSV 的 3 位小数精度下不变；平均 **12.829 → 12.843 Å**，略有升高。它与本项目 RMSD 的计算口径不同，不能逐项比较绝对值。
- **FoldBench lDDT**：9 个目标全部降低，平均 **0.667 → 0.208**。例如 7WII 的两种 RMSD 都下降，但 lDDT 从 **0.777 降至 0.013**；全局 RMSD 的小幅改善没有保证局部结构质量。
- 这 9 个目标只是测试集中与 FoldBench RNA 单体任务重叠的子集，**不是完整 FoldBench 榜单分数**。

## 数据来源

- [rank-1 测试集逐样本结果](test_locked/samples.tsv)：`length`、`input_aligned_rmsd`、`refined_aligned_rmsd`。
- [本地保存的 FoldBench 逐目标 lDDT](../foldbench_epoch49_9targets_comparison/paired_lddt_all.tsv)：含目标、seed、sample 和前后 lDDT。
- FoldBench RMSD 由服务器以下两份原始 CSV 的 `rmsd` 列逐目标读取，并与 `pdb_id`、seed、sample 对齐：
  - `evaluation/v3_20261008/foldbench_9targets/evaluation/ProtenixV3Input/raw/monomer_rna_ost.csv`
  - `evaluation/v3_20261008/foldbench_9targets/evaluation/ProtenixV3Refined/raw/monomer_rna_ost.csv`

服务器 CSV 的逐行 RMSD 数值由用户提供；这两份原始 CSV 尚未下载到本地。FoldBench 评分与本次 rank-1 评估使用相同的 9 组 seed/sample 和 epoch 49 权重，但输出结构来自不同运行。链长按本项目 `.pt` 中的 `length` 字段统计。
