# V3 RNA refinement evaluation

Checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/last-v1.ckpt`

主 RMSD 是 native observed atoms 的逐候选 Kabsch 对齐 RMSD，单位 Å；改善量 = 输入 − refinement 后。
同一链的候选相关，链层面的宏平均是主要统计单位。分层按输入值固定，避免 refinement 后跨组导致口径改变。RMSD 仅用 native 可观测原子；物理 loss 则按训练定义使用预测结构的全部原子。

## 验证集与测试集总览

| split | unit | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 19722 | 8.1844 | 7.7749 | 0.4095 | 0.6796 | 0.3196 |
| val | PDB-chain macro | 103 | 8.9739 | 8.5815 | 0.3924 | 0.7379 | 0.2621 |
| test | candidate | 12310 | 9.0532 | 8.9891 | 0.0641 | 0.6050 | 0.3949 |
| test | PDB-chain macro | 67 | 10.4918 | 10.4320 | 0.0599 | 0.6567 | 0.3433 |

## 输入 RMSD <2 Å

比例分母分别为该 split 的全部候选或全部 PDB 链；链按候选的平均输入 RMSD 判定。

| split | unit | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | worsen_rate | damage_ge_0.5_rate | post_rmsd_ge_2_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 2630 | 0.1334 | 1.2448 | 1.1185 | 0.1263 | 0.2882 | 0.1506 | 0.1209 |
| val | PDB-chain macro | 13 | 0.1262 | 1.2699 | 1.0927 | 0.1772 | 0.2308 | 0.0769 | 0.1538 |
| test | candidate | 780 | 0.0634 | 1.5531 | 1.9666 | -0.4136 | 0.9949 | 0.4756 | 0.4936 |
| test | PDB-chain macro | 4 | 0.0597 | 1.5878 | 1.9790 | -0.3912 | 1.0000 | 0.5000 | 0.5000 |

输入 <2 Å 且恶化最明显的 PDB 链（完整清单见 `low_input_rmsd_chains.tsv`）：

| split | pdb_id | native_chain_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- | --- | --- |
| val | 7POF | A | 200 | 1.7131 | 2.6593 | -0.9462 |
| val | 8CLR | A | 200 | 1.6419 | 2.0590 | -0.4171 |
| val | 7EEM | A | 200 | 1.0960 | 1.2245 | -0.1285 |
| val | 7M3V | A | 200 | 0.4336 | 0.4273 | 0.0063 |
| val | 7E9I | A | 200 | 0.5582 | 0.4914 | 0.0669 |
| val | 7UCR | A | 200 | 0.7261 | 0.4526 | 0.2736 |
| val | 7QP2 | A | 200 | 0.7905 | 0.5161 | 0.2744 |
| val | 7TD7 | A | 200 | 1.4929 | 1.2097 | 0.2832 |
| val | 7TZU | A | 200 | 1.5818 | 1.2776 | 0.3042 |
| val | 7TZT | A | 200 | 1.5847 | 1.2654 | 0.3193 |
| val | 7TDC | A | 200 | 1.6095 | 0.8804 | 0.7291 |
| val | 7TDB | A | 200 | 1.6423 | 0.8786 | 0.7637 |
| val | 7TDA | A | 200 | 1.6388 | 0.8638 | 0.7750 |
| test | 9OD4 | A | 200 | 1.4889 | 2.2000 | -0.7111 |
| test | 8VT5 | A | 200 | 1.8027 | 2.3491 | -0.5464 |
| test | 23JZ | A | 200 | 1.3054 | 1.6128 | -0.3074 |
| test | 9C6J | A | 200 | 1.7542 | 1.7542 | -0.0000 |

## 原始长度与输入 RMSD 分层（PDB 链宏平均）

| split | group | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | length <=50 nt | 47 | 6.6669 | 5.9506 | 0.7163 | 0.7447 |
| val | length 51-100 nt | 20 | 8.1320 | 7.9156 | 0.2164 | 0.8500 |
| val | length 101-200 nt | 14 | 15.1224 | 14.9500 | 0.1724 | 0.9286 |
| val | length >200 nt | 22 | 10.7552 | 10.7548 | 0.0004 | 0.5000 |
| val | input RMSD [0,2) Å | 13 | 1.2699 | 1.0927 | 0.1772 | 0.7692 |
| val | input RMSD [2,5) Å | 21 | 3.5464 | 3.5735 | -0.0271 | 0.3333 |
| val | input RMSD [5,10) Å | 36 | 7.2168 | 6.9603 | 0.2566 | 0.7778 |
| val | input RMSD [10,20) Å | 25 | 15.2328 | 14.0687 | 1.1641 | 0.9600 |
| val | input RMSD [20,30) Å | 8 | 24.0875 | 24.0449 | 0.0426 | 0.8750 |
| val | input RMSD [30,inf) Å | 0 | nan | nan | nan | nan |
| test | length <=50 nt | 23 | 5.0887 | 4.9460 | 0.1427 | 0.6522 |
| test | length 51-100 nt | 18 | 4.7009 | 4.7179 | -0.0170 | 0.5000 |
| test | length 101-200 nt | 7 | 19.9082 | 19.7841 | 0.1240 | 1.0000 |
| test | length >200 nt | 19 | 19.0493 | 19.0406 | 0.0087 | 0.6842 |
| test | input RMSD [0,2) Å | 4 | 1.5878 | 1.9790 | -0.3912 | 0.0000 |
| test | input RMSD [2,5) Å | 25 | 3.6883 | 3.6784 | 0.0100 | 0.5200 |
| test | input RMSD [5,10) Å | 19 | 7.1138 | 6.8864 | 0.2274 | 0.7895 |
| test | input RMSD [10,20) Å | 3 | 16.1619 | 16.1278 | 0.0341 | 0.6667 |
| test | input RMSD [20,30) Å | 15 | 24.9996 | 24.9394 | 0.0602 | 0.8667 |
| test | input RMSD [30,inf) Å | 1 | 45.7507 | 45.7497 | 0.0010 | 1.0000 |

## 测试集输入 RMSD 分布及去向

前后使用相同分箱，下表直接比较各箱占比。

| unit | bin_a | input_count | refined_count | input_share | refined_share |
| --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 780 | 395 | 0.0634 | 0.0321 |
| candidate | [2,5) | 5441 | 5824 | 0.4420 | 0.4731 |
| candidate | [5,10) | 3069 | 3109 | 0.2493 | 0.2526 |
| candidate | [10,20) | 1154 | 1123 | 0.0937 | 0.0912 |
| candidate | [20,30) | 1666 | 1659 | 0.1353 | 0.1348 |
| candidate | [30,inf) | 200 | 200 | 0.0162 | 0.0162 |
| PDB-chain macro | [0,2) | 4 | 2 | 0.0597 | 0.0299 |
| PDB-chain macro | [2,5) | 25 | 27 | 0.3731 | 0.4030 |
| PDB-chain macro | [5,10) | 19 | 19 | 0.2836 | 0.2836 |
| PDB-chain macro | [10,20) | 3 | 3 | 0.0448 | 0.0448 |
| PDB-chain macro | [20,30) | 15 | 15 | 0.2239 | 0.2239 |
| PDB-chain macro | [30,inf) | 1 | 1 | 0.0149 | 0.0149 |

每行按 refinement 前的 RMSD 分箱，展示同一批结构之后的 RMSD；完整跨箱矩阵见 `rmsd_transitions.tsv`。

| unit | input_bin_a | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 780 | 0.0634 | 1.5531 | 1.9666 | -0.4136 | 0.0051 |
| candidate | [2,5) | 5441 | 0.4420 | 3.6908 | 3.6963 | -0.0055 | 0.4973 |
| candidate | [5,10) | 3069 | 0.2493 | 6.9877 | 6.7094 | 0.2783 | 0.8195 |
| candidate | [10,20) | 1154 | 0.0937 | 15.4610 | 15.3288 | 0.1321 | 0.6993 |
| candidate | [20,30) | 1666 | 0.1353 | 25.0388 | 24.9579 | 0.0809 | 0.7761 |
| candidate | [30,inf) | 200 | 0.0162 | 45.7507 | 45.7497 | 0.0010 | 0.6150 |
| PDB-chain macro | [0,2) | 4 | 0.0597 | 1.5878 | 1.9790 | -0.3912 | 0.0000 |
| PDB-chain macro | [2,5) | 25 | 0.3731 | 3.6883 | 3.6784 | 0.0100 | 0.5200 |
| PDB-chain macro | [5,10) | 19 | 0.2836 | 7.1138 | 6.8864 | 0.2274 | 0.7895 |
| PDB-chain macro | [10,20) | 3 | 0.0448 | 16.1619 | 16.1278 | 0.0341 | 0.6667 |
| PDB-chain macro | [20,30) | 15 | 0.2239 | 24.9996 | 24.9394 | 0.0602 | 0.8667 |
| PDB-chain macro | [30,inf) | 1 | 0.0149 | 45.7507 | 45.7497 | 0.0010 | 1.0000 |

## 物理结构指标（PDB 链宏平均）

数值为训练代码中的未加权键长、位阻、碱基平面 loss，单位 Å²；越小越好。位阻 loss 是 cutoff 内非排除原子对的平均平方重叠，随邻域构成变化。它们不是完整的化学有效性检查。

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 103 | 0.0196 | 0.1444 | 0.1249 | 0.9709 |
| val | clash | 103 | 0.0009 | 0.0364 | 0.0356 | 0.9806 |
| val | plane | 103 | 0.0000 | 0.0191 | 0.0190 | 1.0000 |
| test | bond | 67 | 0.0025 | 0.1107 | 0.1082 | 1.0000 |
| test | clash | 67 | 0.0010 | 0.0279 | 0.0270 | 0.9851 |
| test | plane | 67 | 0.0000 | 0.0128 | 0.0128 | 1.0000 |

输入 RMSD <2 Å 链的物理 loss：

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 13 | 0.0017 | 0.0196 | 0.0179 | 1.0000 |
| val | clash | 13 | 0.0001 | 0.0060 | 0.0060 | 1.0000 |
| val | plane | 13 | 0.0000 | 0.0031 | 0.0031 | 1.0000 |
| test | bond | 4 | 0.0088 | 0.0561 | 0.0473 | 1.0000 |
| test | clash | 4 | 0.0001 | 0.0153 | 0.0152 | 0.7500 |
| test | plane | 4 | 0.0000 | 0.0065 | 0.0065 | 1.0000 |

## FoldBench RNA 单体重叠

测试集按 PDB+预测链匹配 9 个 FoldBench monomer_rna 目标。此处仍是本项目 RMSD，不是 FoldBench lDDT 分数。

| foldbench_target_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- |
| 7sxp-assembly1 | 200 | 3.9829 | 4.6390 | -0.6561 |
| 7wia-assembly1 | 200 | 6.4911 | 6.4608 | 0.0303 |
| 7wii-assembly1 | 200 | 4.2626 | 3.7429 | 0.5197 |
| 7zj4-assembly1 | 200 | 45.7507 | 45.7497 | 0.0010 |
| 8hb8-assembly1 | 200 | 8.7618 | 8.5537 | 0.2081 |
| 8its-assembly1 | 200 | 13.0811 | 13.1019 | -0.0208 |
| 8upt-assembly1 | 200 | 4.0924 | 4.1800 | -0.0876 |
| 8v1h-assembly1 | 200 | 2.7302 | 2.7430 | -0.0128 |
| 9g7c-assembly1 | 138 | 24.7431 | 24.7313 | 0.0118 |

## 文件

`rmsd_comparison.tsv` 包含总览、长度、输入 RMSD 和 <2 Å 子集；`low_input_rmsd_chains.tsv` 为低输入误差链清单；`physical_comparison.tsv` 包含对应候选/链级物理 loss；`rmsd_histogram.tsv`、`rmsd_distribution.tsv` 与 `rmsd_transitions.tsv` 记录前后分布。

评估只覆盖 Data_PT_V2 中实际存在的 .pt 候选；FoldBench 分数需使用原始/精修结构文件由 FoldBench OpenStructure 流程另算，并按预先指定的排名分数选每靶标一个候选。
