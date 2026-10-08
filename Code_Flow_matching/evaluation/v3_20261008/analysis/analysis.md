# V3 RNA refinement evaluation

Checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`

主 RMSD 是 native observed atoms 的逐候选 Kabsch 对齐 RMSD，单位 Å；改善量 = 输入 − refinement 后。
同一链的候选相关，链层面的宏平均是主要统计单位。分层按输入值固定，避免 refinement 后跨组导致口径改变。RMSD 仅用 native 可观测原子；物理 loss 则按训练定义使用预测结构的全部原子。

## 验证集与测试集总览

| split | unit | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 19722 | 8.1844 | 7.7522 | 0.4322 | 0.6895 | 0.3104 |
| val | PDB-chain macro | 103 | 8.9739 | 8.5599 | 0.4140 | 0.7476 | 0.2524 |
| test | candidate | 12310 | 9.0532 | 8.9726 | 0.0806 | 0.6582 | 0.3412 |
| test | PDB-chain macro | 67 | 10.4918 | 10.4172 | 0.0747 | 0.6866 | 0.3134 |

## 输入 RMSD <2 Å

比例分母分别为该 split 的全部候选或全部 PDB 链；链按候选的平均输入 RMSD 判定。

| split | unit | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | worsen_rate | damage_ge_0.5_rate | post_rmsd_ge_2_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 2630 | 0.1334 | 1.2448 | 1.1065 | 0.1383 | 0.2700 | 0.1468 | 0.1171 |
| val | PDB-chain macro | 13 | 0.1262 | 1.2699 | 1.0804 | 0.1896 | 0.2308 | 0.0769 | 0.0769 |
| test | candidate | 780 | 0.0634 | 1.5531 | 1.9125 | -0.3594 | 0.7731 | 0.2718 | 0.3359 |
| test | PDB-chain macro | 4 | 0.0597 | 1.5878 | 1.9250 | -0.3372 | 0.7500 | 0.2500 | 0.2500 |

输入 <2 Å 且恶化最明显的 PDB 链（完整清单见 `low_input_rmsd_chains.tsv`）：

| split | pdb_id | native_chain_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- | --- | --- |
| val | 7POF | A | 200 | 1.7131 | 2.5023 | -0.7892 |
| val | 8CLR | A | 200 | 1.6419 | 1.9886 | -0.3467 |
| val | 7EEM | A | 200 | 1.0960 | 1.1698 | -0.0738 |
| val | 7M3V | A | 200 | 0.4336 | 0.4214 | 0.0121 |
| val | 7E9I | A | 200 | 0.5582 | 0.4985 | 0.0598 |
| val | 7QP2 | A | 200 | 0.7905 | 0.5784 | 0.2121 |
| val | 7UCR | A | 200 | 0.7261 | 0.4816 | 0.2445 |
| val | 7TD7 | A | 200 | 1.4929 | 1.1822 | 0.3107 |
| val | 7TZU | A | 200 | 1.5818 | 1.2708 | 0.3109 |
| val | 7TZT | A | 200 | 1.5847 | 1.2591 | 0.3255 |
| val | 7TDC | A | 200 | 1.6095 | 0.9057 | 0.7038 |
| val | 7TDB | A | 200 | 1.6423 | 0.9052 | 0.7372 |
| val | 7TDA | A | 200 | 1.6388 | 0.8815 | 0.7573 |
| test | 9OD4 | A | 200 | 1.4889 | 2.3103 | -0.8214 |
| test | 23JZ | A | 200 | 1.3054 | 1.6478 | -0.3423 |
| test | 8VT5 | A | 200 | 1.8027 | 1.9878 | -0.1850 |
| test | 9C6J | A | 200 | 1.7542 | 1.7541 | 0.0000 |

## 原始长度与输入 RMSD 分层（PDB 链宏平均）

| split | group | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | length <=50 nt | 47 | 6.6669 | 5.8931 | 0.7737 | 0.7872 |
| val | length 51-100 nt | 20 | 8.1320 | 7.9298 | 0.2021 | 0.8500 |
| val | length 101-200 nt | 14 | 15.1224 | 14.9631 | 0.1593 | 0.9286 |
| val | length >200 nt | 22 | 10.7552 | 10.7551 | 0.0001 | 0.4545 |
| val | input RMSD [0,2) Å | 13 | 1.2699 | 1.0804 | 0.1896 | 0.7692 |
| val | input RMSD [2,5) Å | 21 | 3.5464 | 3.5552 | -0.0088 | 0.3810 |
| val | input RMSD [5,10) Å | 36 | 7.2168 | 6.9074 | 0.3094 | 0.7500 |
| val | input RMSD [10,20) Å | 25 | 15.2328 | 14.0820 | 1.1508 | 0.9600 |
| val | input RMSD [20,30) Å | 8 | 24.0875 | 24.0311 | 0.0564 | 1.0000 |
| val | input RMSD [30,inf) Å | 0 | nan | nan | nan | nan |
| test | length <=50 nt | 23 | 5.0887 | 4.9219 | 0.1669 | 0.7391 |
| test | length 51-100 nt | 18 | 4.7009 | 4.6770 | 0.0239 | 0.5000 |
| test | length 101-200 nt | 7 | 19.9082 | 19.8240 | 0.0842 | 0.8571 |
| test | length >200 nt | 19 | 19.0493 | 19.0417 | 0.0076 | 0.7368 |
| test | input RMSD [0,2) Å | 4 | 1.5878 | 1.9250 | -0.3372 | 0.2500 |
| test | input RMSD [2,5) Å | 25 | 3.6883 | 3.6614 | 0.0269 | 0.5200 |
| test | input RMSD [5,10) Å | 19 | 7.1138 | 6.8566 | 0.2573 | 0.7895 |
| test | input RMSD [10,20) Å | 3 | 16.1619 | 16.1224 | 0.0395 | 1.0000 |
| test | input RMSD [20,30) Å | 15 | 24.9996 | 24.9548 | 0.0448 | 0.8667 |
| test | input RMSD [30,inf) Å | 1 | 45.7507 | 45.7500 | 0.0007 | 1.0000 |

## 测试集输入 RMSD 分布及去向

前后使用相同分箱，下表直接比较各箱占比。

| unit | bin_a | input_count | refined_count | input_share | refined_share |
| --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 780 | 518 | 0.0634 | 0.0421 |
| candidate | [2,5) | 5441 | 6018 | 0.4420 | 0.4889 |
| candidate | [5,10) | 3069 | 2794 | 0.2493 | 0.2270 |
| candidate | [10,20) | 1154 | 1121 | 0.0937 | 0.0911 |
| candidate | [20,30) | 1666 | 1659 | 0.1353 | 0.1348 |
| candidate | [30,inf) | 200 | 200 | 0.0162 | 0.0162 |
| PDB-chain macro | [0,2) | 4 | 3 | 0.0597 | 0.0448 |
| PDB-chain macro | [2,5) | 25 | 28 | 0.3731 | 0.4179 |
| PDB-chain macro | [5,10) | 19 | 17 | 0.2836 | 0.2537 |
| PDB-chain macro | [10,20) | 3 | 3 | 0.0448 | 0.0448 |
| PDB-chain macro | [20,30) | 15 | 15 | 0.2239 | 0.2239 |
| PDB-chain macro | [30,inf) | 1 | 1 | 0.0149 | 0.0149 |

每行按 refinement 前的 RMSD 分箱，展示同一批结构之后的 RMSD；完整跨箱矩阵见 `rmsd_transitions.tsv`。

| unit | input_bin_a | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 780 | 0.0634 | 1.5531 | 1.9125 | -0.3594 | 0.2205 |
| candidate | [2,5) | 5441 | 0.4420 | 3.6908 | 3.6744 | 0.0164 | 0.5365 |
| candidate | [5,10) | 3069 | 0.2493 | 6.9877 | 6.6860 | 0.3016 | 0.8628 |
| candidate | [10,20) | 1154 | 0.0937 | 15.4610 | 15.3195 | 0.1415 | 0.8258 |
| candidate | [20,30) | 1666 | 0.1353 | 25.0388 | 24.9821 | 0.0567 | 0.7815 |
| candidate | [30,inf) | 200 | 0.0162 | 45.7507 | 45.7500 | 0.0007 | 0.5400 |
| PDB-chain macro | [0,2) | 4 | 0.0597 | 1.5878 | 1.9250 | -0.3372 | 0.2500 |
| PDB-chain macro | [2,5) | 25 | 0.3731 | 3.6883 | 3.6614 | 0.0269 | 0.5200 |
| PDB-chain macro | [5,10) | 19 | 0.2836 | 7.1138 | 6.8566 | 0.2573 | 0.7895 |
| PDB-chain macro | [10,20) | 3 | 0.0448 | 16.1619 | 16.1224 | 0.0395 | 1.0000 |
| PDB-chain macro | [20,30) | 15 | 0.2239 | 24.9996 | 24.9548 | 0.0448 | 0.8667 |
| PDB-chain macro | [30,inf) | 1 | 0.0149 | 45.7507 | 45.7500 | 0.0007 | 1.0000 |

## 物理结构指标（PDB 链宏平均）

数值为训练代码中的未加权键长、位阻、碱基平面 loss，单位 Å²；越小越好。位阻 loss 是 cutoff 内非排除原子对的平均平方重叠，随邻域构成变化。它们不是完整的化学有效性检查。

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 103 | 0.0196 | 0.1289 | 0.1093 | 0.9709 |
| val | clash | 103 | 0.0009 | 0.0366 | 0.0358 | 0.9417 |
| val | plane | 103 | 0.0000 | 0.0179 | 0.0179 | 1.0000 |
| test | bond | 67 | 0.0025 | 0.0902 | 0.0877 | 1.0000 |
| test | clash | 67 | 0.0010 | 0.0266 | 0.0256 | 0.9552 |
| test | plane | 67 | 0.0000 | 0.0111 | 0.0111 | 1.0000 |

输入 RMSD <2 Å 链的物理 loss：

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 13 | 0.0017 | 0.0200 | 0.0183 | 1.0000 |
| val | clash | 13 | 0.0001 | 0.0050 | 0.0049 | 0.9231 |
| val | plane | 13 | 0.0000 | 0.0034 | 0.0034 | 1.0000 |
| test | bond | 4 | 0.0088 | 0.0488 | 0.0400 | 1.0000 |
| test | clash | 4 | 0.0001 | 0.0146 | 0.0145 | 0.7500 |
| test | plane | 4 | 0.0000 | 0.0061 | 0.0061 | 1.0000 |

## FoldBench RNA 单体重叠

测试集按 PDB+预测链匹配 9 个 FoldBench monomer_rna 目标。此处仍是本项目 RMSD，不是 FoldBench lDDT 分数。

| foldbench_target_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- |
| 7sxp-assembly1 | 200 | 3.9829 | 4.3647 | -0.3818 |
| 7wia-assembly1 | 200 | 6.4911 | 6.4699 | 0.0212 |
| 7wii-assembly1 | 200 | 4.2626 | 3.8978 | 0.3648 |
| 7zj4-assembly1 | 200 | 45.7507 | 45.7500 | 0.0007 |
| 8hb8-assembly1 | 200 | 8.7618 | 8.6477 | 0.1141 |
| 8its-assembly1 | 200 | 13.0811 | 13.0365 | 0.0446 |
| 8upt-assembly1 | 200 | 4.0924 | 4.0967 | -0.0043 |
| 8v1h-assembly1 | 200 | 2.7302 | 2.7356 | -0.0054 |
| 9g7c-assembly1 | 138 | 24.7431 | 24.7388 | 0.0044 |

## 文件

`rmsd_comparison.tsv` 包含总览、长度、输入 RMSD 和 <2 Å 子集；`low_input_rmsd_chains.tsv` 为低输入误差链清单；`physical_comparison.tsv` 包含对应候选/链级物理 loss；`rmsd_histogram.tsv`、`rmsd_distribution.tsv` 与 `rmsd_transitions.tsv` 记录前后分布。

评估只覆盖 Data_PT_V2 中实际存在的 .pt 候选；FoldBench 分数需使用原始/精修结构文件由 FoldBench OpenStructure 流程另算，并按预先指定的排名分数选每靶标一个候选。
