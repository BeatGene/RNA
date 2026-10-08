# V3 RNA refinement evaluation

Checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`

主 RMSD 是 native observed atoms 的逐候选 Kabsch 对齐 RMSD，单位 Å；改善量 = 输入 − refinement 后。
以 PDB 链为主要统计单位；同一链有多个候选时，它们彼此相关。分层按输入值固定，避免 refinement 后跨组导致口径改变。RMSD 仅用 native 可观测原子；物理 loss 则按训练定义使用预测结构的全部原子。

## 验证集与测试集总览

| split | unit | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 103 | 9.3504 | 8.9207 | 0.4297 | 0.6699 | 0.3301 |
| val | PDB-chain macro | 103 | 9.3504 | 8.9207 | 0.4297 | 0.6699 | 0.3301 |
| test | candidate | 67 | 10.4270 | 10.3510 | 0.0760 | 0.5970 | 0.3881 |
| test | PDB-chain macro | 67 | 10.4270 | 10.3510 | 0.0760 | 0.5970 | 0.3881 |

## 输入 RMSD <2 Å

比例分母分别为该 split 的全部候选或全部 PDB 链；链按候选的平均输入 RMSD 判定。

| split | unit | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | worsen_rate | damage_ge_0.5_rate | post_rmsd_ge_2_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | candidate | 12 | 0.1165 | 1.1996 | 0.9632 | 0.2364 | 0.2500 | 0.0833 | 0.0833 |
| val | PDB-chain macro | 12 | 0.1165 | 1.1996 | 0.9632 | 0.2364 | 0.2500 | 0.0833 | 0.0833 |
| test | candidate | 3 | 0.0448 | 1.6455 | 1.7798 | -0.1343 | 0.6667 | 0.0000 | 0.0000 |
| test | PDB-chain macro | 3 | 0.0448 | 1.6455 | 1.7798 | -0.1343 | 0.6667 | 0.0000 | 0.0000 |

输入 <2 Å 且恶化最明显的 PDB 链（完整清单见 `low_input_rmsd_chains.tsv`）：

| split | pdb_id | native_chain_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- | --- | --- |
| val | 7POF | A | 1 | 1.7288 | 2.3995 | -0.6707 |
| val | 7EEM | A | 1 | 0.7722 | 1.0041 | -0.2319 |
| val | 7M3V | A | 1 | 0.3946 | 0.3968 | -0.0022 |
| val | 7UCR | A | 1 | 0.4163 | 0.3626 | 0.0537 |
| val | 7E9I | A | 1 | 0.5738 | 0.5142 | 0.0596 |
| val | 7TZT | A | 1 | 1.4219 | 1.1955 | 0.2264 |
| val | 7QP2 | A | 1 | 0.9208 | 0.6930 | 0.2278 |
| val | 7TZU | A | 1 | 1.6865 | 1.2652 | 0.4213 |
| val | 7TD7 | A | 1 | 1.5051 | 1.0837 | 0.4215 |
| val | 7TDB | A | 1 | 1.6518 | 0.9174 | 0.7344 |
| val | 7TDC | A | 1 | 1.5703 | 0.8111 | 0.7592 |
| val | 7TDA | A | 1 | 1.7531 | 0.9153 | 0.8379 |
| test | 23JZ | A | 1 | 1.3739 | 1.5890 | -0.2151 |
| test | 8VT5 | A | 1 | 1.7905 | 1.9782 | -0.1877 |
| test | 9C6J | A | 1 | 1.7721 | 1.7721 | 0.0000 |

## 原始长度与输入 RMSD 分层（PDB 链宏平均）

| split | group | count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | length <=50 nt | 47 | 6.9535 | 6.1582 | 0.7953 | 0.7234 |
| val | length 51-100 nt | 20 | 8.7798 | 8.5700 | 0.2098 | 0.7000 |
| val | length 101-200 nt | 14 | 16.0364 | 15.8457 | 0.1907 | 0.7143 |
| val | length >200 nt | 22 | 10.7349 | 10.7345 | 0.0004 | 0.5000 |
| val | input RMSD [0,2) Å | 12 | 1.1996 | 0.9632 | 0.2364 | 0.7500 |
| val | input RMSD [2,5) Å | 24 | 3.5420 | 3.5036 | 0.0384 | 0.4583 |
| val | input RMSD [5,10) Å | 30 | 7.3055 | 6.9535 | 0.3519 | 0.7333 |
| val | input RMSD [10,20) Å | 25 | 14.3717 | 13.2087 | 1.1630 | 0.6800 |
| val | input RMSD [20,30) Å | 12 | 23.7689 | 23.6970 | 0.0719 | 0.8333 |
| val | input RMSD [30,inf) Å | 0 | nan | nan | nan | nan |
| test | length <=50 nt | 23 | 5.1532 | 4.9622 | 0.1910 | 0.7391 |
| test | length 51-100 nt | 18 | 4.3058 | 4.3025 | 0.0033 | 0.4444 |
| test | length 101-200 nt | 7 | 19.5980 | 19.5385 | 0.0595 | 0.5714 |
| test | length >200 nt | 19 | 19.2313 | 19.2196 | 0.0117 | 0.5789 |
| test | input RMSD [0,2) Å | 3 | 1.6455 | 1.7798 | -0.1343 | 0.0000 |
| test | input RMSD [2,5) Å | 29 | 3.6221 | 3.6178 | 0.0042 | 0.4828 |
| test | input RMSD [5,10) Å | 15 | 6.9395 | 6.6294 | 0.3101 | 0.8000 |
| test | input RMSD [10,20) Å | 5 | 15.5245 | 15.4798 | 0.0447 | 0.6000 |
| test | input RMSD [20,30) Å | 14 | 26.0895 | 26.0544 | 0.0351 | 0.7143 |
| test | input RMSD [30,inf) Å | 1 | 41.6628 | 41.6590 | 0.0038 | 1.0000 |

## 测试集输入 RMSD 分布及去向

前后使用相同分箱，下表直接比较各箱占比。

| unit | bin_a | input_count | refined_count | input_share | refined_share |
| --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 3 | 3 | 0.0448 | 0.0448 |
| candidate | [2,5) | 29 | 30 | 0.4328 | 0.4478 |
| candidate | [5,10) | 15 | 14 | 0.2239 | 0.2090 |
| candidate | [10,20) | 5 | 5 | 0.0746 | 0.0746 |
| candidate | [20,30) | 14 | 14 | 0.2090 | 0.2090 |
| candidate | [30,inf) | 1 | 1 | 0.0149 | 0.0149 |
| PDB-chain macro | [0,2) | 3 | 3 | 0.0448 | 0.0448 |
| PDB-chain macro | [2,5) | 29 | 30 | 0.4328 | 0.4478 |
| PDB-chain macro | [5,10) | 15 | 14 | 0.2239 | 0.2090 |
| PDB-chain macro | [10,20) | 5 | 5 | 0.0746 | 0.0746 |
| PDB-chain macro | [20,30) | 14 | 14 | 0.2090 | 0.2090 |
| PDB-chain macro | [30,inf) | 1 | 1 | 0.0149 | 0.0149 |

每行按 refinement 前的 RMSD 分箱，展示同一批结构之后的 RMSD；完整跨箱矩阵见 `rmsd_transitions.tsv`。

| unit | input_bin_a | count | share | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | [0,2) | 3 | 0.0448 | 1.6455 | 1.7798 | -0.1343 | 0.0000 |
| candidate | [2,5) | 29 | 0.4328 | 3.6221 | 3.6178 | 0.0042 | 0.4828 |
| candidate | [5,10) | 15 | 0.2239 | 6.9395 | 6.6294 | 0.3101 | 0.8000 |
| candidate | [10,20) | 5 | 0.0746 | 15.5245 | 15.4798 | 0.0447 | 0.6000 |
| candidate | [20,30) | 14 | 0.2090 | 26.0895 | 26.0544 | 0.0351 | 0.7143 |
| candidate | [30,inf) | 1 | 0.0149 | 41.6628 | 41.6590 | 0.0038 | 1.0000 |
| PDB-chain macro | [0,2) | 3 | 0.0448 | 1.6455 | 1.7798 | -0.1343 | 0.0000 |
| PDB-chain macro | [2,5) | 29 | 0.4328 | 3.6221 | 3.6178 | 0.0042 | 0.4828 |
| PDB-chain macro | [5,10) | 15 | 0.2239 | 6.9395 | 6.6294 | 0.3101 | 0.8000 |
| PDB-chain macro | [10,20) | 5 | 0.0746 | 15.5245 | 15.4798 | 0.0447 | 0.6000 |
| PDB-chain macro | [20,30) | 14 | 0.2090 | 26.0895 | 26.0544 | 0.0351 | 0.7143 |
| PDB-chain macro | [30,inf) | 1 | 0.0149 | 41.6628 | 41.6590 | 0.0038 | 1.0000 |

## 物理结构指标（PDB 链宏平均）

数值为训练代码中的未加权键长、位阻、碱基平面 loss，单位 Å²；越小越好。位阻 loss 是 cutoff 内非排除原子对的平均平方重叠，随邻域构成变化。它们不是完整的化学有效性检查。

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 103 | 0.0189 | 0.1169 | 0.0980 | 0.9612 |
| val | clash | 103 | 0.0007 | 0.0336 | 0.0329 | 0.8932 |
| val | plane | 103 | 0.0000 | 0.0152 | 0.0152 | 1.0000 |
| test | bond | 67 | 0.0042 | 0.0803 | 0.0760 | 0.9851 |
| test | clash | 67 | 0.0010 | 0.0266 | 0.0256 | 0.9254 |
| test | plane | 67 | 0.0000 | 0.0098 | 0.0098 | 1.0000 |

输入 RMSD <2 Å 链的物理 loss：

| split | loss | count | input_loss_a2 | refined_loss_a2 | change_after_minus_before_a2 | worsen_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | bond | 12 | 0.0002 | 0.0159 | 0.0157 | 1.0000 |
| val | clash | 12 | 0.0001 | 0.0037 | 0.0037 | 0.9167 |
| val | plane | 12 | 0.0000 | 0.0029 | 0.0029 | 1.0000 |
| test | bond | 3 | 0.0001 | 0.0352 | 0.0350 | 1.0000 |
| test | clash | 3 | 0.0001 | 0.0043 | 0.0043 | 1.0000 |
| test | plane | 3 | 0.0000 | 0.0060 | 0.0060 | 1.0000 |

## FoldBench RNA 单体重叠

测试集按 PDB+预测链匹配 9 个 FoldBench monomer_rna 目标。此处仍是本项目 RMSD，不是 FoldBench lDDT 分数。

| foldbench_target_id | sample_count | input_rmsd_mean_a | refined_rmsd_mean_a | improvement_mean_a |
| --- | --- | --- | --- | --- |
| 7sxp-assembly1 | 1 | 4.1838 | 4.5097 | -0.3259 |
| 7wia-assembly1 | 1 | 9.1782 | 9.1542 | 0.0240 |
| 7wii-assembly1 | 1 | 4.4622 | 4.2132 | 0.2491 |
| 7zj4-assembly1 | 1 | 41.6628 | 41.6590 | 0.0038 |
| 8hb8-assembly1 | 1 | 6.6685 | 6.6409 | 0.0277 |
| 8its-assembly1 | 1 | 13.0042 | 12.9918 | 0.0124 |
| 8upt-assembly1 | 1 | 4.0872 | 4.1135 | -0.0263 |
| 8v1h-assembly1 | 1 | 2.5710 | 2.5754 | -0.0044 |
| 9g7c-assembly1 | 1 | 29.7998 | 29.8034 | -0.0036 |

## 文件

`rmsd_comparison.tsv` 包含总览、长度、输入 RMSD 和 <2 Å 子集；`low_input_rmsd_chains.tsv` 为低输入误差链清单；`physical_comparison.tsv` 包含对应候选/链级物理 loss；`rmsd_histogram.tsv`、`rmsd_distribution.tsv` 与 `rmsd_transitions.tsv` 记录前后分布。

评估数据目录：`/storage9920/home/tinghao.xia/Data_PT_V2_rank1`。只覆盖该目录中实际存在的 .pt 候选；FoldBench 分数需使用原始/精修结构文件由 FoldBench OpenStructure 流程另算。
