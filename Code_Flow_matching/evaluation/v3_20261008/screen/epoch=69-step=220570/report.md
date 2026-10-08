# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=69-step=220570.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7633 Å
- candidate win/worsen rate: 69.87% / 30.09%
- PDB-chain macro mean improvement: 0.4033 Å (95% bootstrap CI 0.1665, 0.7170)
- PDB-chain win/worsen rate: 71.84% / 28.16%
- Spearman(length, improvement): -0.3441
- Spearman(input RMSD, improvement): 0.1705
- Spearman(mean pLDDT, improvement): 0.1425

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9248 | 0.7421 | 0.7447 | 0.7199 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9136 | 0.2184 | 0.8500 | 0.8097 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9583 | 0.1641 | 0.9286 | 0.7630 |
| > 200 | 22 | 3620 | 10.7552 | 10.7551 | 0.0001 | 0.4091 | 0.5009 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0893 | 0.1806 | 0.7692 | 0.7415 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5412 | 0.0052 | 0.3333 | 0.3936 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9359 | 0.2809 | 0.7222 | 0.7358 |
| > 10 | 33 | 5722 | 17.3794 | 16.5014 | 0.8780 | 0.9394 | 0.8284 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.6191 | 10.8286 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.7962 | 7.2524 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 12.1900 | 7.0731 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.5229 | 1.5648 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.2573 | 1.2155 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7568 | 1.0123 | 0.9750 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.7021 | 0.9700 | 0.9850 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.6683 | 0.9522 | 1.0000 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.5275 | 0.8509 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.8542 | 0.8292 | 0.8300 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.4657 | 0.7593 | 1.0000 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8818 | 0.7570 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.8991 | 0.7433 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.8976 | 0.7119 | 1.0000 |
| 7KUC | A | 16 | 200 | 3.7833 | 3.1625 | 0.6209 | 0.9000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7POF | A | 14 | 200 | 1.7131 | 2.6166 | -0.9035 | 0.0000 |
| 8FCS | A | 71 | 200 | 3.2936 | 4.1771 | -0.8834 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.4383 | -0.6919 | 0.0000 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.3199 | -0.4751 | 0.0050 |
| 8THV | A | 29 | 200 | 5.0821 | 5.4901 | -0.4080 | 0.1150 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.6572 | -0.3704 | 0.0250 |
| 8CLR | A | 14 | 200 | 1.6419 | 1.9963 | -0.3543 | 0.2000 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.0412 | -0.2887 | 0.3450 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.1517 | -0.2501 | 0.0200 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.5299 | -0.1297 | 0.0050 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.1993 | -0.1034 | 0.0400 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6347 | -0.0983 | 0.2950 |
| 7MKT | A | 23 | 200 | 8.6643 | 8.7415 | -0.0771 | 0.3800 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.2977 | -0.0342 | 0.2500 |
| 8TNS | A | 24 | 200 | 11.2120 | 11.2385 | -0.0265 | 0.4000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
