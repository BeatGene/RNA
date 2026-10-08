# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=68-step=217419.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7703 Å
- candidate win/worsen rate: 68.47% / 31.50%
- PDB-chain macro mean improvement: 0.3966 Å (95% bootstrap CI 0.1702, 0.6971)
- PDB-chain win/worsen rate: 71.84% / 28.16%
- Spearman(length, improvement): -0.3761
- Spearman(input RMSD, improvement): 0.1407
- Spearman(mean pLDDT, improvement): 0.1799

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9375 | 0.7294 | 0.7660 | 0.7319 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9057 | 0.2263 | 0.9000 | 0.8257 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9755 | 0.1469 | 0.9286 | 0.7942 |
| > 200 | 22 | 3620 | 10.7552 | 10.7557 | -0.0005 | 0.3182 | 0.3677 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0799 | 0.1901 | 0.7692 | 0.7885 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5272 | 0.0192 | 0.2381 | 0.3445 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9265 | 0.2903 | 0.7778 | 0.7171 |
| > 10 | 33 | 5722 | 17.3794 | 16.5452 | 0.8341 | 0.9394 | 0.8128 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.6150 | 10.8327 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 3.0194 | 7.0292 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 12.8216 | 6.4415 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.5085 | 1.5792 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.2571 | 1.2157 | 1.0000 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.6594 | 0.9611 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.8284 | 0.9407 | 0.9800 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.5065 | 0.8719 | 1.0000 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.8730 | 0.7992 | 0.9900 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8767 | 0.7621 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.8949 | 0.7474 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.9477 | 0.7357 | 0.8300 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.8919 | 0.7175 | 1.0000 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.5524 | 0.6726 | 1.0000 |
| 7KUC | A | 16 | 200 | 3.7833 | 3.1644 | 0.6189 | 0.8850 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7POF | A | 14 | 200 | 1.7131 | 2.6405 | -0.9274 | 0.0000 |
| 8FCS | A | 71 | 200 | 3.2936 | 3.9831 | -0.6895 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.3447 | -0.5982 | 0.0000 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.3177 | -0.4161 | 0.0200 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.2446 | -0.3998 | 0.0050 |
| 8CLR | A | 14 | 200 | 1.6419 | 2.0237 | -0.3817 | 0.1850 |
| 8THV | A | 29 | 200 | 5.0821 | 5.4376 | -0.3555 | 0.0550 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.0900 | -0.3375 | 0.3400 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.5738 | -0.2870 | 0.0750 |
| 8TNS | A | 24 | 200 | 11.2120 | 11.3147 | -0.1027 | 0.0650 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6183 | -0.0819 | 0.3250 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.4699 | -0.0696 | 0.0700 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.1387 | -0.0427 | 0.2900 |
| 7XSN | N | 388 | 200 | 3.6488 | 3.6533 | -0.0044 | 0.0300 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.2669 | -0.0034 | 0.3250 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
