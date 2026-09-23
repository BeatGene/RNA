# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v2_residual_mobility/epoch=12-step=39442.ckpt`
- candidates: 21322
- PDB chains: 111
- candidate-level mean RMSD: 7.9468 → 7.6501 Å
- candidate win/worsen rate: 79.26% / 20.72%
- PDB-chain macro mean improvement: 0.2853 Å (95% bootstrap CI 0.0927, 0.5569)
- PDB-chain win/worsen rate: 81.98% / 18.02%
- Spearman(length, improvement): -0.3508
- Spearman(input RMSD, improvement): 0.0067
- Spearman(mean pLDDT, improvement): 0.2783

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 54 | 10800 | 6.3838 | 5.9055 | 0.4783 | 0.7407 | 0.7189 |
| 51-100 | 21 | 4200 | 8.1619 | 7.9692 | 0.1927 | 0.8571 | 0.8279 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9959 | 0.1264 | 0.9286 | 0.8342 |
| > 200 | 22 | 3620 | 10.7552 | 10.7539 | 0.0013 | 0.9091 | 0.8711 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0530 | 0.2169 | 0.7692 | 0.8042 |
| 2-5 | 27 | 5400 | 3.6802 | 3.6161 | 0.0640 | 0.7407 | 0.7093 |
| 5-10 | 38 | 7600 | 7.2384 | 7.0588 | 0.1795 | 0.8421 | 0.7988 |
| > 10 | 33 | 5722 | 17.3794 | 16.7642 | 0.6152 | 0.8788 | 0.8209 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.6326 | 10.8151 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.8947 | 7.1539 | 1.0000 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.0307 | 1.1943 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.3555 | 1.1174 | 1.0000 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.5075 | 1.1131 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 7.0767 | 1.0110 | 1.0000 |
| 7KUC | A | 16 | 200 | 3.7833 | 2.9073 | 0.8760 | 0.9700 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.5767 | 0.8017 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 8.0092 | 0.7599 | 0.9600 |
| 7MKT | A | 23 | 200 | 8.6643 | 7.9339 | 0.7304 | 0.9900 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.9463 | 0.7259 | 0.9900 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.9617 | 0.7217 | 0.8750 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.9510 | 0.6878 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.9667 | 0.6756 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.9461 | 0.6634 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8CQ1 | A | 44 | 200 | 2.8448 | 4.2526 | -1.4078 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.6393 | -0.8929 | 0.0000 |
| 7POF | A | 14 | 200 | 1.7131 | 2.4173 | -0.7042 | 0.0000 |
| 7SXP | A | 22 | 200 | 3.9829 | 4.3747 | -0.3918 | 0.0000 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.1212 | -0.3687 | 0.3950 |
| 8THV | A | 29 | 200 | 5.0821 | 5.4098 | -0.3277 | 0.0600 |
| 7EDN | A | 23 | 200 | 19.2631 | 19.5602 | -0.2971 | 0.0450 |
| 7RWR | A | 38 | 200 | 6.7615 | 7.0130 | -0.2515 | 0.0650 |
| 8CLR | A | 14 | 200 | 1.6419 | 1.8721 | -0.2302 | 0.4000 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.5026 | -0.2158 | 0.0300 |
| 7UGA | A | 43 | 200 | 7.7713 | 7.8798 | -0.1085 | 0.2200 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.1526 | -0.0567 | 0.1100 |
| 7V06 | A | 43 | 200 | 5.1697 | 5.1909 | -0.0212 | 0.6700 |
| 7UMC | A | 70 | 200 | 9.5498 | 9.5689 | -0.0191 | 0.2100 |
| 7KUB | A | 60 | 200 | 18.2282 | 18.2464 | -0.0181 | 0.3550 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
