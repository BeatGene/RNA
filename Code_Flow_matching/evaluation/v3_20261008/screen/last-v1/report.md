# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/last-v1.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7749 Å
- candidate win/worsen rate: 68.00% / 31.93%
- PDB-chain macro mean improvement: 0.3924 Å (95% bootstrap CI 0.1543, 0.7113)
- PDB-chain win/worsen rate: 73.79% / 26.21%
- Spearman(length, improvement): -0.3332
- Spearman(input RMSD, improvement): 0.1708
- Spearman(mean pLDDT, improvement): 0.1468

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9506 | 0.7162 | 0.7447 | 0.6926 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9156 | 0.2163 | 0.8500 | 0.7917 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9499 | 0.1725 | 0.9286 | 0.7507 |
| > 200 | 22 | 3620 | 10.7552 | 10.7547 | 0.0005 | 0.5000 | 0.5361 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0927 | 0.1773 | 0.7692 | 0.7373 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5734 | -0.0270 | 0.3333 | 0.3921 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9602 | 0.2566 | 0.7778 | 0.7113 |
| > 10 | 33 | 5722 | 17.3794 | 16.4872 | 0.8921 | 0.9394 | 0.8262 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.4557 | 10.9920 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 11.9450 | 7.3181 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.9084 | 7.1402 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.6729 | 1.4148 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.0639 | 1.4090 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7015 | 1.0676 | 0.9750 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.6307 | 1.0415 | 0.9850 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.3905 | 0.9879 | 1.0000 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.6412 | 0.9794 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.8304 | 0.8530 | 0.8350 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8640 | 0.7748 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.8786 | 0.7637 | 1.0000 |
| 7KUB | A | 60 | 200 | 18.2282 | 17.4856 | 0.7427 | 0.9950 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.4905 | 0.7345 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.8798 | 0.7297 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 200 | 3.2936 | 4.3119 | -1.0183 | 0.0000 |
| 7POF | A | 14 | 200 | 1.7131 | 2.6590 | -0.9459 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.6592 | -0.9128 | 0.0000 |
| 8THV | A | 29 | 200 | 5.0821 | 5.7027 | -0.6206 | 0.0350 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.4220 | -0.5772 | 0.0050 |
| 8CLR | A | 14 | 200 | 1.6419 | 2.0589 | -0.4170 | 0.2100 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.1331 | -0.3806 | 0.3250 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.6462 | -0.3594 | 0.0400 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.2419 | -0.3403 | 0.0150 |
| 7MKT | A | 23 | 200 | 8.6643 | 8.8670 | -0.2026 | 0.2400 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.5838 | -0.1835 | 0.0050 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.2243 | -0.1283 | 0.0250 |
| 8TNS | A | 24 | 200 | 11.2120 | 11.3380 | -0.1260 | 0.0150 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6590 | -0.1226 | 0.2650 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.3406 | -0.0771 | 0.0650 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
