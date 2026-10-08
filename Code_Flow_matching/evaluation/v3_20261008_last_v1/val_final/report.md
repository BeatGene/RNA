# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/last-v1.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7749 Å
- candidate win/worsen rate: 67.96% / 31.96%
- PDB-chain macro mean improvement: 0.3924 Å (95% bootstrap CI 0.1543, 0.7114)
- PDB-chain win/worsen rate: 73.79% / 26.21%
- Spearman(length, improvement): -0.3331
- Spearman(input RMSD, improvement): 0.1708
- Spearman(mean pLDDT, improvement): 0.1466

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9506 | 0.7163 | 0.7447 | 0.6923 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9156 | 0.2164 | 0.8500 | 0.7917 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9500 | 0.1724 | 0.9286 | 0.7488 |
| > 200 | 22 | 3620 | 10.7552 | 10.7548 | 0.0004 | 0.5000 | 0.5364 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0927 | 0.1772 | 0.7692 | 0.7373 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5735 | -0.0271 | 0.3333 | 0.3926 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9603 | 0.2566 | 0.7778 | 0.7107 |
| > 10 | 33 | 5722 | 17.3794 | 16.4872 | 0.8922 | 0.9394 | 0.8255 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.4556 | 10.9921 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 11.9443 | 7.3188 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.9086 | 7.1400 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.6728 | 1.4149 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.0639 | 1.4090 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7011 | 1.0680 | 0.9750 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.6331 | 1.0391 | 0.9850 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.3906 | 0.9878 | 1.0000 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.6414 | 0.9791 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.8303 | 0.8531 | 0.8350 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8638 | 0.7750 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.8786 | 0.7637 | 1.0000 |
| 7KUB | A | 60 | 200 | 18.2282 | 17.4863 | 0.7420 | 0.9950 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.4906 | 0.7344 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.8804 | 0.7291 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 200 | 3.2936 | 4.3126 | -1.0189 | 0.0000 |
| 7POF | A | 14 | 200 | 1.7131 | 2.6593 | -0.9462 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.6590 | -0.9126 | 0.0000 |
| 8THV | A | 29 | 200 | 5.0821 | 5.7029 | -0.6207 | 0.0350 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.4222 | -0.5774 | 0.0050 |
| 8CLR | A | 14 | 200 | 1.6419 | 2.0590 | -0.4171 | 0.2100 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.1331 | -0.3805 | 0.3250 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.6464 | -0.3596 | 0.0400 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.2419 | -0.3403 | 0.0150 |
| 7MKT | A | 23 | 200 | 8.6643 | 8.8667 | -0.2023 | 0.2400 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.5838 | -0.1835 | 0.0050 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.2245 | -0.1285 | 0.0250 |
| 8TNS | A | 24 | 200 | 11.2120 | 11.3380 | -0.1261 | 0.0200 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6590 | -0.1226 | 0.2600 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.3407 | -0.0771 | 0.0650 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
