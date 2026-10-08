# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=88-step=280439.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7589 Å
- candidate win/worsen rate: 69.56% / 30.37%
- PDB-chain macro mean improvement: 0.4075 Å (95% bootstrap CI 0.1675, 0.7216)
- PDB-chain win/worsen rate: 74.76% / 25.24%
- Spearman(length, improvement): -0.3774
- Spearman(input RMSD, improvement): 0.1730
- Spearman(mean pLDDT, improvement): 0.1877

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9141 | 0.7527 | 0.7872 | 0.7245 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9259 | 0.2061 | 0.8500 | 0.7997 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9457 | 0.1767 | 0.9286 | 0.7704 |
| > 200 | 22 | 3620 | 10.7552 | 10.7550 | 0.0002 | 0.4545 | 0.4755 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0888 | 0.1811 | 0.7692 | 0.7473 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5419 | 0.0045 | 0.3810 | 0.4152 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9455 | 0.2714 | 0.7778 | 0.7236 |
| > 10 | 33 | 5722 | 17.3794 | 16.4776 | 0.9018 | 0.9394 | 0.8123 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.6485 | 10.7992 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 11.4052 | 7.8579 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.9909 | 7.0577 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.0686 | 1.4042 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.6989 | 1.3888 | 1.0000 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.6055 | 1.0667 | 0.9850 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7027 | 1.0664 | 0.9750 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.5656 | 1.0549 | 1.0000 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.4632 | 0.9152 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.8218 | 0.8616 | 0.8300 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8905 | 0.7483 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.9060 | 0.7363 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.9037 | 0.7058 | 1.0000 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.5483 | 0.6767 | 1.0000 |
| 7KUC | A | 16 | 200 | 3.7833 | 3.1866 | 0.5967 | 0.8650 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7POF | A | 14 | 200 | 1.7131 | 2.6045 | -0.8914 | 0.0000 |
| 8FCS | A | 71 | 200 | 3.2936 | 4.1299 | -0.8363 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.5016 | -0.7552 | 0.0000 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.3076 | -0.4628 | 0.0000 |
| 8THV | A | 29 | 200 | 5.0821 | 5.5231 | -0.4410 | 0.0700 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.6856 | -0.3988 | 0.0250 |
| 8CLR | A | 14 | 200 | 1.6419 | 2.0402 | -0.3983 | 0.1700 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.0527 | -0.3002 | 0.3600 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.1613 | -0.2597 | 0.0150 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.5229 | -0.1226 | 0.0550 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.2020 | -0.1061 | 0.0250 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6417 | -0.1053 | 0.2300 |
| 7UQ6 | B | 86 | 200 | 5.2839 | 5.3053 | -0.0214 | 0.2900 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.2794 | -0.0159 | 0.2850 |
| 8BTZ | A | 238 | 200 | 18.2948 | 18.2978 | -0.0030 | 0.0100 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
