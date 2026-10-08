# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=83-step=264684.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7555 Å
- candidate win/worsen rate: 69.33% / 30.61%
- PDB-chain macro mean improvement: 0.4106 Å (95% bootstrap CI 0.1731, 0.7255)
- PDB-chain win/worsen rate: 70.87% / 29.13%
- Spearman(length, improvement): -0.3828
- Spearman(input RMSD, improvement): 0.1610
- Spearman(mean pLDDT, improvement): 0.1828

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.9032 | 0.7637 | 0.7660 | 0.7346 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9189 | 0.2130 | 0.8500 | 0.8065 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9685 | 0.1539 | 0.9286 | 0.7565 |
| > 200 | 22 | 3620 | 10.7552 | 10.7559 | -0.0007 | 0.3182 | 0.4221 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0815 | 0.1884 | 0.7692 | 0.7619 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5339 | 0.0125 | 0.2857 | 0.3910 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9235 | 0.2933 | 0.7222 | 0.7160 |
| > 10 | 33 | 5722 | 17.3794 | 16.4999 | 0.8795 | 0.9394 | 0.8073 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.5484 | 10.8993 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.8586 | 7.1900 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 12.1212 | 7.1419 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.6578 | 1.4299 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.1181 | 1.3547 | 1.0000 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.5074 | 1.1132 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7762 | 0.9929 | 0.9650 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.4739 | 0.9045 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.7849 | 0.8985 | 0.8350 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.7929 | 0.8793 | 0.9850 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8863 | 0.7524 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.9088 | 0.7335 | 1.0000 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.5208 | 0.7042 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.9114 | 0.6981 | 1.0000 |
| 7KUC | A | 16 | 200 | 3.7833 | 3.2004 | 0.5829 | 0.8650 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 200 | 3.2936 | 4.2138 | -0.9201 | 0.0000 |
| 7POF | A | 14 | 200 | 1.7131 | 2.5806 | -0.8675 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.4497 | -0.7033 | 0.0000 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.2734 | -0.4285 | 0.0050 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.1110 | -0.3584 | 0.3100 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.2087 | -0.3071 | 0.0250 |
| 8CLR | A | 14 | 200 | 1.6419 | 1.9402 | -0.2982 | 0.2650 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.5768 | -0.2900 | 0.0650 |
| 8THV | A | 29 | 200 | 5.0821 | 5.2261 | -0.1440 | 0.2600 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.2015 | -0.1056 | 0.0350 |
| 7UME | A | 28 | 200 | 3.5364 | 3.6260 | -0.0896 | 0.3100 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.4629 | -0.0626 | 0.1400 |
| 7UGA | A | 43 | 200 | 7.7713 | 7.7991 | -0.0278 | 0.3600 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.2786 | -0.0150 | 0.2650 |
| 7UQ6 | B | 86 | 200 | 5.2839 | 5.2979 | -0.0140 | 0.3050 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
