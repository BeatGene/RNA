# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 7.7522 Å
- candidate win/worsen rate: 68.97% / 31.01%
- PDB-chain macro mean improvement: 0.4140 Å (95% bootstrap CI 0.1785, 0.7254)
- PDB-chain win/worsen rate: 74.76% / 25.24%
- Spearman(length, improvement): -0.3870
- Spearman(input RMSD, improvement): 0.1713
- Spearman(mean pLDDT, improvement): 0.1631

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 5.8931 | 0.7737 | 0.7872 | 0.7350 |
| 51-100 | 20 | 4000 | 8.1320 | 7.9298 | 0.2022 | 0.8500 | 0.8102 |
| 101-200 | 14 | 2702 | 15.1224 | 14.9632 | 0.1591 | 0.9286 | 0.7344 |
| > 200 | 22 | 3620 | 10.7552 | 10.7551 | 0.0001 | 0.4545 | 0.4528 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.0804 | 0.1895 | 0.7692 | 0.7565 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5552 | -0.0088 | 0.3810 | 0.3902 |
| 5-10 | 36 | 7200 | 7.2168 | 6.9074 | 0.3094 | 0.7500 | 0.7339 |
| > 10 | 33 | 5722 | 17.3794 | 16.4939 | 0.8855 | 0.9697 | 0.8043 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 200 | 11.4477 | 0.5219 | 10.9258 | 1.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 12.0721 | 7.1910 | 1.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 2.9399 | 7.1087 | 1.0000 |
| 7PS8 | A | 23 | 200 | 8.0877 | 6.2439 | 1.8438 | 1.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 4.3595 | 1.1133 | 1.0000 |
| 6XKN | A | 101 | 200 | 8.7691 | 7.7065 | 1.0625 | 0.9750 |
| 6XKO | A | 101 | 200 | 8.6722 | 7.7569 | 0.9153 | 0.9850 |
| 8I44 | A | 19 | 200 | 4.6205 | 3.7102 | 0.9103 | 1.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 9.7987 | 0.8847 | 0.8350 |
| 7TDA | A | 83 | 200 | 1.6388 | 0.8813 | 0.7575 | 1.0000 |
| 8I45 | A | 19 | 200 | 3.3784 | 2.6233 | 0.7551 | 1.0000 |
| 7TDB | A | 83 | 200 | 1.6423 | 0.9051 | 0.7372 | 1.0000 |
| 8I46 | A | 19 | 200 | 4.2250 | 3.5011 | 0.7239 | 1.0000 |
| 7TDC | A | 83 | 200 | 1.6095 | 0.9057 | 0.7037 | 1.0000 |
| 7QA2 | A | 22 | 200 | 11.1835 | 10.5385 | 0.6451 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 200 | 3.2936 | 4.3914 | -1.0978 | 0.0000 |
| 7POF | A | 14 | 200 | 1.7131 | 2.5024 | -0.7893 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 3.3540 | -0.6076 | 0.0000 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 3.2824 | -0.4376 | 0.0050 |
| 7KUD | A | 13 | 200 | 2.7526 | 3.1317 | -0.3791 | 0.3250 |
| 8CLR | A | 14 | 200 | 1.6419 | 1.9887 | -0.3468 | 0.2400 |
| 8THV | A | 29 | 200 | 5.0821 | 5.4147 | -0.3326 | 0.0800 |
| 7FHI | A | 20 | 200 | 3.9016 | 4.2099 | -0.3083 | 0.0200 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.4954 | -0.2086 | 0.0900 |
| 7UZ0 | A | 87 | 200 | 2.4003 | 2.5570 | -0.1567 | 0.0150 |
| 8TNS | A | 24 | 200 | 11.2120 | 11.3487 | -0.1367 | 0.0100 |
| 7EEM | A | 27 | 200 | 1.0960 | 1.1700 | -0.0741 | 0.1250 |
| 7MLW | F | 128 | 200 | 2.2636 | 2.3227 | -0.0592 | 0.1950 |
| 7UQ6 | B | 86 | 200 | 5.2839 | 5.3073 | -0.0234 | 0.3250 |
| 7XSN | N | 388 | 200 | 3.6488 | 3.6509 | -0.0020 | 0.0850 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
