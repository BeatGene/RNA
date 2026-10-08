# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/last.ckpt`
- candidates: 19722
- PDB chains: 103
- candidate-level mean RMSD: 8.1844 → 8.1887 Å
- candidate win/worsen rate: 0.32% / 99.67%
- PDB-chain macro mean improvement: -0.0043 Å (95% bootstrap CI -0.0049, -0.0039)
- PDB-chain win/worsen rate: 0.00% / 100.00%
- Spearman(length, improvement): 0.4154
- Spearman(input RMSD, improvement): -0.1296
- Spearman(mean pLDDT, improvement): 0.0621

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 9400 | 6.6669 | 6.6724 | -0.0056 | 0.0000 | 0.0047 |
| 51-100 | 20 | 4000 | 8.1320 | 8.1358 | -0.0038 | 0.0000 | 0.0045 |
| 101-200 | 14 | 2702 | 15.1224 | 15.1256 | -0.0032 | 0.0000 | 0.0007 |
| > 200 | 22 | 3620 | 10.7552 | 10.7581 | -0.0029 | 0.0000 | 0.0000 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 13 | 2600 | 1.2699 | 1.2736 | -0.0037 | 0.0000 | 0.0008 |
| 2-5 | 21 | 4200 | 3.5464 | 3.5499 | -0.0035 | 0.0000 | 0.0012 |
| 5-10 | 36 | 7200 | 7.2168 | 7.2210 | -0.0042 | 0.0000 | 0.0054 |
| > 10 | 33 | 5722 | 17.3794 | 17.3847 | -0.0053 | 0.0000 | 0.0027 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8I46 | A | 19 | 200 | 4.2250 | 4.2256 | -0.0006 | 0.0150 |
| 7UGA | A | 43 | 200 | 7.7713 | 7.7721 | -0.0008 | 0.1800 |
| 8CQ1 | A | 44 | 200 | 2.8448 | 2.8457 | -0.0008 | 0.0100 |
| 8HD6 | N | 419 | 200 | 13.1115 | 13.1126 | -0.0011 | 0.0000 |
| 7EFH | A | 19 | 200 | 10.0486 | 10.0497 | -0.0011 | 0.0000 |
| 8SP9 | C | 153 | 200 | 8.7100 | 8.7114 | -0.0013 | 0.0000 |
| 7FJ0 | A | 20 | 200 | 5.0450 | 5.0465 | -0.0015 | 0.0000 |
| 7XD7 | N | 405 | 200 | 7.5857 | 7.5872 | -0.0015 | 0.0000 |
| 8BWT | A | 26 | 200 | 2.7464 | 2.7480 | -0.0016 | 0.0000 |
| 7QP2 | A | 27 | 200 | 0.7905 | 0.7920 | -0.0016 | 0.0100 |
| 8FW4 | Z | 210 | 200 | 3.1609 | 3.1626 | -0.0017 | 0.0000 |
| 7XD3 | N | 426 | 200 | 13.6225 | 13.6243 | -0.0018 | 0.0000 |
| 8I7N | N | 418 | 200 | 4.1237 | 4.1255 | -0.0018 | 0.0000 |
| 6XKO | A | 101 | 200 | 8.6722 | 8.6739 | -0.0018 | 0.0100 |
| 8BTZ | A | 238 | 200 | 18.2948 | 18.2966 | -0.0018 | 0.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8TNS | A | 24 | 200 | 11.2120 | 11.2277 | -0.0157 | 0.0000 |
| 7QA2 | A | 22 | 200 | 11.1835 | 11.1984 | -0.0149 | 0.0000 |
| 7Q48 | A | 22 | 200 | 11.3933 | 11.4067 | -0.0134 | 0.0000 |
| 7Q6L | A | 22 | 200 | 11.8988 | 11.9110 | -0.0122 | 0.0000 |
| 7VFT | A | 23 | 200 | 21.3706 | 21.3810 | -0.0104 | 0.0000 |
| 7EDN | A | 23 | 200 | 19.2631 | 19.2720 | -0.0089 | 0.0000 |
| 7Y2B | S | 13 | 200 | 10.6834 | 10.6922 | -0.0088 | 0.0050 |
| 7EXY | A | 16 | 200 | 5.4099 | 5.4183 | -0.0084 | 0.0000 |
| 7UMD | A | 40 | 200 | 5.2166 | 5.2247 | -0.0081 | 0.0000 |
| 7RWR | A | 38 | 200 | 6.7615 | 6.7691 | -0.0076 | 0.0000 |
| 7MKT | A | 23 | 200 | 8.6643 | 8.6715 | -0.0071 | 0.0000 |
| 7SHX | A | 94 | 200 | 19.9318 | 19.9388 | -0.0070 | 0.0000 |
| 8THV | A | 29 | 200 | 5.0821 | 5.0890 | -0.0069 | 0.0000 |
| 8SCF | A | 30 | 200 | 3.2868 | 3.2932 | -0.0064 | 0.0000 |
| 8I43 | A | 19 | 200 | 5.4728 | 5.4789 | -0.0061 | 0.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
