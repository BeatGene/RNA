# test refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`
- candidates: 12310
- PDB chains: 67
- candidate-level mean RMSD: 9.0532 → 8.9726 Å
- candidate win/worsen rate: 65.82% / 34.12%
- PDB-chain macro mean improvement: 0.0747 Å (95% bootstrap CI -0.0039, 0.1507)
- PDB-chain win/worsen rate: 68.66% / 31.34%
- Spearman(length, improvement): -0.3158
- Spearman(input RMSD, improvement): 0.2378
- Spearman(mean pLDDT, improvement): 0.0012

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 23 | 4600 | 5.0887 | 4.9219 | 0.1669 | 0.7391 | 0.6874 |
| 51-100 | 18 | 3600 | 4.7009 | 4.6770 | 0.0239 | 0.5000 | 0.5808 |
| 101-200 | 7 | 1387 | 19.9082 | 19.8240 | 0.0842 | 0.8571 | 0.7574 |
| > 200 | 19 | 2723 | 19.0493 | 19.0417 | 0.0076 | 0.7368 | 0.6104 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 4 | 800 | 1.5878 | 1.9250 | -0.3372 | 0.2500 | 0.2313 |
| 2-5 | 25 | 5000 | 3.6883 | 3.6614 | 0.0269 | 0.5200 | 0.5428 |
| 5-10 | 19 | 3800 | 7.1138 | 6.8566 | 0.2573 | 0.7895 | 0.8121 |
| > 10 | 19 | 2710 | 24.6963 | 24.6547 | 0.0416 | 0.8947 | 0.6968 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9FO8 | A | 29 | 200 | 5.7614 | 4.8539 | 0.9075 | 1.0000 |
| 8ZNQ | A | 30 | 200 | 4.2024 | 3.3165 | 0.8859 | 1.0000 |
| 9IOS | A | 19 | 200 | 5.5315 | 4.7649 | 0.7666 | 1.0000 |
| 9EOW | A | 29 | 200 | 4.7973 | 4.0566 | 0.7407 | 1.0000 |
| 9FO9 | A | 33 | 200 | 6.3014 | 5.5993 | 0.7022 | 1.0000 |
| 9W1L | A | 26 | 200 | 7.0178 | 6.4182 | 0.5997 | 1.0000 |
| 8Q4O | A | 23 | 200 | 9.8867 | 9.3655 | 0.5213 | 0.9850 |
| 8VCI | A | 118 | 200 | 22.0501 | 21.6116 | 0.4385 | 0.9850 |
| 9HRF | A | 70 | 200 | 7.9536 | 7.5407 | 0.4130 | 1.0000 |
| 9G4R | A | 47 | 200 | 7.7232 | 7.3383 | 0.3849 | 1.0000 |
| 7WII | V | 50 | 200 | 4.2626 | 3.8978 | 0.3648 | 0.9700 |
| 9ECQ | A | 17 | 200 | 2.8844 | 2.5964 | 0.2880 | 1.0000 |
| 8XZO | A | 53 | 200 | 5.7300 | 5.4709 | 0.2591 | 1.0000 |
| 9J4O | A | 89 | 200 | 3.9191 | 3.6642 | 0.2548 | 0.9850 |
| 9SY8 | A | 14 | 200 | 3.3952 | 3.1438 | 0.2514 | 0.5500 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8JHP | A | 27 | 200 | 3.9901 | 4.9040 | -0.9139 | 0.0000 |
| 9OD4 | A | 23 | 200 | 1.4889 | 2.3103 | -0.8214 | 0.0250 |
| 8Q6N | A | 46 | 200 | 2.1247 | 2.5257 | -0.4010 | 0.0000 |
| 7SXP | A | 22 | 200 | 3.9829 | 4.3647 | -0.3818 | 0.0150 |
| 9BZC | A | 89 | 200 | 2.9866 | 3.3387 | -0.3521 | 0.0000 |
| 23JZ | A | 35 | 200 | 1.3054 | 1.6478 | -0.3423 | 0.0000 |
| 9HRO | A | 35 | 200 | 4.5639 | 4.8857 | -0.3219 | 0.0000 |
| 9BLM | A | 68 | 200 | 4.3655 | 4.5729 | -0.2075 | 0.0150 |
| 8VT5 | A | 60 | 200 | 1.8027 | 1.9878 | -0.1850 | 0.0000 |
| 9UUG | A | 59 | 200 | 3.1983 | 3.3354 | -0.1372 | 0.0500 |
| 9DE7 | A | 58 | 200 | 3.5858 | 3.6216 | -0.0358 | 0.6500 |
| 8T5O | A | 124 | 200 | 6.5849 | 6.6005 | -0.0156 | 0.4000 |
| 8UPY | C | 72 | 200 | 4.0120 | 4.0248 | -0.0128 | 0.1800 |
| 8V1H | A | 75 | 200 | 2.7302 | 2.7356 | -0.0054 | 0.0450 |
| 8UPT | A | 71 | 200 | 4.0924 | 4.0967 | -0.0043 | 0.3950 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
