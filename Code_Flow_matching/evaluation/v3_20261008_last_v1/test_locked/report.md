# test refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/last-v1.ckpt`
- candidates: 12310
- PDB chains: 67
- candidate-level mean RMSD: 9.0532 → 8.9891 Å
- candidate win/worsen rate: 60.50% / 39.49%
- PDB-chain macro mean improvement: 0.0599 Å (95% bootstrap CI -0.0242, 0.1445)
- PDB-chain win/worsen rate: 65.67% / 34.33%
- Spearman(length, improvement): -0.2256
- Spearman(input RMSD, improvement): 0.2554
- Spearman(mean pLDDT, improvement): -0.0367

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 23 | 4600 | 5.0887 | 4.9460 | 0.1427 | 0.6522 | 0.6348 |
| 51-100 | 18 | 3600 | 4.7009 | 4.7179 | -0.0170 | 0.5000 | 0.5197 |
| 101-200 | 7 | 1387 | 19.9082 | 19.7841 | 0.1240 | 1.0000 | 0.7808 |
| > 200 | 19 | 2723 | 19.0493 | 19.0406 | 0.0087 | 0.6842 | 0.6150 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 4 | 800 | 1.5878 | 1.9790 | -0.3912 | 0.0000 | 0.0100 |
| 2-5 | 25 | 5000 | 3.6883 | 3.6784 | 0.0100 | 0.5200 | 0.5294 |
| 5-10 | 19 | 3800 | 7.1138 | 6.8864 | 0.2274 | 0.7895 | 0.7395 |
| > 10 | 19 | 2710 | 24.6963 | 24.6434 | 0.0529 | 0.8421 | 0.7253 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8ZNQ | A | 30 | 200 | 4.2024 | 3.1266 | 1.0759 | 1.0000 |
| 9IOS | A | 19 | 200 | 5.5315 | 4.7280 | 0.8035 | 1.0000 |
| 9EOW | A | 29 | 200 | 4.7973 | 3.9941 | 0.8032 | 1.0000 |
| 9FO9 | A | 33 | 200 | 6.3014 | 5.5347 | 0.7667 | 1.0000 |
| 9FO8 | A | 29 | 200 | 5.7614 | 5.0536 | 0.7078 | 1.0000 |
| 9W1L | A | 26 | 200 | 7.0178 | 6.4390 | 0.5788 | 1.0000 |
| 7WII | V | 50 | 200 | 4.2626 | 3.7429 | 0.5197 | 0.9900 |
| 8VCI | A | 118 | 200 | 22.0501 | 21.5522 | 0.4978 | 0.9850 |
| 8Q4O | A | 23 | 200 | 9.8867 | 9.4288 | 0.4579 | 0.9800 |
| 9G4R | A | 47 | 200 | 7.7232 | 7.3609 | 0.3623 | 0.9950 |
| 9HRF | A | 70 | 200 | 7.9536 | 7.6415 | 0.3121 | 0.9950 |
| 9ECQ | A | 17 | 200 | 2.8844 | 2.6007 | 0.2837 | 1.0000 |
| 9J4O | A | 89 | 200 | 3.9191 | 3.6590 | 0.2601 | 0.9800 |
| 9FM4 | A | 25 | 200 | 4.2487 | 4.0000 | 0.2488 | 1.0000 |
| 8HB8 | A | 55 | 200 | 8.7618 | 8.5537 | 0.2081 | 0.9950 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8JHP | A | 27 | 200 | 3.9901 | 4.8629 | -0.8728 | 0.0050 |
| 9HRO | A | 35 | 200 | 4.5639 | 5.3557 | -0.7918 | 0.0000 |
| 9OD4 | A | 23 | 200 | 1.4889 | 2.2000 | -0.7111 | 0.0200 |
| 7SXP | A | 22 | 200 | 3.9829 | 4.6390 | -0.6561 | 0.0000 |
| 8VT5 | A | 60 | 200 | 1.8027 | 2.3491 | -0.5464 | 0.0000 |
| 8Q6N | A | 46 | 200 | 2.1247 | 2.5622 | -0.4374 | 0.0000 |
| 23JZ | A | 35 | 200 | 1.3054 | 1.6128 | -0.3074 | 0.0000 |
| 9BZC | A | 89 | 200 | 2.9866 | 3.2801 | -0.2935 | 0.0050 |
| 9BLM | A | 68 | 200 | 4.3655 | 4.5822 | -0.2168 | 0.0300 |
| 9OBM | A | 73 | 200 | 5.3086 | 5.5000 | -0.1914 | 0.0500 |
| 9UUG | A | 59 | 200 | 3.1983 | 3.3813 | -0.1830 | 0.0100 |
| 9DE7 | A | 58 | 200 | 3.5858 | 3.6786 | -0.0928 | 0.5850 |
| 8UPT | A | 71 | 200 | 4.0924 | 4.1800 | -0.0876 | 0.0650 |
| 8ITS | A | 46 | 200 | 13.0811 | 13.1019 | -0.0208 | 0.2650 |
| 8UPY | C | 72 | 200 | 4.0120 | 4.0305 | -0.0185 | 0.1800 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
