# test refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`
- candidates: 67
- PDB chains: 67
- candidate-level mean RMSD: 10.4270 → 10.3517 Å
- candidate win/worsen rate: 61.19% / 38.81%
- PDB-chain macro mean improvement: 0.0752 Å (95% bootstrap CI -0.0113, 0.1622)
- PDB-chain win/worsen rate: 61.19% / 38.81%
- Spearman(length, improvement): -0.2647
- Spearman(input RMSD, improvement): 0.2811
- Spearman(mean pLDDT, improvement): -0.0347

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 23 | 23 | 5.1532 | 4.9622 | 0.1910 | 0.7391 | 0.7391 |
| 51-100 | 18 | 18 | 4.3058 | 4.3048 | 0.0010 | 0.4444 | 0.4444 |
| 101-200 | 7 | 7 | 19.5980 | 19.5405 | 0.0575 | 0.5714 | 0.5714 |
| > 200 | 19 | 19 | 19.2313 | 19.2193 | 0.0120 | 0.6316 | 0.6316 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 3 | 3 | 1.6455 | 1.7791 | -0.1336 | 0.3333 | 0.3333 |
| 2-5 | 29 | 29 | 3.6221 | 3.6188 | 0.0033 | 0.4828 | 0.4828 |
| 5-10 | 15 | 15 | 6.9395 | 6.6307 | 0.3088 | 0.8000 | 0.8000 |
| > 10 | 20 | 20 | 24.2269 | 24.1912 | 0.0357 | 0.7000 | 0.7000 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9IOS | A | 19 | 1 | 6.4937 | 4.9955 | 1.4982 | 1.0000 |
| 8ZNQ | A | 30 | 1 | 4.3486 | 3.4052 | 0.9434 | 1.0000 |
| 9EOW | A | 29 | 1 | 4.8317 | 3.9536 | 0.8781 | 1.0000 |
| 9FO9 | A | 33 | 1 | 6.7472 | 5.8839 | 0.8632 | 1.0000 |
| 9FO8 | A | 29 | 1 | 5.7927 | 5.0054 | 0.7873 | 1.0000 |
| 9L8F | U | 57 | 1 | 7.1186 | 6.7174 | 0.4012 | 1.0000 |
| 9HRF | A | 70 | 1 | 7.6485 | 7.2524 | 0.3961 | 1.0000 |
| 9SY8 | A | 14 | 1 | 3.3229 | 2.9723 | 0.3506 | 1.0000 |
| 8Q4O | A | 23 | 1 | 9.5235 | 9.1811 | 0.3425 | 1.0000 |
| 9ECQ | A | 17 | 1 | 2.8755 | 2.5410 | 0.3346 | 1.0000 |
| 8XZO | A | 53 | 1 | 4.7618 | 4.4461 | 0.3157 | 1.0000 |
| 8VCI | A | 118 | 1 | 23.2520 | 22.9472 | 0.3048 | 1.0000 |
| 9FM4 | A | 25 | 1 | 4.4865 | 4.2149 | 0.2715 | 1.0000 |
| 9G4R | A | 47 | 1 | 7.8032 | 7.5519 | 0.2513 | 1.0000 |
| 7WII | V | 50 | 1 | 4.4622 | 4.2125 | 0.2497 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8JHP | A | 27 | 1 | 3.2507 | 4.6393 | -1.3886 | 0.0000 |
| 8Q6N | A | 46 | 1 | 2.2058 | 2.5905 | -0.3847 | 0.0000 |
| 9HRO | A | 35 | 1 | 4.4905 | 4.8193 | -0.3288 | 0.0000 |
| 7SXP | A | 22 | 1 | 4.1838 | 4.5101 | -0.3264 | 0.0000 |
| 9BZC | A | 89 | 1 | 2.6282 | 2.9351 | -0.3069 | 0.0000 |
| 9DE7 | A | 58 | 1 | 2.7272 | 3.0286 | -0.3014 | 0.0000 |
| 9UUG | A | 59 | 1 | 2.8063 | 3.0771 | -0.2708 | 0.0000 |
| 23JZ | A | 35 | 1 | 1.3739 | 1.5880 | -0.2141 | 0.0000 |
| 8VT5 | A | 60 | 1 | 1.7905 | 1.9773 | -0.1868 | 0.0000 |
| 9OD4 | A | 23 | 1 | 2.3233 | 2.4786 | -0.1553 | 0.0000 |
| 9BLM | A | 68 | 1 | 4.4859 | 4.6232 | -0.1374 | 0.0000 |
| 9OBM | A | 73 | 1 | 5.1327 | 5.2662 | -0.1335 | 0.0000 |
| 8T5O | A | 124 | 1 | 4.2125 | 4.2792 | -0.0667 | 0.0000 |
| 8QO3 | A | 118 | 1 | 16.2645 | 16.3000 | -0.0355 | 0.0000 |
| 8UPT | A | 71 | 1 | 4.0872 | 4.1142 | -0.0270 | 0.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
