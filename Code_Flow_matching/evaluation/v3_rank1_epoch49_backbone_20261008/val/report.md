# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`
- candidates: 103
- PDB chains: 103
- candidate-level mean RMSD: 9.3504 → 8.9215 Å
- candidate win/worsen rate: 66.99% / 33.01%
- PDB-chain macro mean improvement: 0.4288 Å (95% bootstrap CI 0.1951, 0.7341)
- PDB-chain win/worsen rate: 66.99% / 33.01%
- Spearman(length, improvement): -0.2814
- Spearman(input RMSD, improvement): 0.1141
- Spearman(mean pLDDT, improvement): 0.1530

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 47 | 6.9535 | 6.1591 | 0.7944 | 0.7234 | 0.7234 |
| 51-100 | 20 | 20 | 8.7798 | 8.5699 | 0.2099 | 0.7000 | 0.7000 |
| 101-200 | 14 | 14 | 16.0364 | 15.8488 | 0.1876 | 0.7143 | 0.7143 |
| > 200 | 22 | 22 | 10.7349 | 10.7344 | 0.0005 | 0.5000 | 0.5000 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 12 | 12 | 1.1996 | 0.9637 | 0.2359 | 0.7500 | 0.7500 |
| 2-5 | 24 | 24 | 3.5420 | 3.5041 | 0.0380 | 0.4583 | 0.4583 |
| 5-10 | 30 | 30 | 7.3055 | 6.9560 | 0.3495 | 0.7333 | 0.7333 |
| > 10 | 37 | 37 | 17.4194 | 16.6101 | 0.8093 | 0.7297 | 0.7297 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 1 | 11.4458 | 0.5725 | 10.8733 | 1.0000 |
| 7EFH | A | 19 | 1 | 10.0586 | 3.0030 | 7.0556 | 1.0000 |
| 7EDN | A | 23 | 1 | 19.4022 | 12.9331 | 6.4691 | 1.0000 |
| 7PS8 | A | 23 | 1 | 7.8941 | 6.1741 | 1.7200 | 1.0000 |
| 7Y2B | S | 13 | 1 | 11.7362 | 10.3640 | 1.3721 | 1.0000 |
| 6XKN | A | 101 | 1 | 9.9057 | 8.7333 | 1.1725 | 1.0000 |
| 8I43 | A | 19 | 1 | 5.3735 | 4.2953 | 1.0782 | 1.0000 |
| 8I45 | A | 19 | 1 | 3.9059 | 2.8525 | 1.0533 | 1.0000 |
| 6XKO | A | 101 | 1 | 10.0033 | 8.9635 | 1.0398 | 1.0000 |
| 8I44 | A | 19 | 1 | 4.7645 | 3.8475 | 0.9170 | 1.0000 |
| 7TDA | A | 83 | 1 | 1.7531 | 0.9203 | 0.8328 | 1.0000 |
| 7TDC | A | 83 | 1 | 1.5703 | 0.8104 | 0.7599 | 1.0000 |
| 7TDB | A | 83 | 1 | 1.6518 | 0.9154 | 0.7363 | 1.0000 |
| 7EOP | A | 48 | 1 | 8.8425 | 8.1254 | 0.7172 | 1.0000 |
| 7QA2 | A | 22 | 1 | 11.1490 | 10.4834 | 0.6656 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 1 | 3.3077 | 4.4018 | -1.0942 | 0.0000 |
| 8BWT | A | 26 | 1 | 2.6621 | 3.3603 | -0.6982 | 0.0000 |
| 7POF | A | 14 | 1 | 1.7288 | 2.4031 | -0.6743 | 0.0000 |
| 7FHI | A | 20 | 1 | 3.9023 | 4.1957 | -0.2935 | 0.0000 |
| 7KUD | A | 13 | 1 | 2.7434 | 3.0175 | -0.2742 | 0.0000 |
| 7EEM | A | 27 | 1 | 0.7722 | 1.0039 | -0.2317 | 0.0000 |
| 7UZ0 | A | 87 | 1 | 2.2897 | 2.4143 | -0.1247 | 0.0000 |
| 8SCF | A | 30 | 1 | 3.4043 | 3.4920 | -0.0877 | 0.0000 |
| 8CQ1 | A | 44 | 1 | 3.3984 | 3.4834 | -0.0850 | 0.0000 |
| 8SCH | A | 68 | 1 | 20.5364 | 20.6000 | -0.0636 | 0.0000 |
| 8TNS | A | 24 | 1 | 11.2078 | 11.2652 | -0.0574 | 0.0000 |
| 7RWR | A | 38 | 1 | 6.7697 | 6.8245 | -0.0548 | 0.0000 |
| 7MLW | F | 128 | 1 | 2.1677 | 2.2175 | -0.0498 | 0.0000 |
| 7UME | A | 28 | 1 | 3.3581 | 3.3945 | -0.0364 | 0.0000 |
| 7V9E | A | 68 | 1 | 12.7000 | 12.7335 | -0.0336 | 0.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
