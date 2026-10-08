# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt`
- candidates: 103
- PDB chains: 103
- candidate-level mean RMSD: 9.3504 → 8.9207 Å
- candidate win/worsen rate: 66.99% / 33.01%
- PDB-chain macro mean improvement: 0.4297 Å (95% bootstrap CI 0.1959, 0.7347)
- PDB-chain win/worsen rate: 66.99% / 33.01%
- Spearman(length, improvement): -0.2745
- Spearman(input RMSD, improvement): 0.1182
- Spearman(mean pLDDT, improvement): 0.1479

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 47 | 47 | 6.9535 | 6.1582 | 0.7953 | 0.7234 | 0.7234 |
| 51-100 | 20 | 20 | 8.7798 | 8.5700 | 0.2098 | 0.7000 | 0.7000 |
| 101-200 | 14 | 14 | 16.0364 | 15.8457 | 0.1907 | 0.7143 | 0.7143 |
| > 200 | 22 | 22 | 10.7349 | 10.7345 | 0.0004 | 0.5000 | 0.5000 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 12 | 12 | 1.1996 | 0.9632 | 0.2364 | 0.7500 | 0.7500 |
| 2-5 | 24 | 24 | 3.5420 | 3.5036 | 0.0384 | 0.4583 | 0.4583 |
| 5-10 | 30 | 30 | 7.3055 | 6.9535 | 0.3519 | 0.7333 | 0.7333 |
| > 10 | 37 | 37 | 17.4194 | 16.6103 | 0.8091 | 0.7297 | 0.7297 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7EFG | A | 19 | 1 | 11.4458 | 0.5729 | 10.8729 | 1.0000 |
| 7EFH | A | 19 | 1 | 10.0586 | 3.0067 | 7.0519 | 1.0000 |
| 7EDN | A | 23 | 1 | 19.4022 | 12.9426 | 6.4596 | 1.0000 |
| 7PS8 | A | 23 | 1 | 7.8941 | 6.1736 | 1.7205 | 1.0000 |
| 7Y2B | S | 13 | 1 | 11.7362 | 10.3625 | 1.3736 | 1.0000 |
| 6XKN | A | 101 | 1 | 9.9057 | 8.7125 | 1.1933 | 1.0000 |
| 8I43 | A | 19 | 1 | 5.3735 | 4.2960 | 1.0774 | 1.0000 |
| 8I45 | A | 19 | 1 | 3.9059 | 2.8516 | 1.0543 | 1.0000 |
| 6XKO | A | 101 | 1 | 10.0033 | 8.9494 | 1.0539 | 1.0000 |
| 8I44 | A | 19 | 1 | 4.7645 | 3.8478 | 0.9167 | 1.0000 |
| 7TDA | A | 83 | 1 | 1.7531 | 0.9153 | 0.8379 | 1.0000 |
| 7TDC | A | 83 | 1 | 1.5703 | 0.8111 | 0.7592 | 1.0000 |
| 7TDB | A | 83 | 1 | 1.6518 | 0.9174 | 0.7344 | 1.0000 |
| 7EOP | A | 48 | 1 | 8.8425 | 8.1220 | 0.7206 | 1.0000 |
| 7QA2 | A | 22 | 1 | 11.1490 | 10.4833 | 0.6657 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8FCS | A | 71 | 1 | 3.3077 | 4.3989 | -1.0913 | 0.0000 |
| 8BWT | A | 26 | 1 | 2.6621 | 3.3584 | -0.6963 | 0.0000 |
| 7POF | A | 14 | 1 | 1.7288 | 2.3995 | -0.6707 | 0.0000 |
| 7FHI | A | 20 | 1 | 3.9023 | 4.1942 | -0.2919 | 0.0000 |
| 7KUD | A | 13 | 1 | 2.7434 | 3.0178 | -0.2744 | 0.0000 |
| 7EEM | A | 27 | 1 | 0.7722 | 1.0041 | -0.2319 | 0.0000 |
| 7UZ0 | A | 87 | 1 | 2.2897 | 2.4126 | -0.1229 | 0.0000 |
| 8SCF | A | 30 | 1 | 3.4043 | 3.4953 | -0.0910 | 0.0000 |
| 8CQ1 | A | 44 | 1 | 3.3984 | 3.4781 | -0.0797 | 0.0000 |
| 8SCH | A | 68 | 1 | 20.5364 | 20.5954 | -0.0591 | 0.0000 |
| 8TNS | A | 24 | 1 | 11.2078 | 11.2649 | -0.0571 | 0.0000 |
| 7MLW | F | 128 | 1 | 2.1677 | 2.2180 | -0.0503 | 0.0000 |
| 7RWR | A | 38 | 1 | 6.7697 | 6.8124 | -0.0426 | 0.0000 |
| 7UME | A | 28 | 1 | 3.3581 | 3.3955 | -0.0374 | 0.0000 |
| 7V9E | A | 68 | 1 | 12.7000 | 12.7336 | -0.0336 | 0.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
