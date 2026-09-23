# test refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v2_residual_mobility/epoch=12-step=39442.ckpt`
- candidates: 15310
- PDB chains: 82
- candidate-level mean RMSD: 9.2402 → 7.6228 Å
- candidate win/worsen rate: 80.95% / 19.03%
- PDB-chain macro mean improvement: 1.5105 Å (95% bootstrap CI 0.7757, 2.2993)
- PDB-chain win/worsen rate: 84.15% / 15.85%
- Spearman(length, improvement): -0.7433
- Spearman(input RMSD, improvement): 0.1847
- Spearman(mean pLDDT, improvement): 0.4671

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 40 | 8000 | 8.0416 | 4.9604 | 3.0812 | 0.9000 | 0.8543 |
| 51-100 | 17 | 3400 | 4.4621 | 4.4456 | 0.0164 | 0.7059 | 0.7121 |
| 101-200 | 7 | 1387 | 19.9082 | 19.8738 | 0.0344 | 1.0000 | 0.8203 |
| > 200 | 18 | 2523 | 17.5659 | 17.5607 | 0.0052 | 0.7778 | 0.7511 |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 4 | 800 | 1.5878 | 1.9016 | -0.3138 | 0.2500 | 0.2537 |
| 2-5 | 28 | 5600 | 3.6704 | 3.4925 | 0.1779 | 0.7857 | 0.7411 |
| 5-10 | 19 | 3800 | 6.8935 | 6.5746 | 0.3189 | 1.0000 | 0.9124 |
| > 10 | 31 | 5110 | 19.7731 | 16.0932 | 3.6799 | 0.8710 | 0.8528 |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9MDW | A | 16 | 200 | 14.5293 | 1.0078 | 13.5215 | 1.0000 |
| 9MDY | A | 16 | 200 | 14.3858 | 1.0271 | 13.3586 | 1.0000 |
| 9MDX | A | 16 | 200 | 14.5941 | 1.5895 | 13.0046 | 1.0000 |
| 9CSO | A | 16 | 200 | 14.4378 | 1.4436 | 12.9942 | 1.0000 |
| 36LL | A | 16 | 200 | 14.8695 | 1.8880 | 12.9815 | 1.0000 |
| 9CSP | A | 16 | 200 | 14.5159 | 2.0300 | 12.4859 | 1.0000 |
| 8TDZ | A | 16 | 200 | 14.5121 | 6.6773 | 7.8348 | 1.0000 |
| 8TDY | A | 16 | 200 | 14.5135 | 8.3461 | 6.1675 | 1.0000 |
| 8TE0 | A | 16 | 200 | 14.7935 | 9.0234 | 5.7701 | 1.0000 |
| 8TE2 | A | 16 | 200 | 14.5527 | 8.9970 | 5.5557 | 1.0000 |
| 8FEQ | AAA | 16 | 200 | 14.4736 | 10.8542 | 3.6193 | 1.0000 |
| 8FEO | AAA | 16 | 200 | 14.5456 | 11.0629 | 3.4827 | 1.0000 |
| 8FEP | AAA | 16 | 200 | 14.7627 | 11.9918 | 2.7710 | 0.9950 |
| 9IO1 | A | 19 | 200 | 4.3864 | 3.0800 | 1.3064 | 1.0000 |
| 9IOR | A | 19 | 200 | 5.3637 | 4.2195 | 1.1442 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9OD4 | A | 23 | 200 | 1.4889 | 2.3551 | -0.8662 | 0.0150 |
| 23JZ | A | 35 | 200 | 1.3054 | 1.5924 | -0.2870 | 0.0000 |
| 9ECQ | A | 17 | 200 | 2.8844 | 3.0660 | -0.1817 | 0.0000 |
| 9UUG | A | 59 | 200 | 3.1983 | 3.3553 | -0.1570 | 0.0000 |
| 8Q6N | A | 46 | 200 | 2.1247 | 2.2484 | -0.1237 | 0.0000 |
| 8VT5 | A | 60 | 200 | 1.8027 | 1.9047 | -0.1020 | 0.0000 |
| 8UPY | C | 72 | 200 | 4.0120 | 4.1072 | -0.0952 | 0.0050 |
| 9J4N | A | 85 | 200 | 3.4908 | 3.4950 | -0.0042 | 0.4450 |
| 9ISV | A | 580 | 20 | 28.4417 | 28.4438 | -0.0020 | 0.2000 |
| 8K15 | C | 621 | 178 | 21.7597 | 21.7604 | -0.0007 | 0.4101 |
| 9ELY | A | 205 | 164 | 25.8416 | 25.8422 | -0.0006 | 0.3293 |
| 9BZC | A | 89 | 200 | 2.9866 | 2.9871 | -0.0005 | 0.6400 |
| 9LEC | J | 378 | 87 | 27.0902 | 27.0903 | -0.0001 | 0.5517 |
| 9QTJ | 1 | 481 | 200 | 8.5065 | 8.5065 | 0.0000 | 0.5950 |
| 9C6J | A | 495 | 200 | 1.7542 | 1.7541 | 0.0001 | 1.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
