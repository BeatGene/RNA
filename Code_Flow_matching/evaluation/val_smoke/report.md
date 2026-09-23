# val refinement evaluation

- checkpoint: `/storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v2_residual_mobility/epoch=12-step=39442.ckpt`
- candidates: 12
- PDB chains: 1
- candidate-level mean RMSD: 8.4641 → 7.6876 Å
- candidate win/worsen rate: 100.00% / 0.00%
- PDB-chain macro mean improvement: 0.7765 Å (95% bootstrap CI 0.7765, 0.7765)
- PDB-chain win/worsen rate: 100.00% / 0.00%
- Spearman(length, improvement): nan
- Spearman(input RMSD, improvement): nan
- Spearman(mean pLDDT, improvement): nan

正的 improvement 表示 RMSD 下降；负值表示 refinement 后反而变差。
所有主表使用 observed atom 上重新 Kabsch 对齐的 RMSD。

## 按长度分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 50 | 0 | 0 | nan | nan | nan | nan | nan |
| 51-100 | 0 | 0 | nan | nan | nan | nan | nan |
| 101-200 | 1 | 12 | 8.4641 | 7.6876 | 0.7765 | 1.0000 | 1.0000 |
| > 200 | 0 | 0 | nan | nan | nan | nan | nan |

## 按输入 RMSD 分层（PDB-chain macro）

| stratum | pdb_chain_count | candidate_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | pdb_chain_win_rate | candidate_win_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <= 2 | 0 | 0 | nan | nan | nan | nan | nan |
| 2-5 | 0 | 0 | nan | nan | nan | nan | nan |
| 5-10 | 1 | 12 | 8.4641 | 7.6876 | 0.7765 | 1.0000 | 1.0000 |
| > 10 | 0 | 0 | nan | nan | nan | nan | nan |

## 改善最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6XKN | A | 101 | 12 | 8.4641 | 7.6876 | 0.7765 | 1.0000 |

## 变差最大的 PDB chains

| pdb_id | native_chain_id | length | sample_count | input_rmsd_mean | refined_rmsd_mean | improvement_mean | candidate_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6XKN | A | 101 | 12 | 8.4641 | 7.6876 | 0.7765 | 1.0000 |

## 解释限制

- 同一 PDB 的约200个 Protenix候选是相关重复，不是独立实验靶标。
- 长度结论应同时查看长度分层和连续 Spearman 相关，且注意输入 RMSD/长度混杂。
- test 只应用验证集预先选定的 checkpoint，不应再用 test 选择模型。
