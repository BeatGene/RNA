# 验证集、测试集：每个 PDB 仅保留一个 Protenix rank-1 `.pt`

`select_rank1_pt.py` 对 `~/Data_PT_V2/val` 和 `~/Data_PT_V2/test` **分别**按 PDB 分组，读取每个现存 `.pt` 对应的原始 Protenix `summary_confidence_sample_*.json` 中的 `ranking_score`，选数值最高者。平分时取 seed 较小、再取 sample 编号较小者。不使用 native RMSD、lDDT 或精修结果选择候选。

原始 Protenix 文件名 `sample_0` 只代表该 seed 内四个样本的第一名；此脚本重新比较所有 seed，得到该 PDB 在**现存 `.pt` 候选中**的全局第一名。若某 PDB 少于 50 seed × 4 sample，仍按现存候选选最高分，并在清单中记录 `candidate_count`、`seed_count` 和 `complete_50x4`。它不会把缺失候选当作已评估。

脚本创建 `~/Data_PT_V2_rank1/val` 和 `test`，每个 PDB 下仅有一个 `.pt` 符号链接，原始数据不动。还写入 `selection.tsv`、`coverage.tsv`、`summary.json`，方便核对。若原始 `.pt` 将来要删除或移动，运行时可加 `--materialize copy` 创建独立副本。

## 服务器上操作

把上传包解压到 `~/Code_Flow_matching` 后，在 `protenix-1.0.5` 环境执行：

```bash
cd ~/Code_Flow_matching
python scripts/select_rank1_pt.py --dry-run
python scripts/select_rank1_pt.py
cat ~/Data_PT_V2_rank1/summary.json
```

`--dry-run` 会检查全部候选分数与输出数量，不创建目录。若它报告某些 JSON 缺失，先解决路径或提供建集时的 `manifest.tsv`（`--manifest /绝对路径/manifest.tsv`）；脚本不会默默跳过缺分候选。若 `~/Data_V2` 不是真实预测数据根目录，可加 `--prediction-root /实际目录`。

脚本默认要求输出目录尚不存在，防止混入旧候选。成功后在新的 rank-1 数据目录上**重新**比较 checkpoint，再评估验证集和测试集：

```bash
cd ~/Code_Flow_matching
DATA_ROOT="$HOME/Data_PT_V2_rank1" \
EVAL_ROOT="$HOME/Code_Flow_matching/evaluation/v3_rank1_20261008" \
EVAL_GPUS=4 bash scripts/run_v3_evaluation.sh screen

cat evaluation/v3_rank1_20261008/checkpoint_comparison.tsv

DATA_ROOT="$HOME/Data_PT_V2_rank1" \
EVAL_ROOT="$HOME/Code_Flow_matching/evaluation/v3_rank1_20261008" \
EVAL_GPUS=4 FOLDBENCH_EXPORT=0 bash scripts/run_v3_evaluation.sh finalize
```

`screen` 只在 rank-1 验证集比较 checkpoint；`finalize` 用该验证集选出的 checkpoint 评估 rank-1 验证集和测试集。两阶段必须传相同的 `DATA_ROOT` 与 `EVAL_ROOT`。原先 `evaluation/v3_20261008` 的全候选评估结果仍保留，不能与新目录混用。
