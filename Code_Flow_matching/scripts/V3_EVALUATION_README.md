# V3 RNA refinement 评估：服务器操作说明

本地已准备这些新文件，需复制到实验室服务器 `~/Code_Flow_matching` 的相同相对路径：

可以只传 `v3_evaluation_bundle_20261008.zip`，在服务器的项目目录执行 `unzip -o v3_evaluation_bundle_20261008.zip`；压缩包仅包含下列新文件，不包含 checkpoint 或 `.pt` 数据。

- `scripts/evaluate_refinement_v3.py`
- `scripts/analyze_refinement_v3.py`
- `scripts/select_refinement_v3_checkpoint.py`
- `scripts/export_foldbench_v3.py`
- `scripts/summarize_foldbench_v3.py`
- `scripts/run_v3_evaluation.sh`
- `config/foldbench_monomer_rna_targets.csv`

服务器还必须有原来的 `scripts/evaluate_refinement.py`、V3 训练所用的 `config/RNA_train_residual_v3.yaml` 和对应版本的 `etflow/` 代码。评估脚本严格加载权重，若模型代码与 checkpoint 不符会立即报错。

## 第一阶段：仅在验证集比较 checkpoint

在服务器上运行：

```bash
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
EVAL_GPUS=4 bash scripts/run_v3_evaluation.sh screen
```

脚本先对验证集 8 个候选做含物理 loss 的预检，再比较列出的 5 个 `epoch=...ckpt`、`last.ckpt` 和 `last-v1.ckpt`，固定使用 `~/Data_PT_V2/val`，每个候选只评估一次；V3 residual 模式只需一次前向。选模规则事先固定为验证集 **PDB 链宏平均 refined RMSD 最低**，结果在：

```text
~/Code_Flow_matching/evaluation/v3_20261008/checkpoint_comparison.tsv
~/Code_Flow_matching/evaluation/v3_20261008/selected_checkpoint.txt
```

请先把 `checkpoint_comparison.tsv` 的内容发回，再决定是否使用自动选择的 checkpoint。`screen` 可在已完成的 checkpoint 上继续；若某个输出目录只有部分文件而没有 `summary.json`，脚本会停止，避免误用不完整结果。

如果机器可用 GPU 数不同，可修改 `EVAL_GPUS`。若显存允许，可把 `EVAL_BATCH_SIZE` 从默认 1 调高。评估目录可用 `EVAL_ROOT=/path/to/new/run` 指定；两阶段必须使用相同的 `EVAL_ROOT`。

## 第二阶段：选定后评估验证集与测试集

确认 checkpoint 后运行：

```bash
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
EVAL_GPUS=4 bash scripts/run_v3_evaluation.sh finalize
```

如需指定第一阶段比较结果中的另一个 checkpoint，可在命令前添加 `SELECTED_CHECKPOINT="$HOME/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=...ckpt"`。测试集只对该 checkpoint 评估一次。完整评估会计算与 V3 训练代码相同的未加权 bond、clash、plane loss（精修前和精修后）。

主要结果：

```text
evaluation/v3_20261008/analysis/analysis.md
evaluation/v3_20261008/analysis/rmsd_comparison.tsv
evaluation/v3_20261008/analysis/rmsd_histogram.tsv
evaluation/v3_20261008/analysis/rmsd_distribution.tsv
evaluation/v3_20261008/analysis/rmsd_transitions.tsv
evaluation/v3_20261008/analysis/physical_comparison.tsv
evaluation/v3_20261008/analysis/foldbench_overlap.tsv
evaluation/v3_20261008/val_final/samples.tsv
evaluation/v3_20261008/test_locked/samples.tsv
```

`analysis.md` 有验证/测试总体、长度和输入 RMSD 分层的前后数值，以及 `<2 Å` 子集占比和恶化情况。表中候选级与 PDB 链级分别给出；以链级统计作为主要结论。`rmsd_histogram.tsv` 是前后各分箱的样本数，`rmsd_transitions.tsv` 保留每个输入箱流向哪个精修后箱的计数。碱基平面 loss 使用与训练函数相同的协方差最小特征值公式，但对残基批量计算以节省时间。

物理 loss 的数值单位为 Å²，数值越小越好。clash loss 受邻域原子对组成影响，三项均不能单独证明完整化学几何正确性。

## FoldBench RNA 单体分数

V3 测试结果会与随附的 FoldBench `monomer_rna.csv`（15 个目标）按 **PDB ID + 预测链编号** 匹配；epoch 49 实际匹配 9 个。原始链编号有时是 V/E，而预测链编号是 A。`finalize` 默认会对每个匹配目标按原始 Protenix `ranking_score` 选择一个候选，输出成对的原始/精修 CIF 和 FoldBench `prediction_reference.csv`。选择时不使用 native RMSD 或 lDDT。若暂不导出，可设置 `FOLDBENCH_EXPORT=0`。现有 `foldbench/` 目录保留了修复前导出的 6 个目标；9 目标评分请使用 `scripts/FOLDBENCH_EPOCH49_SCORING.md` 中的新入口，写入 `foldbench_9targets/`。

FoldBench 官方 RNA 单体主分数是 **lDDT**；本项目的 Kabsch RMSD 不能当作 FoldBench 分数。需要 FoldBench 的 `ost` 环境和匹配目标的原始 `*-assembly1.cif` ground truth。官方仓库的 README 提供原始 CIF 下载地址。假设 FoldBench 在 `~/Code/FoldBench`，原始 CIF 在 `~/FoldBench_ground_truths`：

```bash
conda activate foldbench
cd ~/Code/FoldBench
EVAL_ROOT="$HOME/Code_Flow_matching/evaluation/v3_20261008"
GROUND_TRUTH_DIR="$HOME/FoldBench_ground_truths"

python evaluate.py \
  --targets_dir "$EVAL_ROOT/foldbench/targets" \
  --evaluation_dir "$EVAL_ROOT/foldbench/evaluation" \
  --algorithm_name ProtenixV3Input \
  --ground_truth_dir "$GROUND_TRUTH_DIR" \
  --targets monomer_rna

python evaluate.py \
  --targets_dir "$EVAL_ROOT/foldbench/targets" \
  --evaluation_dir "$EVAL_ROOT/foldbench/evaluation" \
  --algorithm_name ProtenixV3Refined \
  --ground_truth_dir "$GROUND_TRUTH_DIR" \
  --targets monomer_rna

python task_score_summary.py \
  --evaluation_dir "$EVAL_ROOT/foldbench/evaluation" \
  --target_dir "$EVAL_ROOT/foldbench/targets" \
  --output_path "$EVAL_ROOT/foldbench/summary_table.csv" \
  --algorithm_names ProtenixV3Input ProtenixV3Refined \
  --targets monomer_rna --metric_type rank

conda activate protenix-1.0.5
python "$HOME/Code_Flow_matching/scripts/summarize_foldbench_v3.py" \
  --foldbench-dir "$EVAL_ROOT/foldbench"
```

输出的 `summary_table.csv` 可比较同一批、同一排名候选的输入与精修 lDDT。`evaluation/*/raw/monomer_rna_ost.csv` 保存每个靶标的原始分数。最后一步要求所有目标均有有效 lDDT，并写出 `paired_lddt.tsv` 和 `paired_lddt.md`。因为这里只覆盖测试集与 15 个 FoldBench 目标的交集，候选还受 Data_PT_V2 的建集筛选影响，且 V3 训练集可能含 FoldBench 相近/相同序列，此结果是**交集上的后处理比较**，不等于官方完整且严格低同源条件下的排行榜分数。
