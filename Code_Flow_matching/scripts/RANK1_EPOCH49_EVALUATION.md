# Rank-1 `.pt`：固定 epoch 49 的验证集与测试集评估

此流程只加载 `epoch=49-step=157550.ckpt`，对 `~/Data_PT_V2_rank1/val`、`test` 中每个 PDB 选出的一个 `.pt` 重新推理。沿用 V3 评估器的单步 residual 推理、Kabsch RMSD、键长/碰撞/碱基平面物理 loss 与分析脚本。不会再次比较 checkpoint，也不会运行 FoldBench lDDT 评分。

服务器需已有 `~/Data_PT_V2_rank1/summary.json`、`selection.tsv` 和相应 `.pt` 符号链接；`protenix-1.0.5` 环境中有 4 块 GPU 可用。将上传包解压到 `~/Code_Flow_matching` 后：

```bash
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
nohup bash scripts/run_rank1_epoch49_evaluation.sh \
  > evaluation/rank1_epoch49.log 2>&1 < /dev/null &
echo "PID=$!"
```

运行状态与结束标志：

```bash
tail -n 40 evaluation/rank1_epoch49.log
pgrep -af 'run_rank1_epoch49_evaluation.sh|evaluate_refinement_v3.py|torchrun' || true
cat evaluation/v3_rank1_epoch49_20261008/val_final/summary.json
cat evaluation/v3_rank1_epoch49_20261008/test_locked/summary.json
```

最后一行应为 `RANK1_EPOCH49_COMPLETE`。结果位于：

```text
evaluation/v3_rank1_epoch49_20261008/val_final/
evaluation/v3_rank1_epoch49_20261008/test_locked/
evaluation/v3_rank1_epoch49_20261008/analysis/analysis.md
evaluation/v3_rank1_epoch49_20261008/analysis/rmsd_comparison.tsv
evaluation/v3_rank1_epoch49_20261008/analysis/physical_comparison.tsv
evaluation/v3_rank1_epoch49_20261008/analysis/rmsd_histogram.tsv
evaluation/v3_rank1_epoch49_20261008/analysis/rmsd_distribution.tsv
evaluation/v3_rank1_epoch49_20261008/analysis/rmsd_transitions.tsv
evaluation/v3_rank1_epoch49_20261008/analysis/low_input_rmsd_chains.tsv
```

原全候选目录 `evaluation/v3_20261008` 和 `evaluation/v3_20261008_last_v1` 不参与此次计算，也不会被覆盖。若 SSH 断开，后台任务可以继续；脚本可跳过已完成且 checkpoint、数据目录、样本数一致的 split。若 split 只有部分输出而无 `summary.json`，脚本会停下并提示检查该目录，避免混合残缺结果。
