# epoch 49：FoldBench RNA 单体精修前后 lDDT

运行入口：`scripts/compare_foldbench_refinement_v3.py`。它只使用
`epoch=49-step=157550.ckpt`，只评估测试集与 FoldBench RNA 单体目标表交集中的
9 个 PDB：7SXP、7WIA、7WII、7ZJ4、8HB8、8ITS、8UPT、8V1H、9G7C。

## 上传文件

将以下两个本地文件同步到服务器的同名位置：

- `scripts/compare_foldbench_refinement_v3.py`
- `scripts/export_foldbench_v3.py`（已修复原始链编号 V/E 与预测链编号 A 不同造成的漏选）

服务器还需保留之前的 `scripts/summarize_foldbench_v3.py`、配置文件、epoch 49
checkpoint、`evaluation/v3_20261008/test_locked` 及 `~/Data_PT_V2`。

## 运行前检查

需要 FoldBench 官方仓库及其 `foldbench` Conda 环境；该环境内的 `python` 可运行
`evaluate.py`，且 `ost` 在 PATH 中。`$HOME/FoldBench_ground_truths` 只是示例路径，
不会由脚本自动创建。先从 FoldBench README 链接的
[官方 referenced CIF 包](https://drive.google.com/file/d/17KdWDXKATaeHF6inPxhPHIRuIzeqiJxS/view?usp=sharing)
取得参考结构，再将实际目录传给 `--ground-truth-dir`。

在服务器逐项检查：

```bash
FB_REPO="$HOME/Code/FoldBench"          # 改成实际仓库目录
GT_DIR="$FB_REPO/examples/ground_truths" # 改成官方 CIF 解压后的实际目录
test -f "$FB_REPO/evaluate.py" && test -f "$FB_REPO/task_score_summary.py" && echo "FoldBench 仓库 OK"
conda env list
conda run -n foldbench python -c 'import pandas, tqdm; print("FoldBench Python OK")'
conda run -n foldbench ost compare-structures --help >/dev/null && echo "OST OK"
for id in 7sxp 7wia 7wii 7zj4 8hb8 8its 8upt 8v1h 9g7c; do
  f="$GT_DIR/${id}-assembly1.cif"
  if [[ -s "$f" ]]; then echo "OK $f"; else echo "MISSING $f"; fi
done
```

9 行都显示 `OK` 才能开始评分。文件名与非空检查不能证明文件来源；应保留官方
下载包，并从该包解压。评分脚本还会在 GPU 导出前检查目录和 Conda 环境。

在服务器激活 `protenix-1.0.5` 后执行，按实际位置修改最后两个参数：

```bash
cd ~/Code_Flow_matching
python -u scripts/compare_foldbench_refinement_v3.py \
  --foldbench-repo "$FB_REPO" \
  --ground-truth-dir "$GT_DIR"
```

若 FoldBench 环境名称不同，增加 `--foldbench-conda-env 环境名`。若只想先导出 9 个
目标，增加 `--stage export`；之后加 `--stage score` 完成评分。

脚本写入新目录 `evaluation/v3_20261008/foldbench_9targets`，不会覆盖以前的 6 目标
导出。这个目录里有 FoldBench 官方 `raw/monomer_rna_ost.csv`、`summary_table.csv`，
以及逐目标的 `paired_lddt.tsv`、`paired_lddt.md`。另有汇报文件：

```text
evaluation/foldbench_epoch49_9targets_comparison/paired_lddt_all.md
evaluation/foldbench_epoch49_9targets_comparison/paired_lddt_all.tsv
```

每个目标按照原始 Protenix `ranking_score` 选择一个候选，精修前后使用同一候选。
报告中的平均 lDDT 是 9 个目标的子集平均，不能直接与 FoldBench 官方 15 个 RNA
单体目标的榜单均值比较。若任何目标的导出或 lDDT 缺失，脚本报错，不生成完整报告。
