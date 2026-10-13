# Epoch 49 rank-1：骨架 RMSD 对齐口径与三结构 PyMOL 导出

本次沿用 `Data_PT_V2_rank1` 中已由 **Protenix 原始 ranking_score** 选出的候选，以及固定权重 `epoch=49-step=157550.ckpt`。验证集和测试集分别评估，不重新选择候选或 checkpoint。

## 两种骨架 RMSD

- **先取骨架再对齐（原口径）**：在 native 有对应坐标的骨架原子上做 Kabsch，然后在同一组原子上计算 RMSD。
- **先对齐再取骨架（新增口径）**：在 native 有对应坐标的全部原子上做 Kabsch，再只在骨架原子上计算 RMSD。

两种口径都对精修前和精修后的结构分别拟合。骨架原子集合为 P、OP1、OP2、OP3、O5′、C5′、C4′、O4′、C3′、O3′、C2′、O2′、C1′。改善量均为精修前 RMSD 减精修后 RMSD。新增口径的绝对 RMSD 通常不低于原口径，因为原口径直接最小化骨架误差。

## 上传并运行

将以下四个文件上传至实验室服务器的  同名路径：

```text
evaluate_refinement_v3.py
export_rank1_pymol_v3.py
summarize_backbone_alignment_v3.py
run_rank1_epoch49_backbone_allfit_pymol.sh
```

在服务器执行：

```bash
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
bash -n scripts/run_rank1_epoch49_backbone_allfit_pymol.sh
set -o pipefail
bash scripts/run_rank1_epoch49_backbone_allfit_pymol.sh 2>&1 | tee evaluation/rank1_backbone_allfit_pymol_20261012.log
```

默认使用 4 块 GPU，并读取：

```text
~/Data_PT_V2_rank1/{val,test}
~/Data_V2
~/pdb_data
~/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt
```

若服务器上的数据位置不同，可在运行前设置 `DATA_ROOT`、`PREDICTION_ROOT`、`NATIVE_ROOT`、`PROJECT_ROOT` 或 `EVAL_GPUS`。输出根目录可用 `BACKBONE_PYMOL_ROOT` 指向一个**尚不存在**的新路径。脚本不覆盖已有结果；中断后应先检查部分产物，再选新目录重跑。

完成标志为 `BACKBONE_ALLFIT_PYMOL_COMPLETE`。主要输出：

```text
evaluation/v3_rank1_epoch49_backbone_allfit_pymol_20261012/
  val/samples.tsv
  test/samples.tsv
  backbone_comparison/report.md
  backbone_comparison/summary.json
  backbone_comparison/per_pdb_chain.tsv
  pymol_bundle.zip
```

`samples.tsv` 逐候选包含两种骨架 RMSD；汇总以每条 PDB 链为单位。脚本会核对两套计算使用相同的 rank-1 `.pt`，并确认每个 split 的样本数与选择清单一致。

## PyMOL 查看

下载并解压 `pymol_bundle.zip`，在 PyMOL 命令行先 `cd` 到解压出的 `pymol_bundle` 目录，再运行例如：

```text
@view_val_6xkn_seed_308_sample_0.pml
```

实际脚本名可在目录中查看；每个验证集或测试集候选各有一个 `view_*.pml`。`@render_all.pml` 会依次生成全部 PNG 与 `.pse` 会话。`metrics.tsv` 记录对应指标。灰色为 native、橙色为精修前、青色为精修后。

每组包含 `native.cif`、`input.cif`、`refined.cif`。`native.cif` 保留实验结构中与模型原子可靠配对的原子，并使用建集时保存的刚体变换置于 Protenix 输入坐标系；输入和精修 CIF 只包含模型实际处理的 RNA 原子。三者已处于同一坐标系，PyMOL 脚本不再重新对齐；图片显示不改变 RMSD 计算结果。

本地仓库没有完整的 rank-1 `.pt`、epoch 49 权重和全部原始 CIF，因此实际 103/67 条链的新增数值及全部 PyMOL 结构须在服务器上生成。
