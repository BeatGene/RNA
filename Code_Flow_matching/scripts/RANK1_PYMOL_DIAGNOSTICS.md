# Rank-1 RNA：native、精修前、精修后 PyMOL 诊断

覆盖验证集 103 个 PDB 和测试集 67 个 PDB。每个 PDB 只使用已经选好的一个 PT；默认权重为 `epoch=49-step=157550.ckpt`。不重新选候选或权重。灰色=native，蓝色=精修前，橙色=精修后。

## 服务器运行

把交付的 `rank1_pymol_diagnostics_20261013.tar.gz` 上传至服务器 `~/`，然后执行：

```bash
tar -xzf ~/rank1_pymol_diagnostics_20261013.tar.gz -C ~/Code_Flow_matching
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
bash -n scripts/run_rank1_pymol_diagnostics.sh
```

压缩包包含新版脚本、现有评估与三结构导出脚本、模型源码和 V3 配置；解压会更新同名代码文件。运行输出使用新的时间戳目录。脚本只读取数据和权重，生成评估与可视化文件。

先核对 170 个 PT 是否与归档选择清单中的 PDB/seed/sample 完全一致、每个 PDB 只有一个 PT，以及原始 CIF 是否存在。这步不启动 GPU 推理：

```bash
CHECK_ONLY=1 bash scripts/run_rank1_pymol_diagnostics.sh
```

预期看到 `RANK1_PREFLIGHT_OK split=val pdbs=103` 和 `split=test pdbs=67`。确认 GPU 0 可用后，在同一终端执行：

```bash
mkdir -p evaluation
run_stamp=$(date +%Y%m%d_%H%M%S)
export PYMOL_DIAG_ROOT="$HOME/Code_Flow_matching/evaluation/rank1_pymol_diagnostics_$run_stamp"
nohup env CUDA_VISIBLE_DEVICES=0 EVAL_GPUS=1 \
  bash scripts/run_rank1_pymol_diagnostics.sh \
  > "evaluation/rank1_pymol_diagnostics_${run_stamp}.log" 2>&1 &
echo "PID=$!"
echo "LOG=$HOME/Code_Flow_matching/evaluation/rank1_pymol_diagnostics_${run_stamp}.log"
echo "OUTPUT=$PYMOL_DIAG_ROOT"
```

查看进度：

```bash
tail -f "evaluation/rank1_pymol_diagnostics_${run_stamp}.log"
```

结束后日志最后会输出 `DOWNLOAD_THIS_ZIP .../pymol_bundle.zip`。把这个 ZIP 下载到本地，完整解压。服务器无需安装 PyMOL。默认使用一块 GPU、batch size 1、BF16；如需使用两块空闲 GPU，可把启动命令改为 `CUDA_VISIBLE_DEVICES=0,1 EVAL_GPUS=2`。

如果旧版在最后 ZIP 打包时出现 `ZIP does not support timestamps before 1980`，上传修复包后仅执行：

```bash
python scripts/repack_rank1_pymol_bundle.py
```

这会自动定位最新 `evaluation/rank1_pymol_diagnostics_*/pymol_bundle`，核对所有目标和文件，然后生成新的 `pymol_bundle_repacked.zip`。也可通过 `--bundle-dir /实际路径/pymol_bundle` 指定某次输出。无需重跑推理，已有结构和旧 ZIP 均保留。修复版会把超出 ZIP 年份范围的时间戳限制到格式允许的范围，不修改源文件时间戳。

默认输入路径：

```text
~/Data_PT_V2/{val,test}
~/Data_PT_V2_all_candidates_20261013/rank1_selection.tsv
~/Data_V2
~/pdb_data
~/Code_Flow_matching/config/RNA_train_residual_v3.yaml
~/Code_Flow_matching/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt
```

如路径有变化，可设置 `DATA_ROOT`、`SELECTION_FILE`、`RANK1_ARCHIVE_ROOT`、`PREDICTION_ROOT`、`NATIVE_ROOT`、`CHECKPOINT`、`CONFIG`。中断后保留已有输出，使用新时间戳目录重新运行。

## 本地 PyMOL 打开

在 Anaconda Prompt 中激活已安装 PyMOL 的环境（本机为 `pymol-vis`），然后启动：

```text
conda activate pymol-vis
python -m pymol
```

在 **PyMOL 命令输入框**中执行，路径替换为实际解压路径，Windows 路径建议使用 `/`：

```text
cd "D:/RNA/pymol_bundle"
@view_test_9IOS.pml
```

默认同一窗口叠加三种状态。PyMOL 右侧的 `native`、`before`、`after` 对象可以分别开关，例如只看 native+before 或 native+after。打开其他目标：

```text
@view_val_7POF.pml
@view_test_8JHP.pml
```

`manifest.tsv` 列出了全部 170 个目标及脚本名称。`inspection_order_val.tsv`、`inspection_order_test.tsv` 按平均改善量从小到大排序；最先是变差最明显的目标。

## 诊断命令

以下命令均在 PyMOL 中执行：

| 命令 | 作用 |
| --- | --- |
| `rna_fit backbone` | 切换到独立骨架 Kabsch 对齐，保留相机视角 |
| `rna_fit all_atom` | 切回独立全重原子 Kabsch 对齐 |
| `rna_overview` | 灰/蓝/橙骨架线条和磷原子，适合看整体折叠 |
| `rna_sticks` | 显示完整 RNA sticks，适合看碱基和糖环变形 |
| `rna_observed 1` | 只显示有 native 坐标对应的原子 |
| `rna_observed 0` | 显示全部模型原子，native 仍只含实验对应原子 |
| `rna_residue 12` | 放大第 12 个残基，并打印前后误差 |
| `rna_residue 10, 15` | 放大连续残基区间 |
| `rna_worst` | 打印最差的 10 个残基，并放大最差者 |
| `rna_color_change` | 精修后对象按误差变化上色：红=变差，绿=改善，白=接近不变，灰=缺实验标签 |
| `rna_color_change 1.0` | 颜色在 ±1 Å 改善量处饱和；默认 ±0.5 Å |
| `rna_next` / `rna_prev` | 按当前 split 的检查顺序浏览其他 PDB |
| `rna_save` | 保存当前可编辑 `.pse` 会话至 `sessions/<split>/` |

局部视图中的残基编号是 `.pt` 的连续完整序列位置 `1..R`，三对象统一使用链 A；原实验链名、auth_seq_id、插入码保存在 `atom_metrics.tsv`。颜色对应的逐残基误差始终使用**全原子整体拟合后的坐标**，即使当前窗口切换到了骨架拟合。这不是逐残基单独拟合。切换拟合方式或打开下一目标会恢复完整显示。

想看原始移动幅度时，应读 `raw_movement_a`，它使用精修前后同一原始输入坐标系；重新拟合后的前后坐标差不能直接当作模型移动量。

批量生成两套会话和概览 PNG（总共 340 组，可能耗时）：

```text
@render_all.pml
```

重新打开保存的 `.pse` 时可以直接查看三对象。如需继续使用诊断命令，先在 bundle 目录运行 `run rna_pymol_viewer.py, module`，再用 `rna_open test, 9IOS` 重新加载目标。

## 对齐、原子与键的约定

- 全原子视图：before、after 各自在 native 可观测对应重原子上做一次 proper-rotation Kabsch，native 固定。
- 骨架视图：before、after 各自在有 native 坐标的 P/OP1/OP2/OP3/O5′/C5′/C4′/O4′/C3′/O3′/C2′/O2′/C1′ 上拟合，再把变换用于完整结构。
- native 只包含实验结构中能可靠映射到模型的原子。实验缺原子/缺残基保持缺失，不能用 PT 中的填充值当作实验结构。
- native 本身已由建集时保存的刚体变换放到输入坐标系；该操作只改变姿态。`source_frame/` 保留原始三结构导出及原链/残基标签。
- 三对象的共价连接来自 `.pt/geometry_bond_index`，覆盖 PyMOL 按距离猜出的连接。原子移远后仍显示模板中的连接，原子靠近后不会自动多画键。
- 键在显示中统一为单键，且关闭 valence 显示；用于观察连接和变形，不表达完整芳香性/键级。native 不会跨缺失原子补连接。
- 可视化 CIF 源自评估器导出，其坐标已舍入到 0.0001 Å；打包时核对拟合 RMSD 与原评估数值，容许误差 0.001 Å。最终 CIF 保留五位小数。

## 输出文件与建议检查顺序

```text
pymol_bundle/
  manifest.tsv                       全部目标、候选身份、RMSD、物理损失
  inspection_order_val.tsv           验证集：最差到最好
  inspection_order_test.tsv          测试集：最差到最好
  rank1_selection.tsv                原始选择记录
  bundle_summary.json                数量、权重路径、选择记录哈希与显示约定
  checkpoint.sha256                  本次服务器权重哈希
  config_snapshot.yaml               本次配置快照
  rna_pymol_viewer.py
  view_val_<PDB>.pml
  view_test_<PDB>.pml
  view_test_<PDB>_backbone.pml
  val/<PDB>/                         test 中同样组织
    all_atom/{native,before,after}.cif
    backbone/{native,before,after}.cif
    source_frame/{native,before,after}.cif
    atom_metrics.tsv                 原子误差、移动幅度、pLDDT 与原编号
    residue_metrics.tsv              残基误差、键长误差、碱基平面厚度
    bond_metrics.tsv                 所有模板键前后长度与理想长度
    display.json                     拟合变换、观察原子和模板连接
```

建议先看测试集 `8JHP`（明显恶化）、`9IOS`（明显改善），以及验证集 `7POF`（较好的输入被拉差）；然后查看长链 `7ZJ4`。先比较整体，再用 `rna_worst` 定位残基，查看 sticks 和对应键长表。

`manifest.tsv` 中的三项物理损失来自原评估器，沿用训练定义；逐残基碱基平面指标为最佳拟合平面的 RMS 厚度，单位 Å，和整体平面 loss（Å²）不同。逐残基键长误差会把跨残基键计入其两端残基。几何指标不是独立化学验证分数。画面可以帮助定位问题，原因仍需结合训练与模型机制分析。
