# Data_PT_V2 分布分析与验证集处理

此工具对应当前 `build_refinement_pt.py` 的 schema-v2。按照根目录 `AGENT.md`：本地准备代码和运行包，由用户上传、在服务器执行，再下载结果；工具不连接服务器、不修改任何 PT、划分或模型。

## 已核实的项目背景

- `pos_pred` 是 Protenix 输入，`pos` 是对齐到输入坐标系的 native；只在 `target_mask` 为真处计算监督指标。缺失 native 原子的占位坐标不能计入 RMSD 或二面角误差。
- `atom_plddt` 存为 **0–1**；显示值 0.8 相当于 80。pLDDT 是预测置信度，与使用 native 计算的 lDDT 不同。本工具读取 pLDDT。
- `ranking_score` 通常**没有写入 PT**，而在 Protenix summary-confidence JSON、PT manifest 和 rank-1 清单中。工具优先读取显式提供的分数表，再读 PT 可选字段/原始 JSON，缺失时记录警告，绝不用 pLDDT/RMSD 替代排名分数。
- `SERVER.md` 与 `RANK1_PROMOTION_20261013.md` 记录：`~/Data_PT_V2/train` 保留候选；val/test 每个 PDB 一个已选 PT。完整候选归档为 `~/Data_PT_V2_all_candidates_20261013/{val,test}`。旧报告的 103/67 个 PDB、19,722/12,310 个候选仅为历史记录，本次以实际读取结果为准。
- 旧报告中部分 PDB 少于 50×4 个 PT。实际选择是“**现存可用 PT 中最高 ranking_score**”；完整 200 个预测、成功生成的 PT、经过 RMSD 筛选的 PT 不一定是同一个集合。
- V3 的采样器按 `train_length_table` 中 `RNA_LENGTH >50` 做 2 倍 PT 采样。提供同一张表后，工具按表重建曝光权重；PT 中的长度仍用于结构分析。
- 项目已有实际复现的 CUDA BF16 图均值错误，见 `FP32_GRAPH_MEAN_FIX.md`。分析分布与修复/复评这个问题可以分别进行；不能凭分布图认定模型差的主因。

## 分析设计

| 比较口径 | 含义 |
|---|---|
| `train_candidates` | 每个训练 PT 等权 |
| `train_effective` | 按真实训练长度采样器加权；默认长链 2 倍 |
| `train_pdb_equal` | 先给每个 PDB 相同总权重，再平均它的所有候选 |
| `train_rank1` | 训练端也取最高 ranking_score 候选；只诊断选择效应，不替换训练集 |
| `val_rank1` / `test_rank1` | 已冻结的每 PDB 一个候选，保持不变 |
| `val_all_pdb_equal` / `test_all_pdb_equal` | 可选归档全候选，每个 PDB 等权 |

主要看 `train_effective → test_rank1`，再与 `train_rank1 → test_rank1` 对照。如果选择规则一致后差异大幅缩小，存在候选选择效应；如果长度/序列/输入几何差异仍大，值得检查训练覆盖。两种机制可以同时存在。

抽取指标包括：

- 原子掩码下重新作 proper Kabsch 的 RMSD，保存 RMSD 与重算值的差异；逐残基全局拟合误差、高质量残基比例。
- 长度、原子数、可观测原子比例、GC/核苷酸组成、修饰比例、回转半径。
- 原子/残基均值 pLDDT、残基低分位和低置信度比例；PAE/PDE/contact 矩阵均值。
- 重原子碰撞数/1000 原子、受影响原子比例、平方重叠均值；键长 RMSE、绝对偏差 >0.2 Å 的键比例。
- RNA α、β、γ、δ、ε、ζ、χ 及伪二面角 η、θ：输入/native 的 12 箱角分布与成对圆周误差；跨缺失原子、序列断点、退化角的测量不计为 0。χ 使用嘌呤 O4′–C1′–N9–C4、嘧啶 O4′–C1′–N1–C2。
- RNA-FM 残基向量的均值，每个 PDB 一个向量。

碰撞定义使用项目相同的 0.8×范德华半径和距离阈值，排除 PT 的 1–2/1–3 对，但统计全部 4.5 Å 邻居而不截断到 64。因此此值可以比较数据集几何，**不应当作训练代码的 clash loss 或 MolProbity clashscore**。

标量比较输出加权 ECDF、KS 差异幅度、Wasserstein 距离、均值差和 PDB bootstrap CI，不将多个候选当独立样本报告显著性。归档全候选与该 PDB 的 rank-1 比较使用配对 PDB bootstrap。二面角额外输出圆周 Wasserstein 距离。

降维分为三种空间：

1. **input**：不依赖 native 的输入结构/置信度摘要，不含 RMSD/native 误差，也不含模型没有直接接收的 `ranking_score`。二面角使用 sin/cos 的一、二阶矩，跨 ±180° 连续。
2. **oracle**：再加入 RMSD/native 二面角误差等，诊断监督难度。仅用于分析，不能用于部署时识别输入难度。
3. **RNA-FM**：每 PDB 一个均值表征，帮助观察序列/表征差异；二维聚类不等同于真实同源族。

缺失值填补、标准化、PCA 和可选 UMAP 都**只拟合训练数据**。默认每训练 PDB 最多随机取 8 个候选用于降维/最近邻，避免同一 RNA 的 200 个近似重复点占满图；完整分布统计仍用所有 PT。这个参考集不严格对应有效训练采样概率，因此 PCA 是描述性视图，采样曝光差异看 `train_effective` 的统计表。

`input_coverage.tsv` 在完整标准化输入摘要空间计算到训练参考的最近距离，以训练候选到“其他 PDB”最近距离的第 95 百分位作参考阈值。`rnafm_coverage.tsv` 另在完整均值 RNA-FM 空间计算距离。距离大提示需要逐目标检查，但不是“模型不可能学会”的判定；它依赖特征和参考采样数，也可能受同源 PDB 影响。改变 `--embedding-per-pdb` 后需要重新计算阈值。不要从二维图距离判断覆盖。标量摘要、二面角矩和均值 embedding 均有信息损失，不是模型的完整输入表征。

## 验证集建议

**保留原 PDB 划分，val rank-1 用于选模，val 全候选用于诊断稳定性。**不把同一 PDB 的 200 个候选拆到训练/验证，不把 `holdout_foldbench_val_homology` 移回训练或验证，不根据测试图重划分成更容易的测试集。

- 主验证：每 PDB 在原始可用候选中冻结的最高 ranking_score，推理规则与测试一致。未来从头训练时，用此验证集选择 checkpoint/早停；旧 epoch49 曾按全候选 val 选出的历史结果不等同于已经完成 rank-1 选模。
- 诊断验证：同一批 val PDB 的归档全部候选，按 PDB 宏平均；报告排名分位、RMSD/置信度分层中的稳定性。若它参与调参，就明确登记为验证的一部分。
- 同时看平均/中位 RMSD 改善、改善 ≥0.1 Å 比例、输入 <2 Å 回退率，以及碰撞/坏键/二面角误差恶化。预先固定选模规则，例如先在 val 限制几何回退和高质量输入回退，再比较宏平均 RMSD；具体阈值由 val 确定。
- 固定长度 ≤50、51–100、101–200、>200 nt 的分层，报告每层 PDB 数。RMSD 和置信度分层是分析视角，不能让测试集决定采样权重和部署门槛。小样本分层不单独选 checkpoint。
- 数据量允许时，在**训练 PDB 中按同源簇**抽取额外内部验证集，或做按簇的交叉验证，比较“训练分布内”与当前时间/家族泛化验证。一个簇整体划分；簇内多个 PDB 也不是完全独立样本。不在本工具中自动改划分。
- 本工具只检测完全相同 PDB/序列跨集合，不能代替完整序列/结构同源审计。若用于正式不确定性估计，已有同源簇信息时应再作簇级 bootstrap。
- 测试集本次可以作已声明的事后失败诊断。后续据此迭代后的成绩应说明这段开发历史；要得到新的独立泛化估计，留出新的冻结终测集。

是否要改变训练候选采样，等结果回来再决定。可考虑的受控实验是原采样 vs 混合“每 PDB rank-1 + 随机/难候选”，其余划分/代码/训练预算相同；不能现在仅凭推测删掉低分候选。

## 上传与运行

上传 `rna_dataset_shift_20261014.tar.gz` 到服务器 `$HOME/`。压缩包包含五个新增脚本/说明文件和已有 `etflow/data/constants.py` 的副本，不包括模型、权重或数据。它不覆盖训练、建 PT 或 rank-1 选择脚本。

在现有训练/PyTorch conda 环境中执行：

```bash
cd "$HOME/Code_Flow_matching"
tar -xzf "$HOME/rna_dataset_shift_20261014.tar.gz" -C "$HOME/Code_Flow_matching"
python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib; print('OK')"
python -m unittest discover -s scripts -p 'test_analyze_dataset_shift.py' -v
bash scripts/run_dataset_shift_analysis.sh
```

若缺分析依赖，使用现有环境安装缺少的包，避免替换服务器 CUDA Torch：

```bash
python -m pip install -r scripts/requirements_dataset_shift.txt
```

不需要 GPU。默认 4 个 CPU worker，主要成本是逐个读取 PT，候选多时耗时/IO不可忽视。内存紧张可设 `WORKERS=1`；降维参考候选数设 `--embedding-per-pdb 4`。默认报告会读入完整 TSV 并建立多个分布视图，建议在内存充足的节点执行。

前台命令输出 `OUTPUT ...` 和 `DONE ...`。默认同时分析归档候选；归档目录不存在时只分析主数据。完全不分析归档：

```bash
INCLUDE_ARCHIVE=0 bash scripts/run_dataset_shift_analysis.sh
```

需要后台运行时自行执行（输出写入指定新目录，PID 可记录）：

```bash
OUT="$HOME/Code_Flow_matching/evaluation/dataset_shift_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
nohup bash scripts/run_dataset_shift_analysis.sh "$OUT" > "$OUT/launcher.log" 2>&1 &
echo "$!" > "$OUT/analysis.pid"
echo "$OUT"
```

中断后对**同一输出目录、相同抽取参数**恢复：

```bash
RESUME=1 bash scripts/run_dataset_shift_analysis.sh /完整路径/原结果目录
```

缓存按 PT 大小/mtime 和抽取配置检查；失败或缺 ranking_score 的行重试。新增、变更、删除的样本会同步到表。不要在分析运行时生成/移动 PT；若原始 JSON 改变分数且已有分数缓存，应使用新输出目录重新抽取。候选结果每 100 条落盘，突然退出可能需要重做最后一批。

训练分数 JSON 不存在时，提供对应建集 `manifest.tsv`（以 split/PDB/seed/sample 身份匹配，绝不按新路径猜测）：

```bash
TRAIN_SCORE_TABLE=/完整路径/manifest.tsv bash scripts/run_dataset_shift_analysis.sh
```

有多份 manifest 时直接用 Python 命令重复 `--score-table`。冲突分数会报错。val/test 清单默认读取归档的 `rank1_selection.tsv`。`LENGTH_TABLE=/完整路径/selection_audit.tsv` 可指定训练实际使用的长度表；`LONG_FACTOR`、`LONG_THRESHOLD` 应与被分析的训练配置一致。

可选 UMAP：

```bash
python -m pip install umap-learn
UMAP=1 bash scripts/run_dataset_shift_analysis.sh
```

本地只有 PCA 已测试；可选 UMAP 缺依赖时保留 PCA 并发出警告。UMAP 图适合探索，不能单独用于归因。

## 下载与继续分析

下载整个输出目录最方便。首先提供以下结果供继续判断：

- `summary.json`、`report.md`：规模、数据异常和较大分布差异。
- `distribution_summary.tsv`、`shift_metrics.tsv`、`strata.tsv`、`torsion_shift.tsv`：数值证据。
- `pdb_audit.tsv`、`selection_audit.tsv`、`extraction_errors.tsv`、`exact_split_overlaps.tsv`：候选覆盖、排名和划分检查。
- `distributions.png`、`torsions.png`、`pca_input.png`、`pca_oracle.png`、`pca_rnafm.png`、`input_coverage.tsv`。

`features.tsv`、`rnafm.tsv` 也下载后，本地可以完全脱离 PT 重新生成报告：

```bash
python scripts/analyze_dataset_shift.py --report-only --output-dir /完整路径/下载结果目录
```

重做报告时仍须提供原来的 `--long-factor`、`--long-threshold`；若原来提供了训练长度表，也应下载并传入 `--train-length-table`，保持采样口径相同。

若要检验训练覆盖距离是否与**模型实际改善量**相关，使用同一 checkpoint/代码版本产生的 rank-1 `samples.tsv`，额外执行：

```bash
python scripts/analyze_dataset_shift.py --report-only --output-dir /完整路径/结果目录 \
  --performance-table /完整路径/val_final/samples.tsv \
  --performance-table /完整路径/test_locked/samples.tsv
```

此关联按 split/PDB/seed/sample 严格匹配，还核对输入 RMSD，输出 `coverage_performance.tsv` 和分组摘要。不同 checkpoint 的结果在不同报告目录比较，不混成一份。该关联仍可能受长度、输入质量与旧 BF16 错误混杂，不能单独证明覆盖不足导致失败。优先使用修复后的统一复评结果。

发现 PT 抽取错误或已选候选与归档最高分不符时，工具仍写诊断报告，最后退出码为 **2**；先处理表中异常再做正式结论。缺失分数会在报告中警告并排除该训练 PDB 的 rank-1 比较，不代造排名。

## 文件与删除规则（AGENT.md）

| 新增本地文件 | 用途 | 完成后可删除吗 |
|---|---|---|
| `scripts/analyze_dataset_shift.py` | PT 特征抽取、统计、降维与诊断报告 | 不影响训练；保留便于复现，确认不再重跑后可删 |
| `scripts/test_analyze_dataset_shift.py` | 掩码/几何/选择/统计与端到端回归检查 | 不影响训练；可删，建议随分析代码保留 |
| `scripts/run_dataset_shift_analysis.sh` | 服务器 CPU 一键运行入口 | 可删；Python 脚本仍可直接运行 |
| `scripts/requirements_dataset_shift.txt` | 分析依赖清单 | 可删；已安装依赖不会因此卸载 |
| `scripts/DATASET_SHIFT_ANALYSIS.md` | 方法、验证集策略、命令与结果说明 | 可删；建议保留用于复现 |
| `rna_dataset_shift_20261014.tar.gz` | 上传包 | 解压核对后可直接删除 |

自动产生的 `__pycache__` 可删除。合成测试使用临时目录，退出自动清理。服务器新增 `evaluation/dataset_shift_<时间>/` 是分析输出，归档下载后可删；不影响 PT 或训练。但删除其中的 `features.tsv`/`rnafm.tsv` 后，恢复/离线重画需要重新读取 PT。不要为清理本工具而删除 `~/Data_PT_V2`、归档候选或高 RMSD 保存目录。
