# 三个 PT 数据池的只读审计

新版默认采用完整重建比较。上传 `pt_audit_rebuild_bundle.tar.gz` 到 `/storage9920/home/tinghao.xia/`，解压后在 `pt_audit_rebuild/` 目录运行。
包内包含审计脚本、本地当前 `build_refinement_pt.py` 与 `constants.py` 的参考快照；不覆盖服务器现有代码。
不要只上传新版审计脚本而漏掉参考构建器。脚本仍可用 `--builder` 显式指定另一套构建器，报告会记录哈希及版本。
需要 Python >=3.9、PyTorch（支持 `weights_only=True`）、NumPy 和 Gemmi；不使用 GPU，不需要 Lightning/PyG。
请使用原来构建 PT 的环境。脚本不会自动安装软件、修改 PT、移动数据或重做划分。

## 默认输入

- 当前集：`/storage9920/home/tinghao.xia/Data_PT_V2`
- 全候选归档：`/storage9920/home/tinghao.xia/Data_PT_V2_all_candidates_20261013`
- 高 RMSD 隔离集：`/storage9920/home/tinghao.xia/Data_PT_V2_RMSD_GT30`
- 原始预测：`/storage9920/home/tinghao.xia/Data_V2`
- native：`/storage9920/home/tinghao.xia/pdb_data`
- RNA-FM：`/storage9920/home/tinghao.xia/Data_FM/RNA_FM_embeddings`

当前集和高 RMSD 集识别 train、val、test、holdout_foldbench_val_homology 四个 split；归档只识别 val/test。
只要求当前集 val/test 每个 PDB 一个候选，不把此规则错误应用于 train 或 holdout。
所有路径可用 `--*-root` 参数覆盖。报告输出目录必须是新目录，并在输入目录之外。

## 运行

先激活能读取原 PT 的环境，然后检查依赖：

```bash
cd /storage9920/home/tinghao.xia
tar -xzf pt_audit_rebuild_bundle.tar.gz
cd pt_audit_rebuild
python -c 'import torch, gemmi, numpy; torch.empty(0).numpy(); print("torch", torch.__version__, "gemmi", gemmi.__version__)'
```

先做小规模前台检查（每个池最多 3 个 PT，不进行跨池/排名审计）：

```bash
python -u audit_refinement_pt.py \
  --output "$PWD/pt_audit_smoke_$(date +%Y%m%d_%H%M%S)" \
  --limit 3 --sources rebuild
```

小检查的 `complete` 固定为 false，不能把它当作全量通过。exit=1 表示已经发现数据问题，可以按需要继续全量收集；exit=2 表示任务未正常完成，应先检查运行问题。

正式全量后台运行（不要带 `--limit`）：

```bash
cd /storage9920/home/tinghao.xia/pt_audit_rebuild
audit_out="/storage9920/home/tinghao.xia/pt_audit_$(date +%Y%m%d_%H%M%S)"
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u audit_refinement_pt.py \
  --output "$audit_out" --sources rebuild --threads 1 \
  > "${audit_out}.launcher.log" 2>&1 < /dev/null &
audit_pid=$!
echo "PID=$audit_pid"
echo "OUTPUT=$audit_out"
echo "LOG=${audit_out}.launcher.log"
```

用输出的真实路径查看，不要在新终端假设 `audit_out` 变量仍存在：

```bash
tail -f /storage9920/home/tinghao.xia/pt_audit_实际时间.launcher.log
```

默认 `--sources rebuild` 从原始 CIF、RNA-FM 和置信度 JSON **在内存中完整重新生成一个 PT 字典**，逐字段比较旧 PT，不保存新的大批 PT，也不覆盖旧 PT。
链选择、序列匹配、原子筛选/顺序、原子行号、监督 mask、Kabsch 对齐、图和几何模板、置信度聚合全部重新执行，不借用旧 PT 的映射或原文件路径。仅使用文件相对路径的 split/PDB/seed/sample 确定待重建候选。
这只是重新生成 PT 内容，不重新运行 Protenix 或 RNA-FM 模型；它们已有的预测与表征作为源输入。
可能运行较久，主要消耗共享存储 I/O；没有 GPU 计算。
旧模式 `--sources full` 是部分独立源检查，并非完整重建；`coordinates` 和 `none` 的检查范围更小。最终排名审计仍读取原始 ranking JSON。
不同模式的通过结论覆盖范围不同，记录在 summary.json 中。

## 输出

| 文件 | 用途 |
|---|---|
| run.log | 进度、首批异常、结束标志 |
| files.tsv | 每个实际扫描 PT 的状态、N、L、RMSD、耗时 |
| issues.tsv | 全部 ERROR/WARNING；日志只显示每类前 5 条，完整记录在这里 |
| rebuild_samples.tsv | 每个 PT 的重建状态 MATCH/METADATA_ONLY/DIFFERENT/FAILED、字段数、差异数及重建耗时；仅路径/版本等差异会单独标记 METADATA_ONLY |
| rebuild_differences.tsv | 重建与旧 PT 的字段名、差异种类和最大数值差等详情；只有差异才写入 |
| pdb_counts.tsv | 各池/split/PDB 中成功解析并通过结构合同检查的候选数 |
| rank1_audit.tsv | 当前 val/test 选择与可用归档最高分、原始预测最高分的比较 |
| sequence_overlaps.tsv | 当前四个 split 间完全相同序列的 PDB 对，不是同源簇审计 |
| schema.json | 所有可加载字典的字段、类型、出现次数、缺失次数、实际尺寸范围和版本分布 |
| schema.md | 可读的 PT schema |
| summary.json | 完成状态、扫描数、错误/警告计数、脚本哈希、配置、版本及耗时 |
| audit.sqlite | 磁盘索引，支持跨池比较，避免把全部 PT 留在内存 |

检查 `summary.json` 中的 `complete`、`exit_code` 和 `findings`，并查看 run.log 的 AUDIT_FINISHED：
0=执行成功且无 ERROR（仍可能有警告），1=执行完成并有 ERROR，2=致命异常/未完成。
schema 的字典计数包含可加载但校验失败的字典，不能据 schema 的存在声称数据全部正确。
运行中断时可保留已有 TSV 作为部分结果；重新全量运行须选择新输出目录，本版不支持断点续跑。

## 审计规则和边界

- 检查张量形状、dtype、有限值、范围、残基/原子编号、原子身份唯一性、图索引、双向边、化学键与 1-2/1-3 排除关系。
- 逐 PT 重算 proper-rotation Kabsch RMSD，检查已配对的坐标框架、mask、占位坐标和 native-to-decoy 旋转/平移。
- 源检查逐个核对 CIF 原始行号、原子名称、元素、链和残基；核对 native 声明的单 RNA 链范围。
- native 的 canonical 序列能直接从 CIF 还原时独立比较。修饰残基无法从 CIF 直接还原 canonical 序列时，标记 source_sequence_unverified 警告；本工具不加载整份外部 CCD，不假装已经完成该独立验证。
- full 模式另外比较原始 RNA-FM、pLDDT、原子/token 映射以及残基聚合后的 PAE/PDE/contact 三矩阵。
- rebuild 模式完整执行参考构建器的 build_sample，独立核对所有生成字段。坐标、刚体变换和 RMSD 默认允许 0.002 的绝对差，其余数值张量逐值精确比较，dtype 不同也报告。构建器版本、过滤设置及 source_* 路径变化记 WARNING，模型数据或映射差异记 ERROR。缺失字段也会列出。
- 新构建器无法处理某个旧候选时记 rebuild_failed，不将其当成通过。参考源码、constants 和解析后的 CCD 路径记入 summary.json；更换构建器版本造成的差异不自动等同于旧 PT 损坏，需要看具体字段和历史策略。
- 比较当前 val/test 与归档的候选身份及完整内容摘要；归档不是 200 个时发出警告，不能把实际缺候选误记成完成 200 个。
- rank-1 按 score 降序、seed 升序、sample 升序；分别记录可用归档最优和原始预测最优。如果排名文件缺失，报告无法核验，不静默假定选择正确。
- 检查同 PDB 跨 split、完全相同序列跨 split、高 RMSD 隔离集与另两池的同候选重叠。
- 主数据中 RMSD 超过 metadata 阈值记警告，以便追溯历史 override；不会删除历史困难目标。高 RMSD 池 <=30 Å 则记 ERROR。阈值可用 --rmsd-cutoff 修改。
- 不对未知化学有效性作保证：不会替代独立几何验证、完整同源过滤或数据来源审计；不会要求当前数据覆盖全部 2241 个 native（当前只建模单 RNA 链子集）。
- 重建能验证现存 PT 是否符合参考构建器及输入，不能证明构建器本身没有系统性错误；原有数值/索引检查仍保留。扫描范围是现存 PT，不宣称独立补齐未生成或历史被过滤的全部候选。

## 坐标系与 etflow

当前构建器保留每个 decoy 的原始坐标 Q，将 native N 做 Kabsch 变换为 N@R+t，存成 pos。
所以每个 PT 内的 pos_pred 与 pos 在同一个配对坐标系；同 PDB 的不同 decoy 不保证都在同一个 native 坐标系。

如果要转到 native 框架，必须对整对数据应用同一个逆变换：

```text
pos_pred_new = (pos_pred - t) @ R.T
pos_new      = (pos      - t) @ R.T
```

这也会正确变换未观测原子的占位值。只变换 decoy、不变换原有 target 会破坏监督。
转换后还必须更新坐标来源/变换 metadata 和结构导出代码；当前审计预期原构建器的坐标约定，不直接接受未经说明的另一套约定。

etflow 在模型入口分别按图去均值，SO(3) 等变网络由相对几何预测向量；共同的 proper rigid transform 不改变理想坐标损失及几何损失。
因此改对齐方向通常不带来本质训练收益，也不修复 BF16 归约、困难样本稀缺或局部几何退化。
native 框架可方便离线叠图/分析，但真实推理没有 native，仍应支持输入 decoy 自身的坐标框架。
现阶段先审计已有配对关系，无需为改对齐方向而重建 PT。

## 核心 schema 参考（实际版本以生成报告为准）

N=预测保留的重原子，L=canonical 残基，E=有向静态边，B=化学键，C=碰撞排除对。

| 字段 | 形状 | 当前构建器保存类型 |
|---|---|---|
| pos / pos_pred | N×3 | float32 |
| target_mask | N | bool |
| atomic_numbers / atom_name_id / residue_index / atom_to_token_idx | N | int64 |
| residue_type_id | L | int64 |
| rnafm_embedding | L×640 | float16 |
| edge_index | 2×E | int64 |
| edge_attr | E×7 | float32 |
| geometry_bond_index | 2×B | int64 |
| ideal_bond_length | B | float32 |
| clash_exclusion_index | 2×C | int64 |
| atom_plddt | N | float32 |
| token_pair_pae / token_pair_pde / contact_probs | L×L | float16 |
| native_to_prediction_rotation / translation | 3×3 / 3 | float32 |
| predicted_atom_site_row / native_atom_site_row | N | int64（native 未观测为 -1） |

PT 还保存序列、PDB/链/seed/sample、原文件路径、完整 residue/token 映射、observed/modified 残基掩码和版本等。
node_attr=674、置信度边额外 4 维、num_tokens 和 mobility summaries 是 dataset.py 加载时构造，不是必须存在的序列化字段。
