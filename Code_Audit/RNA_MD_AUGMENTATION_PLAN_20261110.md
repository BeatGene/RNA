**RNA refine 项目的 MD 调研与执行建议（2026-11-10）**

本报告基于本地代码与历史诊断的静态检查；本轮未连接服务器、运行 MD、训练模型或重新计算完整数据集指标。用户已确认：可用 4 张 H20，希望先用 1–2 天完成小规模试验；目前没有 BF16 修复后的新结果。H20 实际显存、软件环境和模拟吞吐尚待服务器实测。

结论：先完成数值检查和修复后基线复评，再复用现有困难 Protenix 候选，随后对“结构化扰动＋短程全原子 MD”和 BioEmu-2 做小规模比较。只有证明新增样本补充了实际需要的错误类型，才扩大生成规模并训练。DynaRNA 值得读，但其原始模型的表示和训练长度限制使它不宜直接承担本项目困难长链扩增。

**从项目中确认的事实**

| 事实 | 项目依据 | 对方案的影响 |
|---|---|---|
| 当前 V3 配置为 `training_objective: residual`，推理返回 source＋residual | `Code_Flow_matching/config/RNA_train_residual_v3.yaml`；`etflow/models/model.py` | 当前是直接坐标残差监督，不需要合成连续时间的真实 MD 标签；单个合理 decoy 就能作为输入候选 |
| 输入除原子坐标外，还包括 RNA-FM、pLDDT、PAE/PDE、contact_probs，以及它们的 mobility 汇总 | `etflow/data/dataset.py`；`BaseFlow.forward/apply_residue_mobility_gate` | 坐标扩增必须考虑置信度特征的语义，不能只替换 `pos_pred` 就认为接入完成 |
| 原始数据由 Protenix `protenix_base_default_v1.0.0` 生成，通常为 50 seeds×4 samples | Protenix Version_3 README；PT 构建脚本 | 已有较多候选，继续无差别增加 seeds 不应排在第一位 |
| 训练集历史统计为 758 个 PDB、151,220 个 PT；val/test rank-1 为 103/67 个 PDB | `evaluation/dataset_shift_20261014_084658/report.md` | 同一 PDB 的大量候选不能当作同等数量的独立 RNA |
| 历史训练实际曝光平均长度 74.2 nt、RMSD 3.645 Å；val rank-1 为 138.3 nt、9.350 Å | 同目录 `local_interpretation/分析解读.md` | 长度与监督难度同时偏移，应联合分层 |
| >200 nt 的训练候选为 4,650 个，RMSD≥10 Å 只有 65 个，来自 5 个 PDB | 同上 | 优先补长链大误差，避免只生成更多短链局部波动 |
| 输入二面角边际分布接近，但相对 native 的误差更大 | 同上 | 支持检查长程安排/螺旋相对位置等错误；尚不能断言全部是全局错误 |
| 当前 source augmentation 被关闭；已有函数仅生成 identity / near-native 输入 | V3 YAML；`augment_residual_source` | 直接打开旧开关不会解决困难 decoy 不足 |
| 历史 epoch49 结果受到已记录的 BF16 图均值错误影响 | `scripts/FP32_GRAPH_MEAN_FIX.md`；历史诊断 | 分布差异是真实诊断结果，但不能据旧模型表现证明它是性能不足的唯一原因 |

长链困难候选涉及 3DHS、4C4Q、5G2Y、6WLT、6WLU。其中 4C4Q/5G2Y 的现有候选只有 10/40 个。先追溯生成及过滤记录，区分预测失败、RMSD 过滤和转换失败，不直接假定需要重新采样。

**优先阅读哪些方法**

| 阅读顺序 | 方法/材料 | 重点看什么 | 本项目定位 |
|---|---|---|---|
| 1 | [BioEmu-2 模型卡](https://github.com/microsoft/bioemu2/blob/main/MODEL_CARD.md)、[RNA 系统定义](https://github.com/microsoft/bioemu2/blob/main/docs/system.md)、[采样/传播文档](https://github.com/microsoft/bioemu2/blob/main/docs/propagation.md) | RNA 输入、全原子输出、长链内存、物理性检查、RNA 验证证据 | 小规模候选生成对照；截至本次检索未找到对应公开论文，不能套用 BioEmu-1 的验证结论 |
| 2 | [DynaRNA 论文](https://doi.org/10.1038/s42003-025-08875-2)、[代码](https://github.com/lizxSJTU/DynaRNA) | partial noising 如何保留输入结构；长度与表示限制；全原子重建 | 借鉴扰动思路，必要时做中短链对照 |
| 3 | [OpenMM 模拟与力场文档](https://docs.openmm.org/latest/userguide/application/02_running_sims.html)、[OL3 力场作者说明](https://fch.upol.cz/ff_ol/refinements.php) | RNA 参数化、溶剂/离子、最小化/平衡、独立副本、轨迹导出 | 第一套要实际跑通的全原子 MD 基线 |
| 4 | [PLUMED OPES](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s.html)、[RNA eRMSD](https://www.plumed.org/doc-v2.10/user-doc/html/_e_r_m_s_d.html) | CV 选择、偏置、结构距离及去相关 | 普通 MD 不产生有用长程变化时，再进入的增强采样路线 |
| 5，可选 | [Rosetta FARFAR2](https://docs.rosettacommons.org/docs/latest/FARFAR2) | fragment assembly、局部重建、不同质量 decoy | 非 MD 的补充生成器；用于增加预测错误类型 |
| 6，后置 | [oxRNA](https://oxdna.org/) | 粗粒化采样、适用结构、全原子回映射 | 只有全原子采样成本确实成为瓶颈时再评估 |

OpenMM、GROMACS 和 AMBER 是模拟引擎/软件；OL3 是 RNA 力场；PLUMED 是采样插件。先选一套引擎跑通。若实验室已有可复用的 GROMACS/AMBER RNA 协议，可以替代 OpenMM，无需同时学习三套。

DynaRNA 的原始模型以每核苷酸 C4′ 一个点表示 RNA，训练结构为 5–200 nt；论文使用 Arena 做全原子重建。对 >200 nt 直接运行属于训练长度范围外的应用，不能假定有效。重建出的原子细节也可能把任务变成“修复重建误差”。以上是其 Methods/Discussion 的具体限制，和有没有论文是两回事。

BioEmu-2 支持核酸与全原子构象生成，但官方模型卡明确说明采样频率未校准，不能解释为 Boltzmann 概率/定量自由能。这里仅将其用于产生结构候选，不用它给 refinement 贴能量或动力学真值。官方列出最小测试设备为 80 GB A100，这不是本项目 H20 的实际最低显存要求；长链能否运行须实测。

**具体执行顺序**

1. **完成数值检查，建立修复后的基线。** 检查当前服务器代码是否与本地 FP32 图均值修复一致；运行现有 CUDA 检查与 BF16 合成 smoke test。通过后用同一 epoch49 权重、同一 val rank-1 做复评，先隔离推理路径变化。旧权重的训练历史仍有数值错误；需要后续从头训练的固定对照才能判断训练影响。不要把旧权重换修复代码后的结果当作“已修复重训”。

2. **检查现有困难候选并固定验证口径。** 复用现有分布分析，不重跑全部 15 万 PT 来重复已知结论。检查训练端原始 200 候选、`Data_PT_V2_RMSD_GT30/train` 和过滤日志；高 RMSD 只是待审候选，需排除映射错误/不完整监督/严重非物理结构。先做长度×误差分层采样对照，并限制每 PDB/同源簇曝光，不能只无限放大上述 5 个 PDB。训练标签 RMSD 可用于训练分桶，不作为部署筛选条件。

3. **从训练集选约 8 个小试验靶标。** 建议 2 个 ≤50 nt、2 个 51–200 nt、4 个 >200 nt，尽量分属不同结构/同源簇，覆盖 loop、多螺旋连接和螺旋相对位置等变化。先检查完整结构、标准核苷酸、断链/末端化学、配体和结构性离子依赖。全部来自 train；不扰动 val/test 再加入训练。长链整条保留，不能靠裁成短片段宣称补齐了长链全局安排。

4. **先建立便宜的结构化扰动对照。** 从训练 Protenix decoy 出发，对 loop/连接区内部坐标或螺旋间角度/距离做分级扰动，保持核苷酸内部几何并处理链连接，再用短程最小化/松弛消除不合理局部应力。不以整体刚体旋转/平移作为困难扩增；大幅独立原子高斯噪声容易只制造坏键。扰动强度先分弱/中/强，由训练和验证诊断确定目标区间，不按测试结果调参。

5. **跑通 OpenMM＋OL3 的一个体系。** 用完整源 CIF 建模，而非把 PT 中裁剪后的原子集合直接当作完整分子。加入氢、匹配的水/离子模型、周期盒，检查电荷和连通性，最小化并分阶段平衡。不能自动删除结构性 Mg²⁺，也不能不经参数检查把修饰碱基交给标准 OL3。原生结构修饰和 PT 的 canonical 映射应明确记录。先运行短 benchmark，记录总原子数、ns/day、显存、NaN/异常键和结构保持情况。

6. **扩展到有限多副本 MD。** 首轮建议每靶标两个不同速度种子的 5 ns 试验，8 个靶标上限约 80 ns，不含平衡。这是初始工作量建议，须按前一步吞吐缩减，不能承诺四张 H20 一定在固定时间完成。其用途是观察能否产生有用扰动，不证明平衡收敛，更不保证找到 >10 Å 的全局变化。先测普通 MD；若主要得到重复局部波动，转向结构化扰动＋物理松弛，或在下一轮用螺旋夹角/距离/接触 CV 做增强采样，不靠无计划升温展开 RNA。

7. **对相同靶标做 BioEmu-2 小对照。** 先用一个中短链和一个 >200 nt 靶标检查参数化输入、输出原子覆盖、运行时和显存，再每靶标生成数十个候选。若在当前时间预算内安装/显存/结构质量不合格，记录原因而不无限调环境。DynaRNA 可在 ≤200 nt 靶标上做辅助对照，重建和松弛后的误差应单独报告。

8. **筛选候选，而不是直接把轨迹帧塞进训练。** 去 PBC、保持 RNA 完整，去除模拟水/氢后按链/残基/原子名映射回模型重原子，统一 nm→Å。检查键/角、碱基平面、糖环、手性、碰撞、连通性；将相邻帧聚类后选代表。统计到 native 的观测原子 RMSD、局部 lDDT、碱基配对/堆积，以及螺旋相对位置等结构量。弱/中/强样本同时保留，不能按 RMSD 大小单独判定有效性。将明显替代生物学状态、体系环境改变或完全解折叠样本独立标记，不能自动当作对同一个实验构象的合理监督。

9. **确定置信度协议，再转换 PT。** 参见下文。此步骤未完成前，生成结构只进入独立候选目录，不混入正式 `Data_PT_V2/train`。

10. **用固定预算验证训练收益，再扩大规模。** 至少保留：A 修复后原始数据基线；B 原始数据＋结构化扰动；C 原始数据＋MD；D 原始数据＋BioEmu-2。先筛选有用来源，再做少量对照训练，保持更新步数、采样预算、验证集合和模型设置一致。旧 checkpoint 上短程继续训练只能作为预试验，不能代替修复后从头训练的因果对照。关注逐 RNA 宏平均改善、困难长链改善、好输入退化率和局部几何；少量目标/短训练不能支持最终性能结论。

**接入当前模型时要先解决的两件事**

第一件是置信度。当前 pLDDT 和 PAE/PDE/contact_probs 描述原 Protenix 预测，MD/BioEmu-2 新坐标并没有相应的 Protenix 置信度。若保留原数值，应明确将其解释为“原始预测先验”，并做对照检验；不得声称它是新坐标的置信度，也不能按 native 误差反推伪造 pLDDT。在现有实现里，只设 `confidence_conditioning=false` 仍不足以移除置信度：mobility gate 还会读入 `atom_plddt` 和 `atom_mobility_attr`。后续需要在网络与 gate 两条路径统一增加缺失/屏蔽协议，并对所有数据来源采用一致语义，避免数据来源成为捷径。第一轮可以先只生成并诊断候选，再实施该协议；不能混杂多个模型改动后把收益全部归因于 MD。

第二件是监督坐标系。现有 `build_refinement_pt.py` 把 native 在 observed atoms 上 Kabsch 到对应 source，并把无法监督的位置通过 `target_mask` 排除。替换 source 后，应使用稳定原子映射重新对齐 native、更新 RMSD/变换及扩增元数据；不能只改 `pos_pred`，保留与原 source 不一致的 `pos`。缺失实验原子仍不可成为伪监督。当前动态半径图会按输入重算，化学键拓扑则需正确保留。

建议扩增记录包含：原 PDB/同源簇、parent sample、生成器与版本、随机种子、副本/时间、温度/偏置、力场/水/离子、原子映射版本、置信度来源与可用掩码、筛选理由。MD 附加氢/水不直接进入现有 PT。

**4 张 H20、1–2 天的首轮安排**

| 时段 | 工作 | 交付与继续条件 |
|---|---|---|
| 0–4 小时，目标预算 | 数值检查、val 基线复评、筛选试验靶标；单体系 MD 吞吐与 BioEmu-2 环境检查 | 有通过记录、真实吞吐/显存和靶标清单；失败先修复，时段不是完成保证 |
| 后续至约 24 小时 | 结构化扰动＋有限 MD；BioEmu-2 在同靶标生成候选 | 先看能否产生通过 QC 的新错误；按实测吞吐调整时长 |
| 24–48 小时 | 聚类、原子映射、误差分桶与可视化，确定置信度协议/下一轮训练配置 | 产出候选质量与覆盖报告，选值得继续的方法 |

四张卡按独立靶标/副本分配，优先一任务一卡，不默认四卡模拟同一条 RNA。基线检查与小试验可在资源允许时交错进行，但使用独立采样环境，避免替换现有训练 CUDA/PyTorch。BioEmu-2 的多卡 context parallel 留待确有单卡内存不足时再评估。

1–2 天的目标是完成可行性筛选和数据协议，完整重新训练可能超出窗口。MD/BioEmu 安装、PT 扩增适配和置信度协议目前尚未为本方案实现，不能把下面已有检查命令当作已具备完整扩增流水线。

**现在可以执行的已有检查命令**

先在服务器记录设备信息：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

以下假设已按现有 [FP32 修复说明](../Code_Flow_matching/scripts/FP32_GRAPH_MEAN_FIX.md) 同步修复文件、激活原训练环境。若尚未上传，先照该说明打包并由用户上传；本轮没有生成新的修复压缩包。

```bash
cd "$HOME/Code_Flow_matching"
CUDA_VISIBLE_DEVICES=0 python scripts/check_bf16_graph_mean.py --device cuda --assert-fixed
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_synthetic.py --device cuda --bf16
```

只有数值检查和 smoke test 均通过，才继续下列 val-only 复评。输出路径必须是未使用的新目录：

```bash
cd "$HOME/Code_Flow_matching"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_refinement_v3.py \
  --config config/RNA_train_residual_v3.yaml \
  --checkpoint 'checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt' \
  --data-dir "$HOME/Data_PT_V2" \
  --split val \
  --output-dir evaluation/md_preflight_20261110_val \
  --batch-size 1 --num-timesteps 1 --precision bf16 --progress-every 5 --physics
```

提供设备查询输出、检查日志和 val 输出目录即可继续安排实际采样。这里没有新增虚构的 MD 命令。已有完整检查脚本与回归测试仍按 FP32 修复说明执行，上述是第一步数值与模型检查加 val 复评的简短入口。

历史诊断已使用过原 test；后续方法与采样比例只用训练/验证确定。若要得到新的独立最终泛化结论，应准备与开发过程隔离的终测集，并保留原 test 的历史记录。验证集中还记录了与训练完全同序列的 21/103 个 PDB，报告总体结果时同时保留 unmatched-exact-sequence 面板；该面板也不等价于严格去同源。

**本轮文件与执行情况**

仅新增本说明文件 `Code_Audit/RNA_MD_AUGMENTATION_PLAN_20261110.md`。它不参与训练，可直接删除，建议保留作为实验规划记录。无需上传此报告即可运行现有检查；本轮未新增采样脚本、未安装软件、未修改数据/模型/配置，未在服务器执行命令。
