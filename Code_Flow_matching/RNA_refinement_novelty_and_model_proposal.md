# RNA 结构 refinement：novelty 收敛与模型方案

> 基于 `20260810_105218_faa1c0_report_4.md`、报告所引文献、截至 2026-08-27 的补充检索，以及当前 `Code_Flow_matching` / 数据管线审阅。

## 1. 先给结论

建议不要把课题表述成泛泛的“把 ETFlow 从小分子改成 RNA”，也不要把“RNA + Flow Matching”或“AI-guided RNA refinement”单独作为 novelty。更稳妥且更有辨识度的主线是：

**Confidence-Gated, Interaction-Aware Residual Flow Matching for RNA Structure Refinement**

中文可表述为：

**面向预测误差的置信度门控、相互作用感知 RNA 残差流精修。**

核心任务定义为：给定 Protenix/AF3 类模型产生的结构、原子/残基置信度和 RNA 序列，学习从预测结构到实验结构的、区域自适应的 correction flow；正确区域尽量保持不动，错误 loop/junction/非规范相互作用区域允许更强修正，同时显式保证 RNA backbone、糖环、碱基配对和堆叠几何。

推荐把论文主张收敛为三个互相支撑的点：

1. **预测器条件化的 paired residual flow**：不是从高斯噪声生成 RNA，而是从 Protenix prediction 直接输运到 native；同时把原始预测结构作为显式条件保留到整个轨迹中。
2. **置信度门控的 no-regret refinement**：利用 Protenix 的 atom pLDDT、token-pair PAE/PDE、局部 clash 和 interaction violation 学习逐区域 mobility gate，解决“本来正确的区域被修坏”的 refinement 特有问题。
3. **连续几何与 RNA 相互作用拓扑联合建模**：用 nucleotide frame + torsion 表示连续结构，并联合预测 base-pair/base-stacking 图；相互作用图既是监督目标，也动态参与 message passing 和采样 guidance。

其中，第 1 点是任务区别，第 2 点是 refinement-specific 方法创新，第 3 点提供 RNA-specific 机制。只做第 1 点可以形成 baseline，但不足以成为强 novelty。

---

## 2. 原报告必须补上的两篇“正面相邻工作”

### 2.1 RNA-FrameFlow：已经存在 RNA + Flow Matching

[RNA-FrameFlow](https://arxiv.org/abs/2406.13839) 已将 SE(3) Flow Matching 用于 de novo RNA backbone generation。它用每个核苷酸的 `C3′/C4′/O4′` 构造刚体 frame，其他 backbone 原子由 8 个 torsion 重建；最优模型对 40–150 nt RNA 进行无条件生成。

它与本课题的边界是：

| 维度 | RNA-FrameFlow | 本课题应当做的内容 |
|---|---|---|
| 任务 | 无条件 de novo backbone generation | prediction-conditioned refinement |
| 起点 | Gaussian translation + uniform SO(3) | Protenix/AF3-like predicted structure |
| 输出目标 | 合理且多样的新 backbone | 与输入同一条 RNA 的更准确结构 |
| 关键风险 | validity、chain break、clash | over-refinement、错误区域定位、纠错上限 |
| 条件 | 主要是长度/生成设定 | sequence、原结构、confidence、interaction graph |
| 评价 | designability、novelty、local descriptors | 每个 target 相对输入的 ΔTM/ΔlDDT/ΔINF/几何改善 |

因此，不能再声称“首次把 Flow Matching 用到 RNA”。可以把 RNA-FrameFlow 的 frame/torsion 表示作为强先验，并将新意放到 **conditional correction、confidence gating 和 interaction co-refinement**。

### 2.2 RNArefine：已经存在 AI-guided atomic-level RNA refinement

2026 年 6 月发布的 [RNArefine](https://www.biorxiv.org/content/10.64898/2026.06.26.734804v1) 是当前最直接的同任务工作。其流程为：

1. 几何注意力网络预测 base pairing / base stacking；
2. 将预测相互作用作为 restraint；
3. Monte Carlo conformational sampling；
4. L-BFGS energy optimization。

作者报告它能改善 stereochemistry、interaction fidelity 和带物理惩罚的结构分数，并在 CASP16 top 30 groups 中改善 28 组的 ranking score。

因此，不能再笼统声称“首次 AI-guided RNA refinement”。本课题相对 RNArefine 的差异应明确写为：

- RNArefine 是 **AI restraint + sampling/optimization**；本课题是 **端到端 learned transport/correction flow**。
- RNArefine 强项是 atomic relaxation 和物理合法性；本课题应证明除局部几何外还能学习 **predictor-specific systematic error**，并在少量 NFE 内进行中尺度坐标纠正。
- RNArefine 默认所有原子参与优化；本课题用 **confidence/mobility gate** 显式保护正确区域。
- RNArefine 的相互作用先预测再固定使用；本课题可把 interaction graph 与坐标在轨迹中 **联合更新**。
- RNArefine 的运行可达到 MC 7200 s + L-BFGS 3600 s 上限；learned flow 应报告质量–时间 Pareto 和 1/5/10/20 NFE 的结果。

RNArefine 必须进入主 baseline，而不是只在 related work 中提一下。

---

## 3. 对原报告各“创新点”的取舍

### 保留并升级

#### A. AF3/Protenix prediction → native coupling

方向正确，而且当前代码已经实现了基本形式：`x0 = pos_pred, x1 = pos`。但这只能称为任务构造或 baseline，不能单独作为最终 novelty。需要升级为：

- 显式条件化：网络在所有时间都同时看到 `x_t` 和原始 `x_pred`，而不是只从 `x_t` 间接推断起点；
- frame/torsion 流，而非自由 Cartesian 原子流；
- confidence-gated path/mobility；
- identity/no-op examples，训练模型在输入已经正确时输出近零修正。

#### B. confidence-aware refinement

这是最值得做成核心创新的点。Protenix 本地代码已经输出：

- `atom_plddt`；
- `token_pair_pae`；
- `token_pair_pde`；
- chain-level confidence 和 ranking score。

这些信息目前没有进入 `.pt` 数据，也没有进入 refinement 网络。建议把它从“步长调度小技巧”升级成统一的 **error localization + mobility control + accept/reject calibration**。

#### C. RNA-specific geometry / base pair / stacking

必须保留，但不要只加几个 penalty。更有说服力的方式是让相互作用图成为模型状态和条件的一部分，几何 loss 负责保证局部合法性，interaction head 负责修复中远程 RNA 拓扑。

#### D. QA / 是否 refine 判别

值得保留。refinement 最关键的统计量不是平均 RMSD，而是：

- improvement rate；
- worst-case regression；
- 原结构较好时的 damage rate；
- QA 选择 `raw vs refined` 后的最终收益。

### 降级为第二阶段或工程优化

#### FlowCast speculative sampling

先不作为主创新。当前最大问题是模型是否能稳定改善 RNA，而不是 50 步能否加速到 10 步。只有在 accuracy baseline 建立后，再报告 velocity reuse / Heun / adaptive solver 的质量–速度曲线。

#### TwinFlow one-step generation

暂不进入第一版。它来自大规模视觉生成，训练复杂度较高，而且 one-step 不能自动解决 RNA stereochemistry。先证明 5–20 step residual flow 有效，再考虑 consistency distillation。

#### FlowBind shared latent

“sequence/confidence/2D/coordinates 都是不同 modality”的类比过于宽泛。当前直接做 feature fusion 或 cross-attention 更清楚，也更容易消融。除非后续确实要支持任意缺失模态，否则不建议作为论文主线。

#### non-Markovian path

概念过宽，不易形成可验证贡献。可以用具体的 multi-scale frame state 取代：核苷酸 frame、helix rigid-body、torsion 分层更新。

---

## 4. 推荐模型：CGIR-Flow

暂定名：**CGIR-Flow**，即 **Confidence-Gated Interaction-aware Residual Flow**。

### 4.1 输入

对第 `i` 个核苷酸和第 `ij` 对核苷酸构造：

#### 逐核苷酸/逐原子特征

- nucleotide identity：A/C/G/U、修饰标志；
- atom identity、atom role；
- RNA-FM residue embedding；
- Protenix atom pLDDT；
- 当前结构的局部 clash、bond/angle violation、suite/pucker indicator；
- residue index、chain id、是否缺失原子。

#### pair 特征

- Protenix token-pair PAE/PDE；
- 序列距离；
- 当前 3D 距离和相对 frame orientation；
- covalent/sequence-neighbor 类型；
- 初始预测结构推断的 canonical/non-canonical base pair、stacking 概率；
- 可选二级结构或 base-pair probability。

#### 几何状态

每个核苷酸用：

\[
T_i=(R_i,t_i)\in SE(3), \qquad \phi_i\in \mathbb{T}^{8}
\]

其中 `T_i` 由 `C3′/C4′/O4′` frame 构造，`φ_i` 描述 backbone/sugar 相关 torsion。完整原子坐标通过模板键长/键角和 torsion 重建。这样可以避免当前模型把每个原子当作自由点而产生大量非法几何。

第一版若改 frame 表示工作量过大，可先用 `P/C4′/C1′/base centroid` 四点 coarse representation 验证主假设，再进入 frame + torsion。

### 4.2 显式 source conditioning

当前代码只向网络提供 `x_t`。建议同时编码：

\[
h_i^{src}=Encoder(\hat T_i,\hat\phi_i, c_i),
\]

并在每层与当前轨迹状态交互：

\[
h_i^t=Block(h_i^t,h_i^{src},h_{ij}^{pair},t).
\]

这让模型在轨迹后期仍知道“输入原来是什么、已经修了多少”，也便于学习 residual magnitude 和 no-op 行为。

### 4.3 置信度门控 mobility

定义每个残基的可移动程度：

\[
m_i=\sigma(MLP[pLDDT_i,PAE_i,clash_i,interactionViolation_i,h_i^{src}]).
\]

将模型预测速度写为：

\[
v_i^{final}=m_i v_i^{learned}.
\]

对 pair/domain 运动可再定义 `m_ij`。训练时用真实局部误差构造辅助标签，例如 Kabsch 对齐后的 per-residue lDDT error 或 frame error，监督 `m_i` 具有误差定位能力。

门控的意义不是机械地令高 pLDDT 不动，而是学习“Protenix confidence 在 RNA 上何时可信”。这比固定阈值更合理，因为 AF3/Protenix confidence 在不同 RNA fold 上可能失准。

### 4.4 paired geometric residual flow

令预测结构为 `\hat T, \hat φ`，native 为 `T*, φ*`。translation 采用线性或 OT path，rotation 使用 SO(3) geodesic，torsion 使用圆周最短路径：

\[
t_i(s)=(1-s)\hat t_i+s t_i^*,
\]

\[
R_i(s)=\hat R_i\exp\left(s\log(\hat R_i^{-1}R_i^*)\right),
\]

\[
\phi_i(s)=\hat\phi_i+s\,wrap(\phi_i^*-\hat\phi_i).
\]

网络分别回归 translation、rotation tangent 和 torsion velocity。与 RNA-FrameFlow 的主要区别是 source 不再来自 Gaussian/uniform prior，而是同一 RNA 的预测结构，而且 source features 始终作为条件。

### 4.5 interaction graph 与坐标联合精修

网络增加 interaction head，预测每对核苷酸的：

- Watson–Crick pair；
- wobble / non-canonical pair；
- stacking；
- no interaction。

每若干 flow step 用当前坐标和 interaction logits 更新 message-passing edges，但共价 backbone edge 永久保留。可使用 soft edge bias，避免离散采样不可导。

损失为：

\[
\mathcal L =
\lambda_T \mathcal L_{trans-FM}
+\lambda_R \mathcal L_{rot-FM}
+\lambda_\phi \mathcal L_{torsion-FM}
+\lambda_I \mathcal L_{interaction}
+\lambda_G \mathcal L_{geometry}
+\lambda_M \mathcal L_{mobility}
+\lambda_Q \mathcal L_{QA}.
\]

`L_geometry` 至少包含：

- bond length / bond angle；
- steric clash；
- base plane；
- sugar pucker；
- suite/torsion legality；
- predicted/native base pair and stacking geometry。

与 RNArefine 的差异在于，这些 interaction 不只是最后优化的固定 restraint，而是在 learned transport 过程中参与结构更新。

### 4.6 no-regret QA 与输出选择

增加 QA head 预测：

- `ΔTM-score`；
- `ΔlDDT`；
- `ΔINF_all/WC/NWC/stacking`；
- clash/bond/angle 是否改善；
- per-residue improvement probability。

推理时输出：

\[
X_{final}=
\begin{cases}
X_{refined}, & \widehat{\Delta Q}>\tau,\\
X_{pred}, & \text{otherwise}.
\end{cases}
\]

也可采用连续 blend，但必须在 internal/frame space 中完成，直接混合 Cartesian 坐标可能破坏几何。

训练集中加入 native 轻扰动、接近 native 的 Protenix 样本以及 identity pair `(x,x)`，要求模型输出零或极小速度。这是控制 over-refinement 的必要训练信号。

---

## 5. 当前数据设计的关键问题

最新 split 报告为：

| split | PDB 数 | RMSD mean/median | RMSD P90 | length median/P90 |
|---|---:|---:|---:|---:|
| train | 774 | 3.18 / 2.36 Å | 7.32 Å | 33 / 119 nt |
| val | 117 | 9.83 / 6.92 Å | 20.23 Å | 60 / 417 nt |
| test | 97 | 13.46 / 8.18 Å | 36.21 Å | 59 / 487 nt |

这意味着当前训练和测试同时存在明显的 **error severity shift** 与 **length shift**：train 被限制为 RMSD ≤ 15 Å，而 val/test 不限制；train 也显著更短。

如果不处理，模型实际上学的是 local correction，却被要求在测试时解决大量 wrong-fold / large-domain error。建议二选一：

### 方案 A：明确定位 local refinement（第一阶段推荐）

- 主结果只报告 initial RMSD ≤ 15 Å、长度在训练支持范围内的 target；
- >15 Å 作为 out-of-domain stress test；
- 模型输出 applicability score，低置信 OOD case 可以 abstain；
- 论文中明确“不声称从错误 fold 重新折叠”。

### 方案 B：扩充大误差训练分布

- 不对 train 用 RMSD ≤ 15 Å 硬过滤，或为大误差样本单独训练 coarse/domain stage；
- 加入 RNA-specific synthetic corruption；
- 对长 RNA 使用 crop + full-structure context 或分层 helix/domain graph；
- 训练时按 PDB 而不是按 decoy 均衡采样，避免多 seed/sample 的 target 被过度加权。

### 结构泄漏控制

当前 test 使用 80% sequence identity + 双向 80% coverage 去冗余，这是必要但不充分。2026 年独立 benchmark 显示 RNA 预测精度与训练集结构相似性高度相关；该 benchmark 对结构 TM-score > 0.7 的冗余也进行了控制。

建议增加：

- 用 RNA-align 计算 test 到 train/val 的最大 TM-score；
- 主表按 `<0.3 / 0.3–0.5 / >0.5` 或至少 hard/easy 分层；
- 设一个结构严格 test subset；
- 保留时间切分，且禁止用 native RMSD 选择 Protenix candidate。

### 训练样本扩充：RNA error bank

native 数量有限，可以从 native 或高质量预测出发构造带标签的 RNA-specific corruption：

- helix rigid-body rotation/translation；
- junction hinge rotation；
- base pair break / false pair；
- base stacking shift；
- suite rotamer flip；
- C3′-endo / C2′-endo pucker perturbation；
- phosphate backbone clash；
- chain break；
- loop local noise；
- global isotropic noise 作为对照。

每个 corruption 同时给出 error type 和 affected residue mask，辅助训练 mobility/error head。真实 Protenix pair 保证任务真实性，synthetic error bank 提供覆盖度和可解释性。还应加入其他 predictor 输出进行 cross-predictor test，防止模型只记住 Protenix 的误差风格。

---

## 6. 当前代码离上述方案还有哪些差距

以下不是风格问题，而是会阻断 baseline 或使实验结论不可信的问题。

### 6.1 数据 schema 与配置不一致

- `dataset.py` 强制读取 `sequence/edge_attr/node_attr`，但当前 `pt_align.py` 生成的字典只有 `pos/pos_pred/atomic_numbers/edge_index/...`。
- `sequence = data['sequence'],` 末尾逗号会把 sequence 变成单元素 tuple。
- config 中 `node_attr_dim: 0`、`edge_attr_dim: 0`，因此即使存了 RNA-FM 和 edge type，模型也会忽略。
- 当前 `.pt` builder 的 radius graph 没有保存 residue index、atom name、covalent bond type；后续无法定义 RNA-specific loss 或 nucleotide frame。

### 6.2 明确的调用错误

- `model.py` 使用 `batched_data[" atomic_numbers"]`，key 前多了空格；dataset 返回的是 `atomic_numbers`。
- `evaluate.py` 调用 `model.sample(edge_index=...)`，而 `sample` 参数名是 `bond_index`。
- `evaluate.py` 给 `BaseDataModule` 传入 `sample_files`，但 datamodule 构造函数没有该参数。

### 6.3 几何图和距离计算问题

- edge 只按初始 `pos_pred` 构图，并在整个 flow 中固定；新接触无法形成、错误接触无法移除。
- 当前 edge 没有稳定保留 covalent backbone connectivity 的独立通道。
- `TorchMD_ET_dynamics` 中 `edge_weight = sum(edge_vec**2)` 得到平方距离，但 RBF/cutoff 按普通距离范围初始化；随后 `edge_vec / edge_weight` 也不是单位方向。应核对并改为 `norm(edge_vec)`，同时做数值保护。

### 6.4 当前模型并未使用报告中提到的关键信息

- 没有 pLDDT/PAE/PDE；
- RNA-FM 实际被 config 关闭；
- 没有 base pair / stacking；
- 没有 torsion/pucker/suite/clash loss；
- 只有 flow vector MSE；
- evaluation 只有 all-atom Kabsch RMSD。

因此，近期最重要的不是立即加入 TwinFlow/FlowCast，而是先把 baseline 做成“数据 schema 正确、训练能跑、至少多指标评价、identity 不被修坏”的可信系统。

---

## 7. 最小可发表实验矩阵

### 7.1 Baselines

必须包含：

1. Protenix raw rank-1；
2. Protenix best-of-N oracle（仅作采样上限，不能当部署 baseline）；
3. 当前 Cartesian paired flow；
4. restrained minimization / 短程 MD；
5. QRNAS 或 BRiQ（按可运行性至少选择一个）；
6. **RNArefine**（最重要的同任务 baseline）；
7. 可选：RNArefine + 本模型，测试 learned coarse correction 与 physics relaxation 是否互补。

RNA-FrameFlow 是 representation/method inspiration，不是严格同任务 baseline；可以用其 frame 表示做消融。

### 7.2 Ablation

| ID | 模型 | 要回答的问题 |
|---|---|---|
| A0 | raw Protenix | 输入基线 |
| A1 | Cartesian paired FM | `pred→native` 本身是否有效 |
| A2 | A1 + explicit source conditioning | 是否比只输入 `x_t` 更稳定 |
| A3 | A2 + RNA-FM | sequence prior 是否有增益 |
| A4 | A3 + pLDDT/PAE/PDE | predictor confidence 是否有增益 |
| A5 | A4 + mobility/no-op gate | 是否降低好结构 damage rate |
| A6 | frame + torsion residual flow | 是否改善物理几何和 sample efficiency |
| A7 | A6 + interaction head/dynamic edges | 是否改善 INF、NWC 和 stacking |
| A8 | A7 + QA accept/reject | 最终系统是否实现 no-regret |

不要一开始同时加入所有模块；否则无法判断真正有效的点。

### 7.3 指标

#### 全局/局部结构

- RNA-align TM-score；
- RMSD；
- GDT-TS；
- lDDT。

#### RNA-specific interaction

- INF_all；
- INF_WC；
- INF_NWC；
- stacking recovery；
- pseudoknot / tertiary motif subset（样本允许时）。

#### stereochemistry

- clashscore；
- bond length / angle violation；
- backbone suite outlier；
- pucker correctness；
- glycosidic χ / base plane；
- chain break count。

#### refinement-specific（主表必须有）

- `Δmetric = refined - raw`；
- target improvement rate；
- good-input damage rate，例如 raw RMSD < 3 Å 后变差超过 0.5 Å 的比例；
- median 与 bootstrap 95% CI；
- paired Wilcoxon；
- 按 initial RMSD、length、max-train-structure-similarity、pLDDT 分层；
- wall time、NFE、GPU memory。

平均 RMSD 容易被少数灾难性 target 主导，不能作为唯一结论。

---

## 8. 分阶段实施路线

### Phase 0：可信 baseline（先完成）

1. 统一 `.pt` schema，加入 atom name、residue id、chain id、covalent edge、spatial edge、RNA-FM、Protenix confidence；
2. 修复 key/参数/距离计算问题；
3. 让 current paired flow 在小数据上 overfit 一个 batch；
4. 实现 raw vs refined 的 RMSD/TM/lDDT/INF/clash 评价；
5. 输出按 initial RMSD 分层的 win rate。

判定标准：不是训练 loss 下降，而是在 held-out local-refinement subset 上显著优于 raw，且 good-input damage rate 可控。

### Phase 1：主创新最小版

1. explicit source encoder；
2. atom pLDDT + token PAE/PDE；
3. mobility/no-op gate；
4. RNA-specific corruption mask supervision；
5. QA accept/reject。

这一阶段仍可使用 key-atom Cartesian 表示，目标是先验证 confidence-gated no-regret correction。

### Phase 2：RNA-specific geometric model

1. nucleotide frame + torsion 数据表示；
2. SE(3) × torus residual flow；
3. full-atom reconstruction；
4. geometry losses。

这一阶段直接吸收 RNA-FrameFlow 和 NuFold 的成熟表示，但任务和条件化方式不同。

### Phase 3：interaction co-refinement

1. base pair/stacking labels；
2. interaction head；
3. soft dynamic graph；
4. interaction-guided flow sampling；
5. 与 RNArefine 正面对比及串联实验。

### Phase 4：加速和 ensemble

只有 Phase 1–3 证明有效后，再做：

- 20/10/5/1 NFE；
- Heun/adaptive ODE；
- FlowCast 式 velocity reuse；
- multiple Protenix samples consensus；
- one-step consistency distillation。

---

## 9. 可对师兄直接汇报的版本

> 我重新核对了报告和最新文献。原方案里“RNA + Flow Matching”和“AI-guided RNA refinement”不能单独作为 novelty：RNA-FrameFlow 已经用 SE(3) Flow Matching 做 RNA backbone generation，2026 年的 RNArefine 也已经用几何网络预测配对/堆叠并结合 MC + L-BFGS 做全原子精修。我的方案会与这两类工作明确区分：输入不是随机噪声而是 Protenix 预测结构，学习 predictor-specific prediction→native correction；进一步用 Protenix 的 pLDDT/PAE/PDE 学习逐区域 mobility gate，保护高可信正确区域，并把 base-pair/stacking 图与 nucleotide-frame/torsion flow 联合更新。这样主线不是简单改 ETFlow，而是“置信度门控、相互作用感知、可 abstain 的 RNA residual flow refinement”。实验上会把 RNArefine 作为最强同任务 baseline，并报告 ΔTM/ΔlDDT/ΔINF、物理几何、improvement rate 和好结构 damage rate，而不是只报平均 RMSD。第一阶段先修通数据 schema 和 Cartesian paired-flow baseline，再依次消融 source conditioning、confidence gate、frame/torsion 和 interaction head。

---

## 10. 主要参考文献

1. [RNA-FrameFlow: Flow Matching for de novo 3D RNA Backbone Design](https://arxiv.org/abs/2406.13839)，TMLR 2025。
2. [RNArefine: AI-guided Atomic-Level Refinement of RNA Structures](https://www.biorxiv.org/content/10.64898/2026.06.26.734804v1)，bioRxiv 2026。
3. [Assessment of Nucleic Acid Structure Prediction in CASP16](https://pubmed.ncbi.nlm.nih.gov/41165252/)，Proteins 2026。
4. [Functional Relevance of CASP16 Nucleic Acid Predictions as Evaluated by Structure Providers](https://pubmed.ncbi.nlm.nih.gov/40905273/)，2025。
5. [Limits of deep-learning-based RNA prediction methods](https://doi.org/10.1093/nar/gkag813)，Nucleic Acids Research 2026。
6. [NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation](https://www.nature.com/articles/s41467-025-56261-7)，Nature Communications 2025。
7. [Geometric deep learning of RNA structure / ARES](https://pmc.ncbi.nlm.nih.gov/articles/PMC9829186/)，Science 2021。
8. [SE(3)-Stochastic Flow Matching for Protein Backbone Generation / FoldFlow](https://arxiv.org/abs/2310.02391)，2023/2024。
9. [Fast protein backbone generation with SE(3) flow matching / FrameFlow](https://arxiv.org/abs/2310.05297)，2023。
10. [Accurate RNA 3D Structure Prediction Using a Language Model-Based Deep Learning Approach / RhoFold+](https://doi.org/10.1038/s41592-024-02487-0)，Nature Methods 2024。
11. [DRfold2 is a deep learning-based tool that enables efficient and accurate RNA structure prediction](https://doi.org/10.1371/journal.pbio.3003659)，PLOS Biology 2026。
12. [RNA-Puzzles Round V: Blind Predictions of 23 RNA Structures](https://doi.org/10.1038/s41592-024-02543-9)，Nature Methods 2025。

## 11. novelty 表述边界

在完成更系统的 prior-art search 和实验前，论文中建议使用：

> To the best of our knowledge, we study the first prediction-conditioned geometric flow specifically trained to transport AF3-like RNA predictions toward experimentally determined structures, with confidence-gated regional mobility and joint interaction-aware refinement.

不要使用：

- “the first flow matching model for RNA”；
- “the first AI RNA refinement method”；
- “the first all-atom RNA refinement”；
- “solves wrong-fold RNA refinement”。

最后一句尤其重要：如果 initial global topology 错误，local refinement 通常无法恢复。模型应当能够识别并 abstain，而不是默认所有输入都能修好。
