# 数据生成

## 对齐
提取的prediction的原子坐标应与native的原子坐标对齐，并且做Kabsch对齐

## edge_attr与node_attr & 模型实际上没有使用 RNA-FM
注意输出字段edge_attr与node_attr；并且配置中写好node_attr_dim与edge_attr_dim。目前node_attr有RNA-FM生成的640维残基特征向量广播得到[node_atoms,640] 

[//]: # (ToDo)
edge_attr目前打算根据边的种类进行独热编码，边的种类与构建还未确认 

## 当前图不是描述中的“RNA 化学图”
当前生成图的脚本是之前版本的脚本，现在生成应当1.永久保留共价键 2.区分共价键与磷酸二酯键 3.保留原子所属残基以及具体的atom_name chain_id不用，因为目前数据集只专注于一条链的情况

# 模型修改

## 训练目标的质心处理不一致 DONE
已经统一为“x0_c = center_of_mass(x0, batch=batch)  x1_c = center_of_mass(x1, batch=batch)  u_t = x1_c - x0_c + ...”

## sequence = data['sequence'], DONE
逗号已经删除

## evaluate 当前无法按接口正常调用 DONE
当前evaluate是之前版本的脚本，等到当前版本模型训练好了之后，再写新的evaluate脚本

## 网络没有显式看到原始预测结构
应加入 source encoder，显式输入：
$$
\Delta x_t = x_t - x_{\mathrm{pred}},\quad
d(x_t),\quad d(x_{\mathrm{pred}})
$$

并在每个 block 中与当前状态交互。

Q:当前的模型中具体如何修改代码？

## 静态radius graph不适合中尺度 refinement
图由初始预测坐标构建并在50步中固定：
  - native 中应形成的新接触不在图中；
  - 错误初始接触会长期占据图；
  - 只有 6 层局部 message passing，长 RNA 的 helix/domain 之间难以通信。

建议拆为：
  - 永久 covalent graph；
  - sequence-neighbor graph；
  - 动态 radius graph；
  - residue-level base-pair/stacking/global graph。

Q:这些图什么含义？如何形成？当前的模型中具体如何修改代码？

## 自由 Cartesian 原子流没有化学约束
线性插值本身可能穿过：

  - 非法键长和键角；
  - 糖环变形；
  - 碱基非平面；
  - backbone chain break；
  - steric clash。

当前 loss 只有 velocity MSE，因此即使 RMSD 改善，也可能得到化学上更差的 RNA。

## stochastic path未必适合当前确定性任务

(\sigma_t=\sigma\sqrt{t(1-t)}) 的导数在两端趋于无穷。当前采样到 1e-4/0.9999 附近时，noise velocity 可明显大于真实 correction。

建议首先消融：
  - sigma=0 的 deterministic paired rectified flow；
  - 当前 sigma=0.1；
  - 有界导数的 (\sigma t(1-t))。

  如果每个 pos_pred 只有一个 native，必须加入“一步 residual EGNN”基线，否则 reviewer 很容易质疑为什么需要 50-step flow，
  而不是直接预测 pos-pos_pred。

# 最值得做的模型创新
  我建议把最终模型定义为：
  > Confidence-Gated Interaction-Aware Residual Flow for RNA Refinement

  ## 创新一：prediction-conditioned residual flow
  不仅令 x0=pos_pred，还让整个轨迹显式条件于：
  - 原始 pos_pred；
  - x_t-pos_pred；
  - atom role、residue identity、chain/residue index；
  - Protenix single/pair embeddings；
  - pLDDT、PAE/PDE、contact probability。

  AF3 官方输出本身提供 per-atom pLDDT、token-pair PAE、contact probability、多 seed/sample，以及可选 single/pair embeddings，因此这些输入具有直接数据来源。AlphaFold 3 输出说明

  ## 创新二：learned mobility gate + no-regret refinement

  预测每个残基/原子的可移动程度：

  [
  m_i=\sigma(\mathrm{MLP}[
  pLDDT_i,\ PAE_i,\ clash_i,\ localError_i,\ h_i])
  ]

  [
  v_i^{final}=m_i v_i
  ]

  并加入：
  - native/native identity pair；
  - 已非常接近 native 的输入；
  - 高置信正确区域 mask；
  - per-residue improvement supervision；
  - QA head 预测 refined 是否真的优于 raw。

  这能把课题从“生成一个新坐标”提升为 refinement 特有的“尽量不修坏正确区域”。

  ## 创新三：两级 RNA 图与 frame/torsion flow

  第一阶段可先做五点或关键原子表示：

  - P
  - C4′
  - C1′
  - 两到三个定义碱基方向的原子/centroid

  之后升级到：

  [
  SE(3){\text{nucleotide frame}}\times \mathbb T{\text{torsion}}
  ]

  RNA-FrameFlow 已证明 C3′/C4′/O4′ frame + 8 torsions 是可行表示，因此“RNA frame”本身不是创新，但把它用于 prediction-
  conditioned correction flow 仍然很合适。RNA-FrameFlow

  ## 创新四：interaction-coordinate co-refinement

  增加 residue-pair head，联合预测：

  - Watson–Crick；
  - wobble；
  - non-canonical pair；
  - stacking；
  - no interaction。

  每若干 ODE 步用 soft interaction logits 更新 message-passing bias。重点不是简单“把二级结构作为输入”，而是让
  interaction graph 和坐标在 flow 中共同修正。

  ## 5. 最新研究对 novelty 的约束

  截至 2026-09-01，以下表述已经不能使用：

  - “首次将 Flow Matching 用于 RNA”：RNA-FrameFlow、RNAbpFlow、RiboGen 都已经做过；
  - “首次 base-pair-aware RNA flow”：RNAbpFlow 已经是 base-pair-conditioned SE(3) flow matching。RNAbpFlow
  - “首次 AI RNA refinement”：RNArefine 已用几何网络预测 pairing/stacking，再结合 Monte Carlo 和 L-BFGS 做全原子精修。
    RNArefine

  - “首次等变生成模型做 RNA refinement”：ChironRNA 已用 E(3)-equivariant diffusion 做 clash 区域重生成和缺失原子补全。
    ChironRNA

  - “首次全原子 RNA flow”：RiboGen 已做 sequence/structure all-atom co-generation。RiboGen

  比较稳妥的创新表述是：

  > 面向 Protenix/AF3-like predictor errors 的 prediction-conditioned RNA correction flow，通过学习型置信度门控保护正确
  > 区域，并联合更新 RNA interaction topology、nucleotide geometry 和 refinement QA。

  这是比单纯“把起点改为 pos_pred”强得多的贡献组合。

  新文献还提供两个很实用的补充方向：

  - ChironRNA 的“高质量区域锚定 + 错误区域局部重生成”可以直接转化为 confidence mask；
  - FlowMol3 发现 self-conditioning 和 train-time geometry distortion 能增强 flow 对推理轨迹分布漂移的纠错能力。对你的模
    型，可让每一步回馈预测终点 (\hat x_1)，并构建 helix rotation、suite flip、base-pair break、pucker error 等 RNA error
    bank。FlowMol3

  ## 6. 推荐实施顺序

  1. 修正 schema、质心监督、evaluate 和 graph；
  2. 做三条可信 baseline：
      - direct residual EGNN；
      - deterministic paired flow；
      - 当前 stochastic paired flow；

  3. 加 explicit source conditioning；
  4. 加 pLDDT/PAE/contact 与 mobility gate；
  5. 加 identity/no-op 数据和 QA accept-reject；
  6. 改为 key-atom/frame/torsion 表示；
  7. 加动态 base-pair/stacking graph；
  8. 最后再研究 FlowCast、few-step distillation 和多候选 consensus。

  数据切分必须同时控制结构相似性，不能只按序列去重。RNA3DB 专门指出，RNA 数据稀少且结构重复会显著夸大测试性能；2026 年的
  新 benchmark 也显示预测精度强烈依赖与训练集的结构相似度。RNA3DB、2026 RNA benchmark

  最终评价不要只报平均 RMSD，至少还应包括：

  - TM-score、lDDT；
  - INF_all/WC/NWC、stacking recovery；
  - clashscore、bond/angle violation、suite/pucker/χ outlier；
  - 相对 Protenix raw 的 improvement rate；
  - good-input damage rate；
  - raw/refined 经 QA 选择后的实际收益；
  - 按 initial RMSD、长度、pLDDT、训练集结构相似度分层。

  现有 Code_Flow_matching/RNA_refinement_novelty_and_model_proposal.md:1 的总体方向是正确的，但其中“atomic_numbers key
  前有空格”和“仍使用平方距离”两项已经在当前代码中修复；Dataset/evaluate/schema、静态图和未使用 RNA 特征等问题仍然存在。
  

完全可以，而且我认为这才是当前阶段更稳妥的路线。当前 TorchMD-ET 的表达能力足以先验证“Protenix → native refinement”是否
  成立；目前瓶颈主要不是 backbone，而是输入条件、图结构、RNA 几何先验和损失函数。

  建议按以下顺序修改。

  1. 先修正确性问题

  - 统一居中后的 x0/x1/u_t；
  - 统一 Dataset、配置和 .pt 字段；
  - 修复 evaluate.py 的 batch.z、edge_index/bond_index 和 sample_files；
  - 推理后恢复 pos_pred 的原始质心；
  - 保留共价键，不能让 radius graph 覆盖化学图。

  2. 显式条件化原始预测结构

  当前网络只接收 x_t。加入：

  [
  \Delta x_t=x_t-x_{\text{pred}}
  ]

  但不能把三维向量直接拼到普通 scalar node feature。可以先对 pos_pred 单独编码：

  source graph(pos_pred) ── TorchMD encoder ── source scalar/vector features
  current graph(x_t)     ── TorchMD encoder ── current scalar/vector features
                                        ↓
                                  等变特征融合
                                        ↓
                                   velocity head

  低成本版本可以共享两条分支的权重，或只增加 2–3 层 source encoder。

  3. 真正打开 RNA 特征

  目前配置中的 node_attr_dim: 0 关闭了 RNA-FM。建议 node feature 至少加入：

  - A/C/G/U；
  - atom name/atom role；
  - residue index、chain id；
  - RNA-FM embedding；
  - Protenix atom pLDDT；
  - 当前原子的 clash/geometry violation；
  - 是否缺失或重建原子。

  不要直接把 640 维 RNA-FM 在每个原子上反复使用。先投影：

  rna_feature = rna_projection(rna_fm)  # 640 -> 32/64

  同一残基的原子可以共享 RNA-FM，但应额外加入不同的 atom-role embedding。

  4. 改造成多类型动态图

  当前 TorchMD 网络可以继续使用，只需要改善送进去的边：

  covalent edges：永久保留
  sequence edges：相邻核苷酸
  spatial edges：根据当前 x_t 动态更新
  interaction edges：base pair / stacking candidates

  建议 edge_attr 使用 one-hot 或 embedding：

  [covalent, intra-residue, phosphodiester,
   spatial, base-pair, stacking, cross-chain]

  无需每个 ODE step 都重构动态图，可以每 2–5 步更新一次，降低成本。

  5. 加 confidence mobility gate

  在当前 vector head 后增加一个标量 head：

  [
  m_i=\sigma(\operatorname{MLP}(h_i,pLDDT_i,PAE_i,\text{violation}_i))
  ]

  [
  v_i^{final}=m_i v_i
  ]

  这里标量乘三维向量不会破坏等变性。

  高置信正确区域的 m_i 应接近 0，错误 loop/junction 的 m_i 应较大。加入 identity pairs：

  x0 = native, x1 = native

  要求模型输出零速度，可以显著降低 over-refinement。

  6. 加 RNA-specific loss

  保留 flow matching loss：

  [
  L_{\mathrm{FM}}=|v_\theta-u_t|^2
  ]

  再加入辅助损失：

  [
  L=L_{\mathrm{FM}}
  +\lambda_bL_{\mathrm{bond}}
  +\lambda_aL_{\mathrm{angle}}
  +\lambda_cL_{\mathrm{clash}}
  +\lambda_pL_{\mathrm{plane}}
  +\lambda_{\mathrm{bp}}L_{\mathrm{basepair}}
  +\lambda_sL_{\mathrm{stacking}}
  ]

  第一版建议只加入：

  - bond length；
  - clash；
  - base plane；
  - base-pair distance/orientation。

  pucker、suite、torsion 可以后续添加，避免第一次修改过多。

  辅助几何 loss 最好作用在模型估计的终点：

  [
  \hat x_1=x_t+(1-t)v_\theta(x_t,t)
  ]

  而不是直接作用在瞬时速度上。

  7. 加终点 self-conditioning

  每一步预测：

  [
  \hat x_1^{(k)}
  ]

  下一步将上一次的终点估计重新编码为条件：

  x_t + previous_endpoint_estimate → velocity network

  这有助于模型发现 ODE 轨迹已经偏离合理 RNA 几何，并在后续步骤纠正。改动比更换 EquiformerV2 小。

  8. 改进采样器

  当前只有 Euler：

  x = x + dt * v_t

  可以先换成 Heun：

  v1 = model(x, t)
  x_euler = x + dt * v1
  v2 = model(x_euler, t + dt)
  x = x + 0.5 * dt * (v1 + v2)

  Heun 每步需要两次网络调用，但通常能用更少步数。例如比较：

  - Euler 50 steps；
  - Heun 20 steps；
  - Heun 10 steps。

  ## 推荐的最小创新版本

  在不更换现有 backbone 的情况下，先实现：

  当前 TorchMD-ET
  + explicit source conditioning
  + RNA-FM/atom-role/pLDDT
  + covalent + dynamic spatial graph
  + confidence mobility gate
  + identity/no-op training
  + bond/clash/base-pair auxiliary loss
  + QA accept/reject

  这个方案已经足以形成有辨识度的模型。创新点不必依赖更大的网络，而可以来自：

  > 同一个轻量等变速度网络，通过 predictor-specific source conditioning、置信度门控和 RNA interaction constraints，实现
  > no-regret RNA refinement。

  最重要的是先证明：

  - 相比 Protenix raw，更多 target 得到改善；
  - 原本较好的结构不会频繁被修坏；
  - INF、clash 和 RNA stereochemistry 与 RMSD 同时改善；
  - 同参数量 direct regression 无法达到同样效果。

  因此答案是：可以，而且建议先在当前模型上改。EquiformerV2 可以保留为以后的一项 backbone ablation，而不是当前工作的前置
  条件。
