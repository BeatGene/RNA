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

## 网络没有显式看到原始预测结构 Modify_1  DONE
应加入 source encoder，显式输入：
$$
\Delta x_t = x_t - x_{\mathrm{pred}},\quad
d(x_t),\quad d(x_{\mathrm{pred}})
$$

并在每个 block 中与当前状态交互。

 ### 14. 必须增加的测试

  至少验证下面四项：

  #### 1. source 能否影响输出
  v1 = model(pos=x_t, pos_source=source_1, ...)
  v2 = model(pos=x_t, pos_source=source_2, ...)
  assert not torch.allclose(v1, v2)

  #### 2. 平移不变
  v_shift = model(
      pos=x_t + shift,
      pos_source=source + shift,
      ...
  )
  assert torch.allclose(v, v_shift, atol=...)

  #### 3. 旋转等变
  v_rot = model(
      pos=x_t @ R.T,
      pos_source=source @ R.T,
      ...
  )
  assert torch.allclose(v_rot, v @ R.T, atol=...)

  #### 4. 输出零质心
  assert center(v, batch).abs().max() < tolerance

  另外应单独验证 identity pair：

  [
  x_{\mathrm{pred}}=x_{\mathrm{native}}
  ]

  训练后输出速度应接近零。

  总结来说，当前代码最合理的修改不是“再建一个完整网络”，而是：

  [
  \boxed{
  v_\theta\big(
  x_t,,
  x_{\mathrm{pred}},,
  x_t-x_{\mathrm{pred}},,
  d_t,,
  d_{\mathrm{pred}},,
  t
  \big)
  }
  ]

  其中 source distance 进入每层 edge attention，source node context 进入每层 scalar channel，(\Delta x_t) 进入 vector
  channel。这样既保留当前 TorchMD-ET，又实现了真正的 prediction-conditioned refinement。

## 静态radius graph不适合中尺度 refinement Modify_2 DONE->PT
图由初始预测坐标构建并在 50步 中固定：
  - native 中应形成的新接触不在图中；
  - 错误初始接触会长期占据图；
  - 只有 6 层局部 message passing，长 RNA 的 helix/domain 之间难以通信。

建议拆为：
  - 永久 covalent graph；
  - sequence-neighbor graph；
  - 动态 radius graph；
  - residue-level base-pair/stacking/global graph。-->放到第二阶段 代码改动太大了

  ### 一、四类图分别是什么意思

  #### 1. 永久 covalent graph->生成.pt文件时生成

  表示真实化学键，在整个 flow/ODE 过程中永远存在，不能因为原子距离变远就删除。

  包括：

  - 同一核苷酸内部的共价键；
  - 相邻核苷酸之间的磷酸二酯键：
    O3'(i) — P(i+1)。

  例如：

  P—O5'—C5'—C4'—C3'—O3'—P(next)

  生成时需要：

  - residue_index
  - atom_name
  - 核苷酸类型 A/C/G/U
  - 标准 RNA 原子键模板


  ##### 1. 标准 RNA 原子键模板如何获得

  不要手工维护 A/C/G/U 的键表，推荐直接使用 wwPDB Chemical Component Dictionary（CCD）。

  CCD 的 _chem_comp_bond 表提供：

  - atom_id_1
  - atom_id_2
  - value_order
  - 芳香键、立体化学等信息

  这是标准化学组分的权威定义。wwPDB CCD、chem_comp_bond 定义

  你的 Protenix 项目已经提供了 CCD 下载和读取代码：

  - Protenix/Protenix/scripts/gen_ccd_cache.py:32：从 wwPDB 下载 components.cif.gz
  - Protenix/Protenix/protenix/data/core/ccd.py:75：get_component_atom_array() 读取指定组分及其 bonds

  以后生成 .pt 时可以使用：

  from protenix.data.core import ccd


  def get_rna_bond_template(residue_name):
      component = ccd.get_component_atom_array(
          ccd_code=residue_name,
          keep_leaving_atoms=False,
          keep_hydrogens=False,
      )

      if component is None or component.bonds is None:
          raise ValueError(
              f"Cannot obtain CCD bonds for {residue_name}"
          )

      atom_names = component.atom_name.tolist()
      bond_array = component.bonds.as_array()
      # bond_array 每行为 [atom_index_1, atom_index_2, bond_type]

      bond_template = []

      for atom_index_1, atom_index_2, bond_type in bond_array:
          bond_template.append(
              (
                  atom_names[atom_index_1],
                  atom_names[atom_index_2],
                  int(bond_type),
              )
          )

      return bond_template

  标准残基直接读取：

  rna_bond_templates = {
      residue_name: get_rna_bond_template(residue_name)
      for residue_name in ["A", "C", "G", "U"]
  }

  注意两点：

  - CCD 主要提供单个核苷酸内部的键。
  - 相邻残基之间的 O3'(i)—P(i+1) 磷酸二酯键仍需根据 residue_index 显式加入。

    安装时要让 torch_cluster 与服务器的 PyTorch/CUDA 版本严格匹配，不建议直接随意使用一个不匹配的 wheel。

  建议统一原子名称：

  atom_name = atom_name.replace("*", "'")

  避免旧结构中的 C4* 和 CCD 中的 C4' 对不上。

  所有边最好是双向的：

  i -> j
  j -> i

  这部分应当在生成 .pt 文件时完成，而不是在模型中根据距离猜测。

  ———

  #### 2. sequence-neighbor graph->生成.pt文件时生成

  它表示“序列上相邻”，并不等同于化学键。

  虽然相邻残基已经通过 O3'—P 相连，但如果只靠原子化学键传播，一个残基的信息传到下一个残基的碱基部分需要经过很多层。

  因此可以给相邻残基的代表原子增加 shortcut：

  C4'(i) ↔ C4'(i+1)
  C4'(i) ↔ C4'(i+2)   # 可选

  不建议把相邻两个残基的全部原子两两相连，边数会非常大。

  第一版建议：

  sequence_window = 1
  anchor_atom = "C4'"

  即只连接相邻残基的 C4' 原子。

  ———

  #### 3. 动态 radius graph->每一步实时生成

  表示当前 flow 状态 x_t 中的空间近邻：

  distance(x_t[i], x_t[j]) < radius

  它和目前图的关键区别是：

  - 现在：由 pos_pred 生成一次，50 个 ODE step 都不变。
  - 修改后：由当前的 x_t 在 forward() 中重新生成。

  这样：

  - 两个原子逐渐靠近后，可以产生新边；
  - 原来错误靠近、后来分开的原子，边可以消失；
  - 图能够随 refinement 轨迹变化。

  训练时，每次随机得到新的 x_t 后构图。

  推理时，每次：

  v_t = model(pos=x, ...)

  都会基于更新后的 x 构图。因此 sample() 本身不需要额外改循环。

  ———

  #### 4. residue-level base-pair/stacking/global graph->第二阶段目标

  这里节点不再是原子，而是核苷酸残基。

  #### Base-pair 边

  表示可能形成碱基配对的两个残基，例如：

  A-U
  G-C
  G-U
  非 canonical pairing

  候选边可来自：

  - Protenix 的 contact probability/PAE；
  - 从 pos_pred 计算的宽松空间候选；
  - RNA 二级结构预测；
  - 模型自己预测的 interaction logits。

  注意：不能训练时根据 native 结构构图、推理时却没有 native，这会造成标签泄漏。

  #### Stacking 边

  表示碱基堆积。通常根据：

  - 碱基中心距离；
  - 碱基平面法向量夹角；
  - 两个碱基的相对位姿

  生成候选边。

  #### Global 边

  用于长距离通信。例如连接：

  i ↔ i+4
  i ↔ i+8
  i ↔ i+16
  i ↔ i+32

  这种 dilated sequence graph 能让长 RNA 的远距离残基在较少层数内通信。

  ———

  ### 二、建议的新 .pt 数据格式

  你当前的数据生成代码只有原子坐标和 radius graph，无法可靠生成 sequence/residue graph。新版本至少需要保存：

  {
      "pos": pos,
      "pos_pred": pos_pred,
      "atomic_numbers": atomic_numbers,

      # 原子所属残基，范围 0 ... num_residues-1
      "residue_index": residue_index,

      # 建议保存整数编码，PyG batching 更方便
      "atom_name_id": atom_name_id,

      # 静态图：化学边 + sequence shortcut + 静态 interaction candidates
      "edge_index": edge_index,

      # 每条静态边的类型
      "edge_attr": edge_attr,

      "node_attr": node_attr,
      "sequence": sequence,
  }

  边类型建议用 one-hot，而不是用一个整数直接输入网络：

   下标    边类型
  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━
      0    核苷酸内部共价键
  ──────  ─────────────────────
      1    磷酸二酯键
  ──────  ─────────────────────
      2    sequence-neighbor
  ──────  ─────────────────────
      3    dynamic spatial
  ──────  ─────────────────────
      4    base-pair candidate
  ──────  ─────────────────────
      5    stacking candidate
  ──────  ─────────────────────
      6    global/dilated
    
你当前不使用后三类边也没关系，静态 .pt 中第 4～6 列保持为零即可。提前保留这三列可以避免以后增加边类型时改变模型输入维度。

  因此：

  edge_attr.shape == [num_edges, 7]

  如果用单个整数 0～6 直接作为连续特征，网络会错误地认为类型 6 在数值上“大于”类型 1，所以应该使用 one-hot 或 embedding。
#### 6. 当前阶段的 .pt 数据要求

  按你目前“不加入 base-pair/stacking/global 图”的计划，建议最终生成：

  {
      # 必需坐标，二者原子顺序必须完全一致
      "pos": FloatTensor[N, 3],
      "pos_pred": FloatTensor[N, 3],

      "atomic_numbers": LongTensor[N],
      "sequence": str,

      # 建议从 0 开始、连续编号
      "residue_index": LongTensor[N],

      # 必须使用整个数据集统一的 atom-name 编码表
      "atom_name_id": LongTensor[N],

      # 只保存永久静态边
      "edge_index": LongTensor[2, E_static],

      # 7 维 one-hot；目前只允许第 0、1、2 列为 1
      "edge_attr": FloatTensor[E_static, 7],

      "node_attr": FloatTensor[N, node_attr_dim],
  }

  静态边要求：

  - 双向；
  - 无 self-loop；
  - 无重复边；
  - 索引范围在 [0, N)；
  - edge_attr.shape[0] == edge_index.shape[1]。

  目前静态 one-hot 分别为：

   核苷酸内部共价键
  [1, 0, 0, 0, 0, 0, 0]

   相邻残基 O3'—P 磷酸二酯键
  [0, 1, 0, 0, 0, 0, 0]

  相邻残基 C4'—C4' sequence edge
  [0, 0, 1, 0, 0, 0, 0]

  .pt 中不要保存 radius graph。第 3 列 dynamic spatial edge 由模型在运行时生成。

  atom_name_id 必须采用全数据集固定映射，例如：

  ATOM_NAME_TO_ID = {
      "P": 0,
      "OP1": 1,
      "OP2": 2,
      "O5'": 3,
      "C5'": 4,
      "C4'": 5,
      # ...
  }

  不能每个样本单独按照出现顺序编号，否则同一个 ID 在不同 RNA 中会代表不同原子。

  当前 EuclideanDataset 还没有把 residue_index 和 atom_name_id 放入返回的 Data。这不影响当前版本，因为静态边已经在 .pt 中构建好，模型尚
  未直接使用这两个字段；以后加入 atom-role embedding 或 residue-level network 时，再同步增加即可。

  ### 五、base-pair/stacking 图怎样接入当前模型

  第一版不用立即写真正的 residue GNN，可以使用“代表原子代理”：

  残基 i 的代表节点 = C4'(i) 或 C1'(i)

  例如预测残基 5 和残基 38 可能配对，则加入：

  C1'(5) ↔ C1'(38)
  edge type = base-pair

  预测发生 stacking，则加入：

  C1'(i) ↔ C1'(j)
  edge type = stacking

  这样当前 TorchMD 完全不需要新的 residue block。

  但它只是近似。真正的 residue-level 实现需要：

  atomic scalar/vector features
          ↓ scatter/mean by residue_index
  residue scalar/vector features
          ↓ residue graph message passing
  broadcast back to atoms
          ↓
  atomic TorchMD blocks/output

  这会涉及：

  - PyG batch 中残基索引的全局偏移；
  - atom-to-residue pooling；
  - residue-level equivariant block；
  - residue-to-atom broadcasting；
  - interaction prediction head。

  因此不建议和动态 radius graph 一次性一起改。

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
