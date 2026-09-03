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

## 网络显式看到原始预测结构 Modify_1  DONE

 ### 必须增加的测试

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

  ##### Base-pair 边

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

  ##### Stacking 边

  表示碱基堆积。通常根据：

  - 碱基中心距离；
  - 碱基平面法向量夹角；
  - 两个碱基的相对位姿

  生成候选边。

  ##### Global 边

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

  ##### 五、base-pair/stacking 图怎样接入当前模型

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

## 自由Cartesian原子流没有化学约束 & stochastic path未必适合当前确定性任务 Modify_3
(\sigma_t=\sigma\sqrt{t(1-t)}) 的导数在两端趋于无穷。当前采样到 1e-4/0.9999 附近时，noise velocity 可明显大于真实 correction。

建议首先消融：
  - sigma=0 的 deterministic paired rectified flow；
  - 当前 sigma=0.1；
  - 有界导数的 (\sigma t(1-t))。

  如果每个 pos_pred 只有一个 native，必须加入“一步 residual EGNN”基线，否则 reviewer 很容易质疑为什么需要 50-step flow，
  而不是直接预测 pos-pos_pred。

  ### 建议加入的四种几何 loss

  #### 共价键长度 loss

  只使用以下两类边：

  edge_attr[:, 0]：核苷酸内部共价键
  edge_attr[:, 1]：磷酸二酯键

  不能把 sequence edge 和 dynamic radius edge 当成化学键。

  设理想键长为 (l_{ij})：

  [
  L_{\mathrm{bond}}

  \frac1{|E_b|}
  \sum_{(i,j)\in E_b}
  \left(
  |\hat x_i-\hat x_j|-l_{ij}
  \right)^2
  ]

  数据中最好额外保存：

  bond_index: LongTensor[2, E_bond]
  bond_length: FloatTensor[E_bond]

  其中理想键长可以来自 CCD，也可以暂时使用 native 中对应键的长度。长期看推荐 CCD 理想长度。

  磷酸二酯键断裂也会被这个 loss 约束，因此不一定需要单独写 chain-break loss。

  i → j，目标长度 1.43
  j → i，目标长度 1.43

  如果保存双向边，loss 会把同一根键计算两次。虽然数值均值通常不受影响，但更干净的做法是几何 loss 只保留一个方向，例如 i < j。

  #### 键角 loss

  对于三原子：

  i — j — k

  约束夹角：

  [
  L_{\mathrm{angle}}

  \frac1{N_a}
  \sum
  \left(
  \cos\hat\theta_{ijk}-\cos\theta^{*}_{ijk}
  \right)^2
  ]

  推荐比较 cosine，不要直接使用 acos()，因为接近 (0) 或 (\pi) 时数值不稳定。

  数据中保存：

  angle_index: LongTensor[3, E_angle]
  angle_target_cos: FloatTensor[E_angle]

  第一版可以从 native 结构计算 target；以后再换成标准模板值。

  ####Steric clash loss

  对非键合原子对计算：

  [
  L_{\mathrm{clash}}

  \operatorname{mean}
  \left[
  \operatorname{ReLU}
  \left(
  r_i+r_j-\delta-d_{ij}
  \right)^2
  \right]
  ]

  其中：

  - (r_i,r_j) 是原子的 van der Waals 半径；
  - (\delta) 是适当容差；
  - 排除共价相连的 1–2 原子；
  - 最好也排除共享一个中心原子的 1–3 原子。

  可以利用当前 dynamic radius graph，只对较近的动态边计算，不需要计算完整 (N^2) 原子对。

  #### 碱基平面 loss

  A/C/G/U 的碱基重原子应大致处于同一平面。

  可对每个残基的碱基原子计算中心：

  base_center = base_pos.mean(dim=0)

  然后对中心化坐标的协方差矩阵求最小特征值：

  [
  L_{\mathrm{plane}}

  \frac1R
  \sum_r
  \lambda_{\min}
  \left(
  X_r^\top X_r
  \right)
  ]

  最小特征值越小，说明这些原子越接近一个平面。

  这项 loss 不要求碱基保持某个固定的空间朝向，所以不会破坏旋转等变性。

  ### 5. 总 loss

  在 Code_Flow_matching/etflow/models/model.py 的 generic_step() 中，目前是：

  loss = batchwise_l2_loss(
      v_t,
      u_t,
      batch=batch,
      reduce="mean",
  )

  以后可以改成：

  flow_loss = batchwise_l2_loss(
      v_t,
      u_t,
      batch=batch,
      reduce="mean",
  )

  t_atom = unsqueeze_like(t[batch], x_t)
  pos_estimate = x_t + (1 - t_atom) * v_t

  bond_loss = compute_bond_loss(...)
  angle_loss = compute_angle_loss(...)
  clash_loss = compute_clash_loss(...)
  plane_loss = compute_base_plane_loss(...)

  loss = (
      flow_loss
      + lambda_bond * bond_loss
      + lambda_angle * angle_loss
      + lambda_clash * clash_loss
      + lambda_plane * plane_loss
  )

  建议第一轮只加入：

  bond loss
  clash loss
  base-plane loss

  键角、糖环 pucker、手性和 torsion 后续再加，避免一次引入太多变量。

  各辅助 loss 最好先单独归一化，并使它们在训练初期的梯度量级明显小于主要的 flow loss。


  ### 7. 如果还想保证 ODE 中间步骤也合理

  终点几何 loss 主要保证最终结构，不保证所有中间 x_t 都严格合法。可以进一步采用两种方法。

  #### 轻量方案：每步或最终进行约束优化

  在 ODE 输出后，对以下能量做少量梯度优化：

  bond energy
  angle energy
  clash energy
  plane energy
  与模型输出之间的 restraint

  最后一项非常重要，否则能量最小化可能把结构拉离模型预测结果。

  也可以使用 RNA force field 做 restrained minimization。

  #### 最严格方案：改成内部坐标 flow

  不直接预测每个原子的自由 Cartesian 位移，而是预测：

  - backbone torsion；
  - sugar pucker；
  - nucleotide rigid frame；
  - 碱基相对位姿。

  这样键长和大部分键角可以天然固定。但这相当于更换坐标表示和输出头，改动远大于增加辅助 loss，适合作为后续版本。

  
  #### Protenix CCD 是否提供理想键长

  分两层看：

  1. CCD 的 chem_comp_bond 数据模式支持 value_dist 和 value_dist_esd。
  2. Protenix 的：

  ccd.get_component_atom_array(...)

  主要返回 AtomArray、理想坐标和 BondList。BondList 保留连接关系和 bond type，但不一定直接暴露 value_dist。

  最稳妥的方法是利用 Protenix 返回的 CCD 理想坐标计算：

  component = ccd.get_component_atom_array(
      residue_name,
      keep_leaving_atoms=False,
      keep_hydrogens=False,
  )

  bond_array = component.bonds.as_array()
  ideal_pos = component.coord

  for atom_i, atom_j, bond_type in bond_array:
      ideal_length = np.linalg.norm(
          ideal_pos[atom_i] - ideal_pos[atom_j]
      )

  因为 Protenix/Protenix/protenix/data/core/ccd.py:94 使用：

  use_ideal_coord=True

  所以 component.coord 是 CCD 理想坐标。

  对于跨残基的：

  O3'(i)—P(i+1)

  它不属于单个组分内部的 component.bonds。可以使用：

  - RNA force-field 中的理想值；
  - 化学连接字典；
  - 或先统计训练集 native 结构中该键长度的稳健中位数。

  不建议每个样本直接把自己的 native 键长作为输入；可以将训练集统计得到的全局模板作为 target。

  ### 6. Steric clash loss 具体怎么算

  #### 输入

  需要：

  pos_estimate       # [N,3] 模型估计的最终结构
  atomic_numbers     # [N]
  batch              # [N]
  bond_index         # 排除直接成键的原子对
  angle_index        # 可选，用于排除1-3原子对

  首先为元素准备 van der Waals 半径。可以使用一致的力场参数集，例如大致包括：

  vdw_radius = {
      6: 1.70,   # C
      7: 1.55,   # N
      8: 1.52,   # O
      15: 1.80,  # P
      16: 1.80,  # S
  }

  实际实验中应固定一个来源，不要在不同数据中混用多套半径。

  #### 第一步：找较近的原子对

  应基于 pos_estimate 单独构建 clash candidate graph：

  clash_index = radius_graph(
      x=pos_estimate,
      r=4.5,
      batch=batch,
      loop=False,
      max_num_neighbors=64,
  )

  不能直接依赖当前 x_t 的动态图，因为 x_t 没有 clash，不代表预测终点 pos_estimate 没有 clash。

  #### 第二步：删除不应计算 clash 的原子对

  排除：

  - 自己和自己；
  - 同一共价键的 1–2 原子；
  - 最好排除通过一个中心原子连接的 1–3 原子；
  - 使用 i < j 避免重复计数。

  #### 第三步：计算允许距离

  对候选原子对 (i,j)：

  [
  d_{ij}=|\hat x_i-\hat x_j|
  ]

  定义允许的最小距离：

  [
  d^{\min}_{ij}=s(r_i+r_j)
  ]

  其中 (s) 是容差系数，可以从较宽松的值开始，例如 0.8。

  #### 第四步：计算穿透量

  [
  o_{ij}=\max(0,d^{\min}{ij}-d{ij})
  ]

  distance = torch.linalg.vector_norm(
      pos_estimate[row] - pos_estimate[col],
      dim=-1,
  )

  minimum_distance = 0.8 * (
      radius[row] + radius[col]
  )

  overlap = torch.relu(
      minimum_distance - distance
  )

  clash_loss = overlap.square().mean()

  含义：

  - 距离足够大：loss 为0；
  - 两个原子发生重叠：距离越小，惩罚越大。

  注意 radius_graph() 只负责选出候选边，随后必须使用 pos_estimate 重新计算 distance，这样 clash loss 才能对模型输出反向传播。

  ### 7. 碱基平面 loss 具体怎么算

  #### 第一步：选出每个残基的碱基环原子

  糖和磷酸原子不能参与碱基平面拟合。

  嘌呤 A/G 可选环原子：

  N9 C8 N7 C5 C6 N1 C2 N3 C4

  嘧啶 C/U 可选：

  N1 C2 N3 C4 C5 C6

  通过：

  residue_index
  atom_name_id

  得到每个残基的 base_atom_mask。

  #### 第二步：提取一个残基的碱基坐标

  base_pos = pos_estimate[base_atom_mask]
  base_center = base_pos.mean(dim=0)
  centered = base_pos - base_center

  #### 第三步：计算协方差矩阵

  covariance = (
      centered.transpose(0, 1) @ centered
      / centered.size(0)
  )

  形状是：

  [3, 3]

  #### 第四步：计算最小特征值

  eigenvalues = torch.linalg.eigvalsh(covariance)
  residue_plane_loss = eigenvalues[0]

  原理是：

  - 完全位于二维平面时，垂直平面方向的方差为0；
  - 碱基翘曲越严重，最小特征值越大。

  全部残基取平均：

  plane_losses = []

  for residue_id in residue_ids:
      atom_mask = (
          (residue_index == residue_id)
          & base_atom_mask
      )

      if atom_mask.sum() < 3:
          continue

      base_pos = pos_estimate[atom_mask]
      base_pos = base_pos - base_pos.mean(
          dim=0,
          keepdim=True,
      )

      covariance = (
          base_pos.transpose(0, 1) @ base_pos
          / base_pos.size(0)
      )

      eigenvalues = torch.linalg.eigvalsh(
          covariance
      )
      plane_losses.append(eigenvalues[0])

  plane_loss = torch.stack(plane_losses).mean()

  对于 batch size 大于1，不能只用 residue_index，因为不同 RNA 都可能有 residue 0。应使用：

  (batch, residue_index)

  共同确定唯一残基。

  这个 loss：

  - 平移不变；
  - 旋转不变；
  - 可微；
  - 不强迫碱基朝向某个固定方向；
  - 只约束碱基不要变成非平面。

  ### 8. 是否意味着直接预测 pos-pos_pred，不再用ODE

  不是自动意味着，而是现在有两条都应该测试的路线。

  #### 路线A：一步 residual baseline

  训练：

  x = x0_centered
  target = x1_centered - x0_centered

  delta = model(
      pos=x,
      pos_source=x,
      t=zeros,
      ...
  )

  loss = MSE(delta, target)

  推理：

  pos_refined = x0_centered + delta

  不需要ODE，也不需要时间。

  #### 路线B：deterministic flow

  训练随机采样 (t)：

  x_t = (1 - t) * x0_centered + t * x1_centered
  u_t = x1_centered - x0_centered

  推理仍然进行ODE：

  x = x0_centered

  for i in range(n_timesteps):
      v_t = model(x, t_i)
      x = x + delta_t * v_t

  但步数未必需要50，可以实验1、5、10、20、50步。

  ### #我的建议

  不要现在就把当前 flow 改成只有一步。应该保留两个独立实验：

  实验1：专门训练的一步 residual model
  实验2：sigma=0 的 deterministic flow model

  如果实验1不弱于实验2，就采用一步模型，速度更快、逻辑更简单。

  如果多步 deterministic flow 明显更好，就说明模型确实在利用“根据中间结构反复纠错”的能力，此时可以合理保留5～20步，而未必需要50步。


 你的方案可以整理成两个正交开关：

  training_objective:
      flow       多步 deterministic/stochastic flow
      residual   一步直接预测 pos-pos_pred

  flow_path:
      deterministic
      stochastic

  这样可以公平比较：

  1. 一步 residual regression；
  2. deterministic flow 的 1/5/10/20/50 步；
  3. stochastic flow 的不同步数。

  ## 一、回答第3个问题

  “把 (t) 固定为0”指的是：为一步 residual 模型单独改变训练数据构造，而不是只在推理时把一个随机 (t) 训练的 flow 模型强行设置成 (t=0)。

  ### 当前 flow 训练

  当前模型训练时：

  t = self.sample_time(...)
  x_t = (1 - t) * x0 + t * x1
  target = x1 - x0

  所以模型见过的是整个 (t\in(0,1)) 的中间结构分布。

  ### 一步 residual 训练

  一步 residual 必须训练成：

  t = 0
  x_t = x0
  target = x1 - x0

  也就是整个训练过程只学习：

  [
  v_\theta(x_0,0)=x_1-x_0
  ]

  推理才是：

  x1_estimate = x0 + model(x0, t=0)

  因此下面两者不完全相同：

  A. 随机 t 训练的 flow 模型，推理时只调用 t=0 一次
  B. 始终在 t=0 专门训练的一步 residual 模型

  B 才是公平的一步 residual baseline。

  ———

  # 二、增加两个模式参数

  在 BaseFlow.__init__() 增加：

  training_objective: str = "flow",
  flow_path: str = "deterministic",

  保留：

  sigma: float = 0.1,

  其中：

  training_objective="flow"
  flow_path="deterministic"

  表示确定性 flow。

  training_objective="flow"
  flow_path="stochastic"
  sigma=0.1

  表示带噪 flow。

  training_objective="residual"

  表示一步 residual；此时 flow_path 和 sigma 不参与训练路径。

  初始化时增加：

  if training_objective not in {"flow", "residual"}:
      raise ValueError(
          f"Unknown training_objective: {training_objective}"
      )

  if flow_path not in {"deterministic", "stochastic"}:
      raise ValueError(
          f"Unknown flow_path: {flow_path}"
      )

  if flow_path == "stochastic" and sigma <= 0:
      raise ValueError(
          "stochastic flow requires sigma > 0"
      )

  self.training_objective = training_objective
  self.flow_path = flow_path

  ## 修改噪声函数

  由：

  def sigma_t(self, t):
      return self.sigma * torch.sqrt(t * (1 - t))

  def sigma_dot_t(self, t):
      return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

  改为：

  def sigma_t(self, t):
      if self.flow_path == "deterministic":
          return torch.zeros_like(t)

      return self.sigma * torch.sqrt(t * (1 - t))


  def sigma_dot_t(self, t):
      if self.flow_path == "deterministic":
          return torch.zeros_like(t)

      return (
          self.sigma
          * 0.5
          * (1 - 2 * t)
          / torch.sqrt(t * (1 - t))
      )

  这样切换：

  flow_path: deterministic

  就完全关闭噪声，而不用删除 stochastic 代码。

  ## 修改噪声采样

  在 sample_conditional_pt() 中，由：

  eps = torch.randn_like(x1)
  eps = center_of_mass(eps, batch=batch)

  改为：

  if self.flow_path == "deterministic":
      eps = torch.zeros_like(x1)
  else:
      eps = torch.randn_like(x1)
      eps = center_of_mass(eps, batch=batch)

  并让 compute_conditional_vector_field() 返回 eps：

  return x_t, u_t, eps

  这是为了将来 stochastic 模式下也能构造无噪声的预测终点。

  ———

  # 三、修改训练数据构造

  在 generic_step() 中，原来的：

  t = self.sample_time(
      num_samples=batch_size,
      stage=stage,
  )

  x_t, u_t = self.compute_conditional_vector_field(
      x0=x0,
      x1=pos,
      t=t,
      batch=batch,
  )

  改为：

  x0_centered = center_of_mass(x0, batch=batch)
  x1_centered = center_of_mass(pos, batch=batch)

  if self.training_objective == "residual":
      t = torch.zeros(
          batch_size,
          1,
          dtype=x0.dtype,
          device=x0.device,
      )

      x_t = x0_centered
      u_t = x1_centered - x0_centered
      eps = torch.zeros_like(x_t)

  else:
      t = self.sample_time(
          num_samples=batch_size,
          stage=stage,
      )

      x_t, u_t, eps = (
          self.compute_conditional_vector_field(
              x0=x0,
              x1=pos,
              t=t,
              batch=batch,
          )
      )

  网络调用保持：

  v_t = self(
      z=z,
      t=t,
      pos=x_t,
      pos_source=x0,
      bond_index=bond_index,
      edge_attr=edge_attr,
      node_attr=node_attr,
      batch=batch,
  )

  ———

  # 四、构造用于几何 loss 的预测终点

  网络输出后增加：

  t_atom = unsqueeze_like(t[batch], x_t)

  if self.training_objective == "residual":
      pos_estimate = x0_centered + v_t

  elif self.flow_path == "deterministic":
      pos_estimate = x_t + (1 - t_atom) * v_t

  else:
      clean_displacement = (
          v_t
          - self.sigma_dot_t(t_atom) * eps
      )
      pos_estimate = x0_centered + clean_displacement

  含义：

  - residual：直接预测完整位移；
  - deterministic flow：从当前 (x_t) 估计终点；
  - stochastic flow：先从速度中减去已知的噪声速度。

  几何 loss 都计算在 pos_estimate 上。

  ———

  # 五、键长 loss

  为了不与现有作为完整静态图的 bond_index 混淆，建议 .pt 使用：

  geometry_bond_index: LongTensor[2, E_bond]
  ideal_bond_length: FloatTensor[E_bond]

  其中只包含：

  - 核苷酸内部共价键；
  - 相邻核苷酸的磷酸二酯键。

  最好每根键只保存一次，不保存双向副本。

  在 Code_Flow_matching/etflow/models/loss.py 增加：

  def bond_length_loss(
      prediction: torch.Tensor,
      geometry_bond_index: torch.Tensor,
      ideal_bond_length: torch.Tensor,
  ) -> torch.Tensor:
      if geometry_bond_index.numel() == 0:
          return prediction.sum() * 0.0

      atom_i = geometry_bond_index[0]
      atom_j = geometry_bond_index[1]

      predicted_bond_length = torch.linalg.vector_norm(
          prediction[atom_i] - prediction[atom_j],
          dim=-1,
      )

      ideal_bond_length = ideal_bond_length.to(
          dtype=prediction.dtype,
          device=prediction.device,
      ).view(-1)

      return (
          predicted_bond_length
          - ideal_bond_length
      ).square().mean()

  在 Dataset 中读取：

  geometry_bond_index = data["geometry_bond_index"]
  ideal_bond_length = data["ideal_bond_length"]

  并放进 Data(...)：

  geometry_bond_index=geometry_bond_index,
  ideal_bond_length=ideal_bond_length,

  由于字段名包含 index，PyG 通常会在 batching 时自动增加原子索引偏移。

  ———

  # 六、Steric clash loss

  ## 采用哪套 van der Waals 半径

  当前第一版建议使用 Bondi 半径，只处理 RNA 常见的 C/N/O/P/S：

  BONDI_VDW_RADII = {
      6: 1.70,   # C
      7: 1.55,   # N
      8: 1.52,   # O
      15: 1.80,  # P
      16: 1.80,  # S
  }

  暂时不对 Mg、K 等离子计算这个 clash loss。离子相互作用不能简单按照普通有机原子的 steric clash 处理。

  在 BaseFlow.__init__() 中创建 buffer：

  vdw_radius_table = torch.zeros(max_z)

  for atomic_number, radius in {
      6: 1.70,
      7: 1.55,
      8: 1.52,
      15: 1.80,
      16: 1.80,
  }.items():
      if atomic_number < max_z:
          vdw_radius_table[atomic_number] = radius

  self.register_buffer(
      "vdw_radius_table",
      vdw_radius_table,
  )

  变量名使用 vdw_radius_table，不会和当前 edge_weight、cutoff 等变量混淆。

  ## clash loss 代码

  在 loss.py 中增加：

  from etflow.models.utils import build_dynamic_radius_graph


  def steric_clash_loss(
      prediction: torch.Tensor,
      atomic_numbers: torch.Tensor,
      batch: torch.Tensor,
      geometry_bond_index: torch.Tensor,
      vdw_radius_table: torch.Tensor,
      clash_cutoff: float = 4.5,
      clash_distance_scale: float = 0.8,
      clash_max_num_neighbors: int = 64,
  ) -> torch.Tensor:
      clash_edge_index = build_dynamic_radius_graph(
          pos=prediction,
          batch=batch,
          cutoff=clash_cutoff,
          max_num_neighbors=clash_max_num_neighbors,
      )

      if clash_edge_index.numel() == 0:
          return prediction.sum() * 0.0

      atom_i = clash_edge_index[0]
      atom_j = clash_edge_index[1]

      # 双向边只保留一次。
      unique_pair_mask = atom_i < atom_j
      atom_i = atom_i[unique_pair_mask]
      atom_j = atom_j[unique_pair_mask]

      num_nodes = prediction.size(0)

      candidate_pair_id = atom_i * num_nodes + atom_j

      # geometry_bond_index 可能方向不固定，先转换为无向pair。
      bond_atom_i = torch.minimum(
          geometry_bond_index[0],
          geometry_bond_index[1],
      )
      bond_atom_j = torch.maximum(
          geometry_bond_index[0],
          geometry_bond_index[1],
      )
      bonded_pair_id = (
          bond_atom_i * num_nodes
          + bond_atom_j
      )

      nonbonded_mask = ~torch.isin(
          candidate_pair_id,
          bonded_pair_id,
      )

      atom_i = atom_i[nonbonded_mask]
      atom_j = atom_j[nonbonded_mask]

      if atom_i.numel() == 0:
          return prediction.sum() * 0.0

      atom_radius = vdw_radius_table[
          atomic_numbers.view(-1)
      ]

      supported_atom_mask = (
          (atom_radius[atom_i] > 0)
          & (atom_radius[atom_j] > 0)
      )

      atom_i = atom_i[supported_atom_mask]
      atom_j = atom_j[supported_atom_mask]

      if atom_i.numel() == 0:
          return prediction.sum() * 0.0

      predicted_distance = torch.linalg.vector_norm(
          prediction[atom_i] - prediction[atom_j],
          dim=-1,
      )

      minimum_distance = clash_distance_scale * (
          atom_radius[atom_i]
          + atom_radius[atom_j]
      )

      overlap = torch.relu(
          minimum_distance - predicted_distance
      )

      return overlap.square().mean()

  更严格的版本还需要排除 1–3 原子对。建议生成 .pt 时额外保存：

  clash_exclusion_index

  它包含：

  - 1–2 共价原子对；
  - 1–3 键角原子对。

  第一版只排除直接成键原子可能把部分正常的 1–3 接触误判成 clash，所以正式训练前最好补上该字段。

  ———

  # 七、碱基平面 loss

  ## 固定 atom-name 编码

  atom_name_id 必须使用全数据集统一映射。

  碱基环原子名称集合：

  BASE_RING_ATOM_NAMES = {
      "N9", "C8", "N7", "C5", "C6",
      "N1", "C2", "N3", "C4",
  }

  根据你的 ATOM_NAME_TO_ID 得到：

  base_atom_name_ids = torch.tensor(
      [
          ATOM_NAME_TO_ID[name]
          for name in BASE_RING_ATOM_NAMES
      ],
      dtype=torch.long,
  )

  可以像 vdW 半径一样注册为 buffer：

  self.register_buffer(
      "base_atom_name_ids",
      base_atom_name_ids,
  )

  ## plane loss 代码

  在 loss.py 增加：

  def base_plane_loss(
      prediction: torch.Tensor,
      residue_index: torch.Tensor,
      atom_name_id: torch.Tensor,
      batch: torch.Tensor,
      base_atom_name_ids: torch.Tensor,
  ) -> torch.Tensor:
      if batch is None:
          batch = torch.zeros(
              prediction.size(0),
              dtype=torch.long,
              device=prediction.device,
          )

      base_atom_mask = torch.isin(
          atom_name_id.view(-1),
          base_atom_name_ids,
      )

      # 不同RNA都可能存在residue_index=0，
      # 因此必须结合batch区分残基。
      residue_key = torch.stack(
          [
              batch.view(-1),
              residue_index.view(-1),
          ],
          dim=1,
      )

      _, residue_group = torch.unique(
          residue_key,
          dim=0,
          return_inverse=True,
      )

      plane_loss_list = []

      for group_index in torch.unique(residue_group):
          atom_mask = (
              (residue_group == group_index)
              & base_atom_mask
          )

          if atom_mask.sum() < 3:
              continue

          base_position = prediction[atom_mask]
          base_position = (
              base_position
              - base_position.mean(
                  dim=0,
                  keepdim=True,
              )
          )

          covariance = (
              base_position.transpose(0, 1)
              @ base_position
              / base_position.size(0)
          )

          eigenvalues = torch.linalg.eigvalsh(
              covariance
          )

          plane_loss_list.append(
              eigenvalues[0].clamp_min(0)
          )

      if not plane_loss_list:
          return prediction.sum() * 0.0

      return torch.stack(plane_loss_list).mean()

  最小特征值就是碱基原子沿“最佳拟合平面法向方向”的平均平方偏离。

  ———

  # 八、在 generic_step() 中组合 loss

  导入：

  from etflow.models.loss import (
      base_plane_loss,
      batchwise_l2_loss,
      bond_length_loss,
      steric_clash_loss,
  )

  读取新数据：

  geometry_bond_index = batched_data[
      "geometry_bond_index"
  ]
  ideal_bond_length = batched_data[
      "ideal_bond_length"
  ]
  residue_index = batched_data["residue_index"]
  atom_name_id = batched_data["atom_name_id"]

  计算：

  flow_matching_loss = batchwise_l2_loss(
      v_t,
      u_t,
      batch=batch,
      reduce="mean",
  )

  bond_loss = bond_length_loss(
      prediction=pos_estimate,
      geometry_bond_index=geometry_bond_index,
      ideal_bond_length=ideal_bond_length,
  )

  clash_loss = steric_clash_loss(
      prediction=pos_estimate,
      atomic_numbers=z,
      batch=batch,
      geometry_bond_index=geometry_bond_index,
      vdw_radius_table=self.vdw_radius_table,
  )

  plane_loss = base_plane_loss(
      prediction=pos_estimate,
      residue_index=residue_index,
      atom_name_id=atom_name_id,
      batch=batch,
      base_atom_name_ids=self.base_atom_name_ids,
  )

  loss = (
      flow_matching_loss
      + self.bond_loss_weight * bond_loss
      + self.clash_loss_weight * clash_loss
      + self.plane_loss_weight * plane_loss
  )

  建议新增初始化参数：

  bond_loss_weight: float = 0.1,
  clash_loss_weight: float = 0.01,
  plane_loss_weight: float = 0.1,

  这只是初始实验值。正式选择时应观察每项原始 loss 和梯度量级，不能只比较数字大小。

  ———

  # 九、修改 sample() 支持一步 residual 和多步 flow

  首先正确计算 batch size：

  batch_size = (
      int(batch.max().item()) + 1
      if batch is not None
      else 1
  )

  ### residual 分支

  在 ODE 循环前增加：

  if self.training_objective == "residual":
      t = torch.zeros(
          batch_size,
          1,
          dtype=source.dtype,
          device=source.device,
      )

      residual = self(
          z=z,
          t=t,
          pos=source,
          pos_source=source,
          bond_index=bond_index,
          edge_attr=edge_attr,
          node_attr=node_attr,
          batch=batch,
      )

      return source + residual

  ### flow 分支

  保留当前 ODE，但建议把当前按原子构造 t：

  t = t_schedule[i].repeat(x.size(0))
  t = unsqueeze_like(t, x)

  改为按 batch 构造：

  t = torch.full(
      (batch_size, 1),
      fill_value=t_schedule[i].item(),
      dtype=x.dtype,
      device=x.device,
  )

  因为 BaseFlow.forward() 内部已经执行：

  t=t[batch]

  所以它期望 t 是每个 RNA 一个时间，而不是每个原子一个时间。

  之后通过：

  n_timesteps=1
  n_timesteps=5
  n_timesteps=10
  n_timesteps=20
  n_timesteps=50

  比较 deterministic flow 步数。

  ## 最终实验矩阵

  建议至少训练三个独立模型：

  模型A
  training_objective=residual
  一步推理

  模型B
  training_objective=flow
  flow_path=deterministic
  分别测试1/5/10/20/50步

  模型C
  training_objective=flow
  flow_path=stochastic
  sigma=0.1
  分别测试10/20/50步

  其中“模型B使用1步ODE”和“模型A一步residual”必须分开报告，因为它们架构调用形式相似，但训练分布不同。
 ## 3. clash_exclusion_index 的含义和格式

  推荐格式：

  clash_exclusion_index: LongTensor[2, E_exclusion]

  例如：

  tensor([
      [0, 1, 0],
      [1, 2, 2],
  ])

  表示排除：

  (0, 1)  1–2
  (1, 2)  1–2
  (0, 2)  1–3

  建议每个无向原子对只保存一次，并满足：

  clash_exclusion_index[0] < clash_exclusion_index[1]

  PyG 会自动对名称中包含 index 的二维索引进行 batch 偏移，因此这个名称可以正常批处理。

  ### 生成逻辑

  在以后生成 .pt 时可以使用：

  from itertools import combinations

  import torch


  def build_clash_exclusion_index(
          geometry_bond_index: torch.Tensor,
          num_atoms: int,
  ) -> torch.Tensor:
      neighbors = [set() for _ in range(num_atoms)]
      one_two_pairs = set()

      for atom_i, atom_j in (
          geometry_bond_index.detach().cpu().t().tolist()
      ):
          atom_i = int(atom_i)
          atom_j = int(atom_j)

          if atom_i == atom_j:
              continue

          pair = tuple(sorted((atom_i, atom_j)))
          one_two_pairs.add(pair)

          neighbors[atom_i].add(atom_j)
          neighbors[atom_j].add(atom_i)

      one_three_pairs = set()

      for center_atom in range(num_atoms):
          for atom_i, atom_j in combinations(
                  sorted(neighbors[center_atom]),
                  2,
          ):
              one_three_pairs.add(
                  tuple(sorted((atom_i, atom_j)))
              )

      exclusion_pairs = sorted(
          one_two_pairs | one_three_pairs
      )

      if not exclusion_pairs:
          return torch.empty((2, 0), dtype=torch.long)

      return torch.tensor(
          exclusion_pairs,
          dtype=torch.long,
      ).t().contiguous()

 ### 新建统一常量文件

  建议新建：

  etflow/data/constants.py

  内容可以是：

  UNKNOWN_ATOM_NAME_ID = 0

  RNA_ATOM_NAMES = (
      # 磷酸和糖环
      "P", "OP1", "OP2", "OP3",
      "O5'", "C5'", "C4'", "O4'",
      "C3'", "O3'", "C2'", "O2'", "C1'",

      # 碱基环
      "N9", "C8", "N7", "C5", "C6",
      "N1", "C2", "N3", "C4",

      # 碱基环外原子
      "N6", "O6", "N2", "O2", "N4", "O4",
  )

  ATOM_NAME_TO_ID = {
      atom_name: atom_id
      for atom_id, atom_name in enumerate(
          RNA_ATOM_NAMES,
          start=1,
      )
  }

  BASE_RING_ATOM_NAMES = (
      "N9", "C8", "N7", "C5", "C6",
      "N1", "C2", "N3", "C4",
  )

  BASE_ATOM_NAME_IDS = tuple(
      ATOM_NAME_TO_ID[atom_name]
      for atom_name in BASE_RING_ATOM_NAMES
  )

  ATOM_NAME_ALIASES = {
      "O1P": "OP1",
      "O2P": "OP2",
      "O3P": "OP3",
  }


  def normalize_atom_name(atom_name: str) -> str:
      atom_name = atom_name.strip().replace("*", "'")
      return ATOM_NAME_ALIASES.get(atom_name, atom_name)

  ### 生成 .pt 时

  from etflow.data.constants import (
      ATOM_NAME_TO_ID,
      UNKNOWN_ATOM_NAME_ID,
      normalize_atom_name,
  )

  atom_name_id = torch.tensor(
      [
          ATOM_NAME_TO_ID.get(
              normalize_atom_name(atom_name),
              UNKNOWN_ATOM_NAME_ID,
          )
          for atom_name in atom_names
      ],
      dtype=torch.long,
  )

  每个原子对应一个整数，所以形状为：

  atom_name_id: LongTensor[N]

  residue_index 也必须是：

  residue_index: LongTensor[N]

  对于多链 RNA，residue_index 必须在单个 .pt 内唯一。不要直接使用可能在不同链重复的 PDB residue number；应重新编码为 0, 1, ..., R-1。
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
