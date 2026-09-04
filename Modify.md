# 数据生成

## 对齐
提取的prediction的原子坐标应与native的原子坐标对齐，并且做Kabsch对齐

## edge_attr与node_attr & 模型实际上没有使用 RNA-FM
注意输出字段edge_attr与node_attr；并且配置中写好node_attr_dim与edge_attr_dim。目前node_attr有RNA-FM生成的640维残基特征向量广播得到[node_atoms,640] 

[//]: # (ToDo)
edge_attr目前打算根据边的种类进行独热编码，边的种类与构建还未确认 

## 当前图不是描述中的“RNA 化学图”
当前生成图的脚本是之前版本的脚本，现在生成应当1.永久保留共价键 2.区分共价键与磷酸二酯键 3.保留原子所属残基以及具体的atom_name chain_id不用，因为目前数据集只专注于一条链的情况


**当前模型已经通过第一阶段的CPU/GPU工程可运行性验证，可以进入.pt数据生成阶段。下一步最重要的是明确并验证.pt数据契约，然后用少量真实样本做过拟合测试。**
# 模型修改

## 网络显式看到原始预测结构 Modify_1  DONE

必须增加的测试

  至少验证下面四项： 
  
1.source能否影响输出
  v1 = model(pos=x_t, pos_source=source_1, ...)
  v2 = model(pos=x_t, pos_source=source_2, ...)
  assert not torch.allclose(v1, v2)

2.平移不变
v_shift = model(pos=x_t + shift, pos_source=source + shift, ...)
assert torch.allclose(v, v_shift, atol=...)

3.旋转等变
v_rot = model(pos=x_t @ R.T, pos_source=source @ R.T, ...)
assert torch.allclose(v_rot, v @ R.T, atol=...)

4. 输出零质心
assert center(v, batch).abs().max() < tolerance

另外应单独验证 identity pair:x_pred=x_native 训练后输出速度应接近零。


## 多样图 Modify_2 DONE->PT
covalent graph； 
sequence-neighbor graph；
动态 radius graph；

1. covalent graph->生成.pt文件时生成

  表示真实化学键
  包括：

  - 同一核苷酸内部的共价键；
  - 相邻核苷酸之间的磷酸二酯键：

  生成时需要：

  - residue_index
  - atom_name
  - 核苷酸类型 A/C/G/U
  - 标准 RNA 原子键模板

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

`from protenix.data.core import ccd

def get_rna_bond_template(residue_name):
    component = ccd.get_component_atom_array(ccd_code=residue_name, keep_leaving_atoms=False, keep_hydrogens=False,)
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
    return bond_template`

  标准残基直接读取：

rna_bond_templates = { residue_name: get_rna_bond_template(residue_name) for residue_name in ["A", "C", "G", "U"]}

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

2.sequence-neighbor graph ->生成.pt时生成

  它表示“序列上相邻”，并不等同于化学键。

  虽然相邻残基已经通过 O3'—P 相连，但如果只靠原子化学键传播，一个残基的信息传到下一个残基的碱基部分需要经过很多层。

  因此可以给相邻残基的代表原子增加 shortcut：

  C4'(i) ↔ C4'(i+1)

  不建议把相邻两个残基的全部原子两两相连，边数会非常大。

  第一版建议：
  sequence_window = 1
  anchor_atom = "C4'"

  即只连接相邻残基的 C4' 原子。

3.动态 radius graph->每一步实时生成

  表示当前flow状态x_t中的空间近邻：

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


4.residue-level base-pair/stacking/global graph->第二阶段目标

  这里节点不再是原子，而是核苷酸残基。

  **Base-pair 边**

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

  **Stacking 边**

  表示碱基堆积。通常根据：

  - 碱基中心距离；
  - 碱基平面法向量夹角；
  - 两个碱基的相对位姿

  生成候选边。

  **Global 边**

  用于长距离通信。例如连接：

  i ↔ i+4
  i ↔ i+8
  i ↔ i+16
  i ↔ i+32

  这种 dilated sequence graph 能让长 RNA 的远距离残基在较少层数内通信。

{
    "pos": pos,
    "pos_pred": pos_pred,
    "atomic_numbers": atomic_numbers,
    "sequence": str,
    # 原子所属残基，范围 0 ... num_residues-1
    "residue_index": residue_index,
    #建议保存整数编码，PyG batching 更方便 # 必须使用整个数据集统一的 atom-name 编码表
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
    
edge_attr.shape == [num_edges, 7]

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

## 增加化学约束/LOSS & 三种模式 Modify_3 DONE->PT

  ### 建议加入的四种几何 loss

  #### 共价键长度 loss

  只使用以下两类边：

  edge_attr[:, 0]：核苷酸内部共价键
  edge_attr[:, 1]：磷酸二酯键

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

  #### Steric clash loss

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

  ### Steric clash loss 具体怎么算

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

  为了不与现有作为完整静态图的 bond_index 混淆，建议 .pt 使用：

  geometry_bond_index: LongTensor[2, E_bond]
  ideal_bond_length: FloatTensor[E_bond]

  其中只包含：

  - 核苷酸内部共价键；
  - 相邻核苷酸的磷酸二酯键。

  最好每根键只保存一次，不保存双向副本。

当前第一版建议使用 Bondi 半径，只处理 RNA 常见的 C/N/O/P/S：

BONDI_VDW_RADII = {
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    15: 1.80,  # P
    16: 1.80,  # S
}

### 生成逻辑

在以后生成 .pt 时可以使用：构建clash_index(要去除掉1-2共价键和1-3)

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

唯一需要记住的是：CUDA float32 下存在 scatter 非确定性，所以固定随机种子也不保证逐位相同。它通常不妨碍训练，但如果以后 要求严格复现实验，需要专门处理确定性图聚合算子，并记录 CUDA、PyTorch、PyG 与 torch_cluster 版本。
  ## 创新一：prediction-conditioned residual flow 
###计划在生成 .pt 时
    把 atom_role 正确编码并拼入 node_attr，且最终让 node_attr_dim 等于实际总维度->PT 
    需要注意不要把无序类别 ID 直接当连续数值使用，例如：atom_role = 17.0 模型可能错误理解为角色 17 比角色 3“更大”。建议采用：
  - atom-role one-hot；或
  - atom_role_id + nn.Embedding


4. 是的，A/C/G/U/其他类别信息也可以直接拼进 node_attr。
  最简单的是使用 5 维 one-hot：

  A     = [1, 0, 0, 0, 0]
  C     = [0, 1, 0, 0, 0]
  G     = [0, 0, 1, 0, 0]
  U     = [0, 0, 0, 1, 0]
  other = [0, 0, 0, 0, 1]

  同一个核苷酸内的所有原子使用相同的 residue one-hot。最终例如：

  node_attr = torch.cat(
      [
          rna_fm_feature,       # 640维
          atom_role_one_hot,    # 假设 R 维
          residue_one_hot,      # 5维
      ],
      dim=-1,
  )

  那么：

  node_attr_dim = 640 + R + 5

  如果以后再加入 per-atom pLDDT：

  node_attr_dim = 640 + R + 5 + 1

  如果还加入投影后的 Protenix single embedding，也继续累加其投影维度。

  RNA-FM 本身已经包含核苷酸类型和上下文信息，因此 5 维 one-hot 不是绝对必需；但它代价极小，能够明确告诉网络该残基是 A/C/
  G/U，建议保留。


  不仅令 x0=pos_pred，还让整个轨迹显式条件于：
  - 原始 pos_pred；
  - x_t-pos_pred；
  - atom role、residue identity、chain/residue index；
  - Protenix single/pair embeddings；
  - pLDDT、PAE/PDE、contact probability。

  ## 2a. Protenix 输出文件中有没有这些数据

  需要区分“构象 CIF 文件”“置信度 JSON”和“模型内部 embedding”。

  ### pLDDT

  你本地这版 Protenix 会把 atom pLDDT 乘以 100 后写入 CIF 的 B-factor 字段，见 /C:/Users/49586/Desktop/Learning/
  Laboratory/Admis/graduate_first/RNA/Protenix/Protenix/runner/dumper.py:201。

  所以：

  - CIF 中可以获得每原子 pLDDT；
  - 它不是真正的实验 B-factor，而是借用该字段保存置信度；
  - CIF 中通常是 0–100，内部 atom_plddt 是 0–1。

  ### PAE、PDE、contact probability

  它们不在普通 CIF 坐标文件里。

  运行 Protenix 时打开：

  --need_atom_confidence true

  会额外生成：

  *_full_data_sample_*.json

  其中包含：

  - atom_plddt：[N_atom]
  - token_pair_pae：[N_token,N_token]
  - token_pair_pde：[N_token,N_token]
  - contact_probs：[N_token,N_token]
  - atom_to_token_idx：原子到 token 的映射

  这些字段在本地源码 /C:/Users/49586/Desktop/Learning/Laboratory/Admis/graduate_first/RNA/Protenix/Protenix/protenix/
  model/sample_confidence.py:95 中构造，并由 /C:/Users/49586/Desktop/Learning/Laboratory/Admis/graduate_first/RNA/
  Protenix/Protenix/runner/dumper.py:270 写入 JSON。Protenix 官方文档也说明了 CIF 与置信度 JSON 的输出结构。Protenix 输
  出文档

  更合适的方式是作为软条件输入，让模型自己决定是否采用：

  ### 节点条件

  对原子 (i)：

  node_attr_i =
      RNA-FM residue embedding
      + atom role embedding
      + residue identity embedding
      + Protenix single embedding[token_i]
      + atom pLDDT_i

  single embedding 是 token 级的，可以通过 atom_to_token_idx 扩展到原子。

  ### 边条件

  对当前图中的边 (i\rightarrow j)：

  edge_attr_ij =
      当前边类型
      + PAE[token_i, token_j]
      + PAE[token_j, token_i]
      + PDE[token_i, token_j]
      + contact_prob[token_i, token_j]
      + projected_pair_embedding[token_i, token_j]

  PAE 有方向性，因此最好同时保存 PAE_ij 和 PAE_ji。PDE 更接近“这对 token 的距离可能错多少”；contact probability 是结构先验，不严格等于误差置信度。

  第一版我建议先只加入：

  1. atom pLDDT；
  2. PAE/PDE/contact probability；
  3. Protenix single embedding。

  pair embedding 是 (O(L^2\times128))，存储和载入都很重，适合放到后续消融实验中，而不是第一步全部加入。

  ## 置信度与真实误差的相关性



  所以，AF3 已经证明了“置信度能反映相应类型的准确度”；但“这些置信度能否预测你的 Protenix RNA refinement 位移”仍然需要你
  在自己的数据上证明。

  建议至少测：

  - pLDDT 与真实 per-atom local lDDT：验证原指标；
  - pLDDT 与 Kabsch 对齐后原子误差：验证是否能指导 refinement；
  - PDE 与真实 pair-distance error；
  - contact probability 与 native contact 的 AUROC/AUPRC；
  - PAE 与 RNA helix/domain 相对位置误差；
  - Spearman 相关系数和分箱校准曲线。

  最关键的是第二组，因为你的目标不是解释置信度，而是预测修正位移。

  ———

  ## 2c. 会不会限制性能上限

  如果采用硬冻结或硬缩放，会。

  例如：

  delta_x = (1 - confidence) * delta_x

  一旦某个原子“高置信但预测错误”，模型无论多强都无法充分修正，确实会降低理论上限。

  但如果只是把 confidence 连接到 node_attr/edge_attr：

  delta_x = model(
      coordinates,
      plddt,
      pae,
      pde,
      contact_prob,
  )

  模型可以学习使用，也可以学习忽略它，理论上不会因为置信度而失去移动自由度。有限数据下仍可能出现过拟合，因此需要消融，但没有人为设置的硬上限。


  因此推荐的原则是： confidence 告诉模型“哪里可能存在什么类型的不确定性”，但不直接决定“这个原子能不能动、最多动多少”。

  实验上应同时报告全原子 RMSD、lDDT、键长/冲突、不同初始质量分组下的提升，以及 refinement 后变差的比例。最合适的实施顺序
  是：先加入 pLDDT+PAE/PDE/contact，完成置信度—真实误差分析，再决定是否值得抽取体积很大的 Protenix single/pair
  embeddings。

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

  

比较稳妥的创新表述是：面向 Protenix/AF3-like predictor errors 的 prediction-conditioned RNA correction flow，通过学习型置信度门控保护正确 区域，并联合更新 RNA interaction topology、nucleotide geometry 和 refinement QA。

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


  最终评价不要只报平均 RMSD，至少还应包括：

  - TM-score、lDDT；
  - INF_all/WC/NWC、stacking recovery；
  - clashscore、bond/angle violation、suite/pucker/χ outlier；
  - 相对 Protenix raw 的 improvement rate；
  - good-input damage rate；
  - raw/refined 经 QA 选择后的实际收益；
  - 按 initial RMSD、长度、pLDDT、训练集结构相似度分层。




1.真正打开 RNA 特征

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

2.加 confidence mobility gate

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

3.加终点 self-conditioning

  每一步预测：

  [
  \hat x_1^{(k)}
  ]

  下一步将上一次的终点估计重新编码为条件：

  x_t + previous_endpoint_estimate → velocity network

  这有助于模型发现 ODE 轨迹已经偏离合理 RNA 几何，并在后续步骤纠正。改动比更换 EquiformerV2 小。

4.改进采样器

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

  > 同一个轻量等变速度网络，通过 predictor-specific source conditioning、置信度门控和 RNA interaction constraints，实现 no-regret RNA refinement。

  最重要的是先证明：

  - 相比 Protenix raw，更多 target 得到改善；
  - 原本较好的结构不会频繁被修坏；
  - INF、clash 和 RNA stereochemistry 与 RMSD 同时改善；
  - 同参数量 direct regression 无法达到同样效果。


