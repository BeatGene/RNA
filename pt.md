• 下面给出一套确定的 .pt 数据契约和代码修改方案。设计原则是：

  - .pt 保存原始、可追溯的数据；
  - node_attr 在 dataset.py 中构造；
  - 置信度单独保存，通过 confidence_conditioning 控制；
  - 动态边生成后，再为最终边索引 PAE/PDE/contact；
  - 不需要修改真正使用的 networks/torchmd_net/model_dynamics.py 内部计算逻辑。

  设：

  - (N)：原子数；
  - (R)：RNA 残基数；
  - (T)：Protenix token 数；
  - (E_s)：静态有向边数；
  - (E_b)：唯一共价键数；
  - (E_x)：clash 排除对数。

  # 一、每个 .pt 文件保存的数据

  推荐结构如下：

  sample = {
      # 版本和来源
      "schema_version": 1,
      "sample_id": "7abc_chain_A_seed_101_sample_0",

      # 坐标与原子
      "pos": ...,                       # FloatTensor [N, 3]
      "pos_pred": ...,                  # FloatTensor [N, 3]
      "atomic_numbers": ...,            # LongTensor  [N]
      "atom_name_id": ...,              # LongTensor  [N]
      "residue_index": ...,             # LongTensor  [N]

      # 序列与基础节点特征
      "sequence": "AUGC...",
      "residue_type_id": ...,           # LongTensor  [R]
      "rnafm_embedding": ...,           # FloatTensor [R, 640]

      # 静态图
      "edge_index": ...,                # LongTensor  [2, E_s]
      "edge_attr": ...,                 # FloatTensor [E_s, 7]

      # 几何约束
      "geometry_bond_index": ...,       # LongTensor  [2, E_b]
      "ideal_bond_length": ...,         # FloatTensor [E_b]
      "clash_exclusion_index": ...,     # LongTensor  [2, E_x]

      # Protenix置信度
      "atom_plddt": ...,                # FloatTensor [N]
      "atom_to_token_idx": ...,         # LongTensor  [N]
      "token_pair_pae": ...,            # FloatTensor [T, T]
      "token_pair_pde": ...,            # FloatTensor [T, T]
      "contact_probs": ...,              # FloatTensor [T, T]
  }

  ## 1. 坐标和原子信息

  ### pos: FloatTensor[N,3]

  真实 native RNA 的重原子坐标，单位 Å。

  要求：

  - 与 pos_pred 原子数量相同；
  - 与 pos_pred 原子顺序完全相同；
  - 不需要预先质心居中，模型内部会处理。

  最重要的是原子对应关系：

  pos[i]
  pos_pred[i]
  atomic_numbers[i]
  atom_name_id[i]
  residue_index[i]
  atom_plddt[i]
  atom_to_token_idx[i]

  必须全部表示同一个原子。

  ### pos_pred: FloatTensor[N,3]

  从同一个 Protenix sample 的 CIF 中读取的预测坐标。

  例如：

  seed_101/sample_0.cif

  必须和同一个 sample 的：

  full_data_sample_0.json

  配对，不能将 sample 0 的坐标与 sample 1 的 pLDDT/PAE 混用。

  ### atomic_numbers: LongTensor[N]

  元素原子序数，例如：

  C = 6
  N = 7
  O = 8
  P = 15
  S = 16

  来源是 CIF/native 原子记录中的 element 字段。

  ### atom_name_id: LongTensor[N]

  使用 /C:/Users/49586/Desktop/Learning/Laboratory/Admis/graduate_first/RNA/Code_Flow_matching/etflow/data/
  constants.py:18 中的全局映射：

  ATOM_NAME_TO_ID = {
      "P": ...,
      "OP1": ...,
      "C4'": ...,
      "N9": ...,
  }

  未知原子使用：

  UNKNOWN_ATOM_NAME_ID = 0

  这个字段具有两个用途：

  - 供 base_plane_loss() 判断碱基环原子；
  - 转成 one-hot 后作为 atom role 输入网络。

  因此不需要再额外保存一个重复的 atom_role_id。精确原子名称本身就是一种更细粒度的 atom role。

  ### residue_index: LongTensor[N]

  每个原子所属的残基编号，必须重新映射成：

  0, 1, 2, ..., R-1

  例如：

  residue_index = tensor([
      0, 0, 0, ...,  # 第0个核苷酸的所有原子
      1, 1, 1, ...,  # 第1个核苷酸的所有原子
  ])

  不要直接使用可能不连续的 PDB auth_seq_id。

  ## 2. 序列和基础节点特征

  ### sequence: str

  RNA 序列：

  "AUGCC..."

  长度应等于 (R)。

  ### residue_type_id: LongTensor[R]

  推荐固定映射：

  A = 0
  C = 1
  G = 2
  U = 3
  其他/未知 = 4

  这是残基级数据，不需要为每个原子重复保存。dataset.py 会通过：

  residue_type_id[residue_index]

  扩展到原子。

  ### rnafm_embedding: FloatTensor[R,640]

  RNA-FM 对每个核苷酸产生的 640 维表示。

  注意：

  - RNA-FM 可能包含 BOS/EOS token，写入 .pt 前必须去掉；
  - 最终第一维必须严格等于 RNA 残基数 (R)；
  - 推荐在磁盘中保存为 float16，加载时转换为 float32；
  - 不要扩展成 [N,640] 再保存，否则同一残基的向量会重复很多次。

  dataset.py 中再执行：

  rnafm_atom = rnafm_embedding[residue_index]

  ## 3. 静态图

  ### edge_index: LongTensor[2,E_s]

  仅保存永久静态边：

  - 核苷酸内部共价键；
  - 相邻核苷酸间磷酸二酯键；
  - 相邻残基的 C4'–C4' sequence-neighbor 边。

  建议每条静态边保存两个方向：

  i → j
  j → i

  不要把 radius dynamic edges 写入 .pt；它们由模型每一步重新生成。

  ### edge_attr: FloatTensor[E_s,7]

  7维边类型 one-hot。按照当前 smoke test 的约定：

  type 0：残基内部共价键
  type 1：残基之间磷酸二酯键
  type 2：相邻残基 C4'–C4' 边
  type 3：动态 radius 边，静态.pt中不应出现
  type 4–6：预留

  例如：

  edge_attr = F.one_hot(
      edge_type,
      num_classes=7,
  ).float()

  这里始终保持 7 维。PAE/PDE/contact 不是边类型，不在数据生成时拼入这个字段。

  ## 4. 几何约束

  ### geometry_bond_index: LongTensor[2,E_b]

  只包含真正的共价键：

  - 核苷酸内部共价键；
  - 相邻核苷酸间磷酸二酯键。

  建议每条共价键只保存一次，不需要双向重复：

  i—j

  ### ideal_bond_length: FloatTensor[E_b]

  与 geometry_bond_index 每一列严格对应：

  geometry_bond_index[:, k]
  ideal_bond_length[k]

  理想长度来自 CCD/标准 RNA 化学模板，不应直接使用 native 中测量出的长度作为理想值。

  ### clash_exclusion_index: LongTensor[2,E_x]

  保存：

  - 1–2 共价原子对；
  - 1–3 键角原子对。

  每对只保留一次，建议满足：

  clash_exclusion_index[0] < clash_exclusion_index[1]

  ## 5. Protenix 置信度

  全部来自开启：

  --need_atom_confidence true

  后生成的 full_data_sample_*.json。

  ### atom_plddt: FloatTensor[N]

  范围规定为：

  0–1

  如果读取值是 0–100，生成脚本必须先除以 100。

  ### atom_to_token_idx: LongTensor[N]

  每个原子对应的 Protenix token，范围：

  0 <= atom_to_token_idx[i] < T

  即使纯 RNA 中 token 通常对应核苷酸，也必须使用 Protenix 的真实映射，不能用 residue_index 替代。

  ### token_pair_pae: FloatTensor[T,T]

  单位 Å，具有方向性：

  PAE[i,j] 不一定等于 PAE[j,i]

  ### token_pair_pde: FloatTensor[T,T]

  token 对之间的预测距离误差，单位 Å。

  ### contact_probs: FloatTensor[T,T]

  范围 0–1，表示 token 对成为接触的预测概率。

  ## 6. 推荐额外保存的审计信息

  模型不直接使用，但强烈建议保留：

  "native_structure_id": "7ABC",
  "protenix_seed": 101,
  "protenix_sample": 0,
  "protenix_model_name": "...",
  "atom_mapping_version": 1,
  "edge_type_version": 1,
  "rnafm_model_name": "...",

  这能避免以后无法追踪数据来自哪个模型、seed 和映射规则。

  ———