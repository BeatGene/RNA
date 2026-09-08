# RNA refinement `.pt` 数据契约（schema v2）

本契约对应 `Code_Flow_matching` 当前模型和 `scripts/build_refinement_pt.py`。
一个 Protenix seed/sample 与一个 native 结构生成一个 `.pt`。

## 最重要的三个约束

1. **预测结构是原子主索引。** `pos_pred[i]`、所有逐原子特征及图中的节点
   `i` 永远表示 Protenix CIF 的同一个 RNA 重原子。不能为了适配 native
   缺失原子而删除预测节点。
2. **native 缺失通过 `target_mask` 表达。** 能按“序列比对后的残基位置 +
   标准化原子名”唯一对应的原子为 `True`，其他为 `False`。禁止按 CIF 行号、
   `auth_seq_id` 或最近坐标强行配对。
3. **native 必须先刚体对齐到预测坐标系。** 用所有可靠匹配原子做 Kabsch
   对齐，再写入 `pos`。两份独立结构的平移/旋转没有学习意义，仅做质心居中
   不能解决任意整体旋转。

设 `N` 为预测 RNA 重原子数，`R` 为该 RNA 链残基/token 数，`E_s` 为静态
有向边数，`E_b` 为唯一共价键数，`E_x` 为 clash 排除对数。

## 必需字段

```python
sample = {
    "schema_version": 2,
    "sample_id": "7abc_seed_300_sample_0",

    # 坐标和监督
    "pos_pred": FloatTensor[N, 3],       # Protenix 坐标，Å
    "pos": FloatTensor[N, 3],            # 对齐后的 native target；见填充值规则
    "target_mask": BoolTensor[N],         # 该原子是否有可靠 native 坐标监督

    # 原子/残基
    "atomic_numbers": LongTensor[N],
    "atom_name_id": LongTensor[N],
    "residue_index": LongTensor[N],       # 必须连续，范围 0..R-1
    "sequence": "AUGC...",               # 长度 R
    "residue_type_id": LongTensor[R],     # A/C/G/U/unknown = 0/1/2/3/4
    "rnafm_embedding": Tensor[R, 640],    # 推荐磁盘 float16，加载转 float32

    # 静态图
    "edge_index": LongTensor[2, E_s],
    "edge_attr": FloatTensor[E_s, 7],

    # 几何约束
    "geometry_bond_index": LongTensor[2, E_b],
    "ideal_bond_length": FloatTensor[E_b],
    "clash_exclusion_index": LongTensor[2, E_x],

    # 与同一个 sample CIF 配对的 Protenix full_data JSON
    "atom_plddt": FloatTensor[N],         # 归一化到 0..1
    "atom_to_token_idx": LongTensor[N],   # 当前单 RNA 链中范围 0..R-1
    "token_pair_pae": Tensor[R, R],       # Å，可为 float16
    "token_pair_pde": Tensor[R, R],       # Å，可为 float16
    "contact_probs": Tensor[R, R],        # 0..1，可为 float16
}
```

### `pos` 对 native 缺失原子的填充值

模型和 PyG batch 要求 `pos` 与 `pos_pred` 同为 `[N,3]`。因此：

```python
pos = pos_pred.clone()
pos[target_mask] = aligned_native_coordinates
```

`target_mask=False` 的填充值只用于保持形状和构造完整的推理图，绝不能进入
坐标 loss、mobility target、no-regret loss 或 protect loss。几何正则（键长、
clash、碱基平面）仍可作用于全部预测原子，因为它们不依赖 native 标签。

每个样本至少要有 3 个非共线匹配原子；生成脚本默认还要求
`observed_atom_fraction >= 0.5`，低于阈值会进入 issues 而不生成训练样本。
可用 `--min-observed-atom-fraction` 调整，但不建议为了追求样本数盲目降低。

## 映射规则

### 残基映射

- 优先从 native CIF 的 `_entity_poly_seq` 得到完整聚合物序列，因此整个残基
  即使没有任何 `_atom_site` 行也不会导致后续残基错位。
- 用 `_chem_comp.mon_nstd_parent_comp_id` 将可识别修饰残基映射到母体碱基。
- 将 Protenix/RNA-FM 序列与 native 完整序列做全局序列比对。
- 当前脚本要求 identity 与 query coverage 均不低于 0.8，并把实际值写入
  审计字段。单链标准数据通常应为 1.0；低于 1.0 要看报告。

### 原子映射

- 在已比对残基内，用 `normalize_atom_name()` 后的原子名匹配；`O1P/O2P/O3P`
  会规范为 `OP1/OP2/OP3`，星号会规范成撇号。
- native 多 altloc 时优先空 altloc/`A`，再取 occupancy 较高者；只读取第一个
  model。
- Protenix `full_data_sample_k.json` 中的 `atom_to_token_idx` 是 token 映射权威；
  CIF 是原子身份和坐标权威。sample `k` 的 CIF 只能配 sample `k` 的 JSON。

## 图与化学约束

- `edge_index` 保存双向静态边：残基内共价键（type 0）、相邻残基
  `O3'—P`（type 1）、相邻残基 `C4'—C4'` sequence edge（type 2）。
- type 3 留给模型每步根据当前坐标生成的 dynamic-radius edge；不得写入 `.pt`。
- `edge_attr` 始终为 7 类 one-hot。PAE/PDE/contact 不是 edge type，dataset
  会在最终动态边确定后按 token 对索引它们。
- `geometry_bond_index` 只存唯一、无向的真实共价键；`ideal_bond_length` 来自
  标准 RNA 模板，而不是从 native 实测距离复制。
- `clash_exclusion_index` 存唯一的 1–2 和 1–3 原子对，且第一行索引小于第二行。

## RNA-FM 文件

RNA-FM 源文件的实际 payload 字段是 `residue_embedding`，不是
`rnafm_embedding`；生成脚本选中单链切片后改名写入最终 sample。源文件还含
`sequences`、`chain_offsets`、`original_chain_ids` 和
`expected_protenix_chain_ids`，脚本用它们防止拿错链。

本地报告表明：

- `RNA_FM_EMBED_20260811T144010Z` 只是 5 个 PDB 的 smoke run；
- 完整生成是 `RNA_FM_EMBED_20260811T144755Z`：2241 个 PDB（2236 新生成、
  5 个复用），0 failure；
- `RNA_FM_VALIDATE_20260811T145152Z` 验证 2241 个 PDB 全部 PASS。

所以服务器 `~/Data_FM/RNA_FM_embeddings` 可作为源，但 bulk 生成前仍建议先用
几个当前单链 split ID 做 smoke test。

## 推荐审计字段（模型不直接使用）

生成脚本还保存：

- PDB、native/predicted chain、seed、sample、模型名及四个源文件路径；
- `protenix_original_token_index`（pair matrix 切片前的 token 编号）；
- `native_sequence_identity`、`native_sequence_coverage`；
- `observed_atom_count/fraction`、`observed_residue_mask`；
- Kabsch rotation/translation 和 `pre_refinement_aligned_rmsd`；
- generator/mapping/edge 版本。

`observed_residue_mask`、变换矩阵和源路径是审计信息，不应拼入 node feature。

## 与当前模型的维度对应

dataset 构造的基础 `node_attr` 是：

```text
RNA-FM 640 + atom-name one-hot 29 + residue-type one-hot 5 = 674
```

所以 YAML 中 `node_attr_dim: 674` 正确。开启 `confidence_conditioning` 后，模型
内部再添加 1 维 pLDDT；不应把 YAML 改成 675。静态 `edge_attr_dim: 7` 也正确，
模型内部再添加 4 维 PAE/PDE/contact confidence edge feature。

旧 v1 schema 的字段中没有需要删除的模型输入；真正缺少的是
`target_mask`、native→prediction 刚体对齐规则，以及 RNA-FM 源 payload 与最终
字段名之间的转换说明。

## 生成命令

先 smoke test：

```bash
python ~/Code_Flow_matching/scripts/build_refinement_pt.py \
  --prediction-root ~/Data_V1 \
  --native-root ~/pdb_data \
  --rnafm-root ~/Data_FM/RNA_FM_embeddings \
  --output-root ~/Data_Refinement_PT \
  --split train --pdb-id 125D --fail-fast
```

检查 `generation_manifest.tsv`、随机打开若干 `.pt` 后再全量运行：

```bash
python ~/Code_Flow_matching/scripts/build_refinement_pt.py \
  --prediction-root ~/Data_V1 \
  --native-root ~/pdb_data \
  --rnafm-root ~/Data_FM/RNA_FM_embeddings \
  --output-root ~/Data_Refinement_PT
```

默认不覆盖已有 `.pt`，可断点续跑；明确需要重建时加 `--overwrite`。
