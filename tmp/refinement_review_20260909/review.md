# Code_Flow_matching 审查（2026-09-09）

本轮只审查，没有修改模型、schema、生成器或 YAML。audit.py 是新增的独立复现脚本。
结论：显式条件化、向量通道 Linear、标量 LayerNorm 的旋转等变性设计成立；数据生成与 masked mobility 仍有必须修复的问题，不能据原先输出直接宣告可以全量生成和训练。

## 1. 等变性

采用列向量约定，Q 为旋转，a 为平移，网络速度应满足：

    v(Q x_t + a, Q x_pred + a, t) = Q v(x_t, x_pred, t)

两组坐标必须同时变换。

- model_dynamics.py:483–499 的 source edge 分支只使用距离及 RBF，属于旋转不变量。
- :510–528 的 source_x 来自 NeighborEmbedding 的距离和标量特征，同样是旋转不变量。
- :535–549 的 delta = pos - pos_source 按向量变换，delta_norm 不变；delta_dir 乘仅依赖长度和 t 的标量通道，仍按向量变换。
- :91 / :149 的 vec_proj 在 [N,3,H] 的 H 维进行无偏置投影，保留三维空间轴。
- utils.py:260–261 的 vec1_proj、vec2_proj 也是 H 维无偏置投影；输出头两个 GatedEquivariantBlock 共使用四个这样的实例。连同六个 attention 的 vec_proj，当前六层模型共十个直接作用于向量的 Linear 实例。
- v_proj、dv_proj、o_proj、source_delta_mlp、source_node_proj 等处理的是标量通道或向量的标量系数；名称含 v 不表示直接对三维坐标做 Linear。有 bias 不构成等变性问题。
- :574 的 out_norm 处理 [N,H] 标量，安全。输出头的 LayerNorm 也处于标量 update_net 中。
- CoorsNorm 使用三维范数作除数，安全。
- 源码注释 “vec ... (all invariant)” 不准确，应是 equivariant。

对于 V∈R^(3×H)，L(V)=VW^T，因此 L(QV)=QL(V)。若向量层加普通非零 bias，或对三维空间轴用任意 Linear(3,3)，就不再有这个一般保证。

### 范围限制

1. YAML :45 开启 so3_equivariant，attention :220 使用 cross。叉积在 proper rotation 下等变，在反射下多一个 det(Q)。当前架构只应宣称旋转/平移对称性，不应宣称一般 O(3)/E(3) 反射等变性。
2. models/utils.py:49 用有邻居上限的 radius_graph。CPU 实现存在方向偏置的官方警告；补反向边并不能恢复原本丢弃的邻居。完整链路应分别检查图是否保持不变和网络输出是否等变。GPU 的实际行为需在服务器后端实测。
3. model.py:1239 的 sample 返回居中坐标，没有加回原 pos_pred 质心。速度的等变性和“原坐标系中的最终坐标等变性”是不同接口约定。若要后者，输出时需加回预测结构质心。

官方参考：

- [PyG radius_graph 文档](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.radius_graph.html)

## 2. edge_vec 最小改法

当前 :506 对分母施加 0.01 Å 下限，并没有删除短边。其效果是短边方向向量长度小于 1；它本身也不破坏旋转等变性。

只针对非零边，YAML 改为 clip_during_norm: false 就能取消该下限。
若同时处理不同原子坐标完全重合的情况，建议用以下两行替换 :503–508 的归一化块：

```python
denom = edge_weight.masked_fill(edge_weight == 0, 1.0).unsqueeze(-1)
edge_vec = edge_vec / denom
```

所有非零边按真实长度归一化；恰为零的向量仍为零；无 0.01 Å 短边阈值。若替换整块，旧 clip_during_norm 参数在此处不再控制行为，配置与说明应同步清理或标记弃用。极短非零边的导数可能很大，这是数值稳定性与等变性不同的问题。

## 3. 高优先级问题

### P1：CIF 原子名未解码，导致错误 mask、unknown atom 和不完整图

build_refinement_pt.py:75–82 / :110：find_values 得到原始 CIF 字符串，保留语法引号；normalize_atom_name 仅 strip 空白和处理星号/别名。

实际复现：

- 原始 `'P'`、`'C4''` 解码后应为 P、C4'，脚本却保留外围引号，均变成 atom_name_id=0。
- 两份坐标完全相同、原子完全相同，仅 native 使用双引号、prediction 使用单引号，就报 only 0 native atoms could be mapped。
- 两边使用相同引号时 build_sample 能生成；四原子例子 observed=4，但所有 atom_name_id=0、num_bonds=0。
- 原有测试没有断言 atom_name_id 或图连接，因此通过并不能排除此错误。

应在 CIF 解析层用 gemmi.cif.as_string 或 Column.str 解码所有相关字符串列，再标准化原子名；不能简单 strip 单引号，否则可能删掉 C4' 本身的撇号。

[Gemmi 官方 CIF 文档](https://github.com/project-gemmi/gemmi/blob/master/docs/cif.rst) 明确解释 DOM 保存含引号的 raw strings。

### P1：整个残基无监督时，residue_rms_error 反向传播出现 NaN

model.py:603–633 对所有残基计算 sqrt(sum_sq / observed_count.clamp_min(1))，之后 :1096–1110 才排除无标签残基。

无标签残基的 sum_sq 恒为 0，sqrt(0) 反向出现无穷导数；后续零梯度与该导数相乘产生 NaN，并会沿 no_regret 分支传回模型。已观察残基恰好零误差也存在 sqrt 的奇异点。

从源码提取原函数，使用 torch.index_add 实现其 scatter(sum) 后复现：两个残基、第二个全无标签，loss=1.0，但第二个残基的 prediction.grad=[NaN,NaN,NaN]。

修复必须发生在 sqrt 之前：只对有效项计算，并安全处理有效项的零误差；或者对平方均值施加明确的数值下限。单纯在最终 loss 乘 mask 或排除 valid_residue 不够。

### P1：A/G 键模板漏 N9—C4

build_refinement_pt.py:378–383 的 A/G 模板都没有该键。build_graph 的直接复现表明，N9—C4 既不在 geometry_bond_index，也不在 clash_exclusion_index。

这不仅漏掉一个键长项，还会把真实相邻原子送入非键合 clash 惩罚，并遗漏相关 1–3 排除对。
应补全模板，并验证完整环闭合、键集合及 1–3 排除集合。修复后需重建旧 pt，默认 SKIPPED 不会更新旧图。

[RCSB A CCD](https://files.rcsb.org/ligands/download/A.cif) 的 _chem_comp_bond 明确列有 N9 C4。

## 4. 其他需要明确的问题

- **序列比对不保证唯一。** global_align :255 在多个同分路径中选第一个；build_sample :460 允许 identity 和 query coverage 最低 0.8，:464–471 仍映射错配位置的同名原子。重复序列、缺少完整 entity_poly_seq 或真实序列不一致时，可能将不可靠配对标为 True。这与 schema 中“唯一可靠对应”不完全一致。应对错配/歧义分开处理或拒绝，并保存逐残基/逐原子的对应记录用于审计。标准且完整一致序列下，此风险较小。
- **残基缺失测试并未覆盖完整路径。** test_global_alignment_handles_missing_native_residue 只测试 ACGU→ACU，coverage=0.75，甚至低于 build_sample 的 0.8 接受门槛。它不是通过 entity_poly_seq 保留缺失残基位置的端到端测试。
- **smoke test 未覆盖 mask=False。** smoke_test_synthetic.py:201 将 target_mask 全设 True，不能证明 masked mobility 的梯度正确。
- **smoke 等变测试未激活非零 source delta。** :752–759 的 pos 与 pos_source 相同，只测试绕 z 轴的一个旋转。应增加 x_t≠x_pred、任意轴旋转、共同平移、真实邻居上限、零距离边及目标缺失情形。
- **YAML 改变了训练方法。** RNA_test.yaml:52 为 residual；model.py:961–970 固定 t=0 且 x_t=x0；因此 vec 的 delta 初始化恒为零，source_rbf-current_rbf 也恒为零。source 节点分支仍执行，但此配置不是跨时间 flow matching。改成 residual 是实验选择，不是 schema/mask 修复必需步骤。
- **断点续跑只检查文件存在。** :619 跳过任何已有 pt，不验证 schema/generator/source fingerprint；:656–676 每次覆盖 manifest/issues/summary。故“可续跑”成立，但不等于验证旧文件有效或保留跨运行审计历史。
- **训练 CLI 的环境前提。** datamodule.setup 总是加载 train/val/test。只生成单个 train PDB 后直接 train.py，若 val/test 还没有 pt 会失败。8 GPU、BF16、early stopping 是配置值，不能据此推断显存充足或 refinement 有效。

## 5. 原输出中正确的部分

- 以预测重原子为主索引，native 缺原子用 target_mask；不删除预测节点，方向正确。
- 当前逐点位移目标下，应消除 native 相对预测的任意刚体位姿；生成时 Kabsch 是合理实现。并非所有 refinement 方法都必须“预先 Kabsch”，但本实现需要一致坐标系。
- Kabsch 实现包含反射修正；随机旋转和平移复现 RMSD≈2.00e-7 Å、最大坐标误差≈4.77e-7 Å。
- 在正确 Kabsch 和缺失位置填 pos_pred 的条件下，两份完整张量的质心相同（忽略浮点误差）；不应把后续各自居中自动判定为缺失标签污染。
- node_attr_dim=674 正确：640 + 29（28 个 atom names 加 unknown）+ 5。confidence 开启后模型内部追加 1 维，实际网络 node_attr 为 675。
- 静态 edge_attr_dim=7 正确；模型内部追加 4 维 confidence，再结合 RBF。
- RNA-FM payload 的 residue_embedding→最终 rnafm_embedding 转换与本地生成器一致。
- Protenix 本地 dumper 同时按对应 rank 命名 CIF 与 full_data JSON；使用相同 sample 后缀配对方向正确。本地 sample_confidence.py 确实导出 atom_to_token_idx。服务器实际文件是否存在且一致，仍需真实样本检查。
- 坐标 loss、mobility/no-regret/protect 的前向选择都已经使用 mask；问题主要是上述 sqrt 的反向安全性。
- RNA-FM 报告已核实：144010Z=5；144755Z=2236 新建+5复用、0失败；145152Z=2241 PASS。此结论仅对应当时报告，不是服务器当前文件完整性检查。

## 6. 验证范围

运行环境：torch 2.6.0+cpu，gemmi 可用；torch_geometric、torch_cluster、Lightning 不可用。

已运行：

```text
python Code_Flow_matching/scripts/test_build_refinement_pt.py -v
2 tests: PASS

python tmp/refinement_review_20260909/audit.py
CIF 引号、漏键、masked RMS NaN 均复现。
Kabsch 旋转/平移数值验证通过。
无 bias 的向量通道 Linear 旋转误差约 2.38e-7。
建议的零距离归一化：1e-3 边得到单位向量，零边得到零向量。

python Code_Flow_matching/scripts/smoke_test_synthetic.py --device cpu
在依赖检查阶段停止：Missing torch_geometric。
```

未声称完整网络前向/反向、PyG batching、8 GPU/BF16 训练或真实结构 refinement 改善已验证。
外部文档检索先尝试 agent-reach 的 Exa，但本机未配置 exa server，随后使用可用网页检索核对官方文档。
