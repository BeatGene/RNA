# RNA Refinement 项目代码审计报告

- **审计人**:Claude Code(基于对仓库的静态审读)
- **审计日期**:2026-09-09
- **审计范围**:
  - 模型侧 `Code_Flow_matching/`(约 5100 行)
  - 数据切分 `Code/Split_RNA_dataset/Version_2/`(约 2500 行)
  - 推理评估 `Code/predict_protenix/`(约 1500 行)
  - `.pt` 构建 `scripts/build_refinement_pt.py`(683 行)
  - 契约文档 `pt.md`、提案 `RNA_refinement_novelty_and_model_proposal.md`
- **审计方法**:逐文件通读 + 交叉核对提案 §6"代码与设计差距"清单;静态审读,未在服务器上实际运行数据/训练。

---

## 0. 执行摘要

| 类别 | 结论 |
|---|---|
| **Data leakage** | **未发现确定的泄漏。** 三条红线(用 native RMSD 选候选、test 与 train 同源、native 信息进特征)均被代码或契约挡住。有一个**残余泄漏风险**和两个**训练/评估分布错位**。 |
| **Bug** | **无阻断性 bug**(smoke test 覆盖 forward/训练/采样/等变/置信度开关,维度自洽)。有 2 个**潜伏 bug**(当前 config 未触发)、1 个**输出坐标系问题**、若干结构性隐患。 |
| **与提案 §6 对齐** | 报告列出的差距在本 repo **基本全部修复**;但产生了**报告未写的新差距**(最主要:QA accept/reject 未实现、结构相似度去冗余未做、refinement-specific 指标缺失)。 |

---

## 1. 结论:Data leakage 审计

### 1.1 红线核查(逐条)

| 红线 / 风险点 | 判定 | 证据 |
|---|---|---|
| **禁止用 native RMSD 选 Protenix candidate** | ✅ 未泄漏 | `summarize_foldbench_rna_rmsd.py:147-177` `select_strict_rank1` 严格按 `ranking_score` 排序;RMSD 仅出现在 `oracle_min_rmsd` 报告字段,不参与选择。README 明确 "never falls back based on RMSD" |
| **test 与 train 同源** | ✅ 已去冗余 | `split_rna_dataset_v1.py:689-710, 752-819` test 对 train+val 做 MMseqs2 `≥0.80` seq-id + `≥0.80` 双向 coverage;test 内部亦做同源聚类去冗余 |
| **native 信息进模型特征** | ✅ 未泄漏 | `.pt` 中 `pos` / `target_mask` 仅作监督目标;`pre_refinement_aligned_rmsd`、Kabsch 变换矩阵等审计字段在 `dataset.py` 中**未**拼入 node feature(pt.md 明确"不应拼入") |
| **Protenix 训练数据重叠** | ⚠️ 有风险 | 当前用 `protenix_base_default_v1.0.0`(cutoff 2021-09-30),与 `train_end=2021-09-30` 对齐,test(>2023-12-31)不在 cutoff 内 → **安全**。但若有脚本切到 `protenix_base_20250630_v1.0.0`(cutoff 2025-06-30),则 2024-2026 的 test 结构**直接泄漏**。**高风险隐患,务必只用 default 模型** |
| **train / val / test 分布错位**(非泄漏,但影响结论) | ⚠️ 重要 | 提案 §5 明确:train 被限制 RMSD≤15Å,val/test 不限,且 train 显著更短 → **error severity shift + length shift**。`select_entries`(`split_rna_dataset_v1.py:292`)仅对 `split=="train"` 应用 cutoff。模型将被要求在 test 上处理大量 wrong-fold(提案 §9 已指出"local refinement 通常无法恢复") |

### 1.2 残余泄漏风险(真实存在,未堵死)

1. **结构相似度未去冗余**。test 去冗余只按序列同一性(80% identity/coverage),**未按结构 TM-score 控制**。提案 §5 明确"这是必要但不充分"——序列不同但三级结构高度相似的情况会被放过,尤其对 length 长、拓扑简单的 RNA。当前代码未实现 RNA-align 结构去冗余。**建议补**。
2. **val 未对 train 去冗余**。`reference_ids = {train, val}`(`split_rna_dataset_v1.py:689`)仅用于 test 去冗余;val 本身未对 train 做同源控制。后果:早停/超参选择偏乐观,报告性能可能虚高。**建议 val 也对 train 去冗余,或至少在报告中给出 homology 占比。**

---

## 2. 模型框架流程(当前代码实际实现)

### 2.1 数据端到端

```
实验纯RNA PDB CIF (RCSB)
  │
  ├─ Download_PDB_RAW/pdb_cif_pipeline.py     审计1979迁移 + 下载2241当前目标
  ├─ Classify_PDB/classify_rna.py             按组成分类(纯RNA/复合物/核糖体...)
  ├─ Split_RNA_dataset/start_split_v1.sh      单链时间切分 → ~/Data_V1/{train,val,test}
  │       (train≤2021-09-30, val≤2023-12-31, test>2023-12-31)
  │       (train仅: rank-1 RMSD≤15Å; val/test 不限)
  │       (test 对 train+val 做 ≥0.80 seq-id/coverage 去冗余)
  ├─ RNA_FM_pipeline/generate_rnafm_embeddings.py  → rnafm_t12_residue_embeddings.pt [R,640]
  ├─ predict_protenix/run_foldbench_rna_rmsd.py     Protenix推理 + FoldBench式C3'RMSD评估
  └─ build_refinement_pt.py  ← 合成 .pt (schema v2)
        │
        ▼
   Data_Refinement_PT/{split}/{pdb}/seed_S/sample_K.pt
        │
        ▼
   EuclideanDataset → RNAData (batch) → BaseFlow
```

`.pt` 每个样本含:`pos_pred`(Protenix 预测)、`pos`(Kabsch 对齐后 native,仅填充 `target_mask=True`)、`target_mask`、`atomic_numbers`、`atom_name_id`、`residue_index`、`sequence`、`rnafm_embedding [R,640]`、静态 `edge_index/edge_attr[7]`、`geometry_bond_index`+`ideal_bond_length`、`clash_exclusion_index`、Protenix 置信度 `atom_plddt` / `token_pair_pae,pde` / `contact_probs`。

### 2.2 模型结构

```
BaseFlow
 ├── TorchMDDynamics (network)
 │     └── TorchMD_ET_dynamics
 │           ├── z(原子序数)→Embedding → x_initial
 │           ├── node_attr: RNA-FM(640)+atom_name one-hot(29)+residue_type one-hot(5) = 674
 │           │        + [confidence_cond时] pLDDT(1) → 网络侧=675
 │           ├── 静态边(类型0,1,2: 残基内键/O3'-P磷酸二酯/C4'-C4'序邻)
 │           ├── + merge_dynamic_radius_edges → 动态半径边(类型3)
 │           │        (每步用当前pos重建, 非固定图)
 │           ├── edge_attr: rbf(当前距离) + 7类one-hot + [confidence时] 4维PAE/PDE/contact
 │           ├── source_conditioning=True: 叠加pos_source的边距离(RBF差)与节点差
 │           └── 8层 EquivariantMultiHeadAttention → x(标量), vec(向量)
 ├── EquivariantVectorOutput → raw_velocity(纯量, 已质心化)
 └── [use_mobility_v1时] mobility_head(hidden+7) → σ门控 → gated_velocity
```

**关键点**:显式 source 条件化已实现(网络每步同时看当前 `x_t` 与原始 `pos_source`,通过边距离差 `source_rbf - current_rbf` 与节点投影 `source_node_proj`)。置信度条件化已实现(节点拼 pLDDT、边拼 PAE/PDE/contact 均值)。

### 2.3 训练流程(`generic_step`)

按 `training_objective` 分两条主线:

**A. `residual` 目标(当前 config 使用)**
- `t` 恒为 0,`x_t = x0_centered` = 质心化的 `pos_pred`(预测结构)
- `u_t = x1_centered − x0_centered`(一步指向 native)
- 网络预测 `v_t`(经 mobility 门控),`pos_estimate = x0_centered + v_t`
- 损失 = `flow_matching_L2(v_t, u_t, mask=target_mask)` + bond + clash + plane
- `use_mobility_v1` 开启时再加 4 项:mobility_gate / no_regret / protect / velocity_budget

**B. `flow` 目标(未用但存在)**
- 均匀采样 `t∈[1e-4,0.9999]`,`x_t = (1−t)·x0 + t·x1 + σ_t·eps`
- 目标 `u_t = x1 − x0 + σ̇_t·eps`;采样用欧拉 `x += Δt·v_t`

### 2.4 推理 / 采样(`sample`)

- `residual` 目标:单次前向,`return source + residual`(无 num_timesteps 积分)
- `flow` 目标:欧拉积分 N 步(`n_timesteps`;`s_churn`/`std` 为死代码)

---

## 3. Bug 清单

### 3.1 潜伏 Bug(当前 config 未触发,但真实)

**Bug-1 随机流路径 + 残留死参数**(`model.py:1242, 1218-1220`)
`sample()` 的 `flow` 分支有 `s_churn`(默认1.0)与 `std`(默认1.0),**完全未使用**——欧拉循环仅 `x += Δt·v_t`,无任何 stochastic churn。若改用 `flow_path="stochastic"` 采样,将得到确定性 ODE 结果而非 SDE,与训练目标不一致。建议:删除两参数,或实现 churn。

**Bug-2 `sigma_dot_t` 在 t=0/1 处除零**(`model.py:291-300`)
```python
return self.sigma * 0.5 * (1 - 2*t) / torch.sqrt(t*(1-t))
```
当 `t` 恰为 0 或 1,`sqrt(t(1-t))=0` → inf/NaN。当前 `sample_time` 采样 `[1e-4,0.9999]` 避开端点;`residual` 路径 t=0 但 `flow_path=deterministic` 下 `sigma_dot_t` 直接返回 0,故**未触发**。隐患:任何人把采样下界改为 0,或改 `flow_path` 即触发。建议加 `clamp_min` 保护。

### 3.2 输出坐标系问题(值得注意)

**Bug-3 残差模型输出丢失绝对帧**(`model.py:1269-1271` + `generic_step:960-971`)
`residual` 目标中 `pos_estimate = x0_centered + v_t`,且 `v_t` 被 `center_of_mass` 化(模型内部与 `apply_residue_mobility_gate:720-723`)。因此**模型输出永远是质心在原点的结构**,绝对平移量与原始 `pos_pred` 的参考帧丢失。
- 对 RMSD(经 Kabsch 对齐)、bond/clash/plane 等**平移不变**指标无影响(评估也正是如此)。
- 但若需将精修结构**存盘/回加载**到原始帧(如与 raw `pos_pred` 做逐原子位形对比,或喂给不做质心对齐的下游),**需手动补回 `pos_pred` 质心**。当前 `sample()` 未补。建议 `sample()` 最后补回质心,或文档明确"输出为质心对齐,评价一律 Kabsch"。

### 3.3 结构性隐患

**Bug-4 mobility 门控 + 质心化**(`model.py:716-723`)
门控速度 `COM(mobility_atom * raw_velocity)` 强制质心为零;训练目标 `u_t = x1_centered − x0_centered` 质心亦为零 → **一致,不构成 bug**。但由于门控是逐残基标量权重,`mobility_atom×raw_velocity` 的质心不一定是零,再强制 COM 化会**微调掉部分可允许的平移速度**。属设计选择(避免漂移),但若模型学不出"整体平移精修",根因在此。

**Bug-5 数据增强占比偏高**(`RNA_test.yaml:57-58` + `model.py:544-597`)
`identity_pair_probability=0.1` + `near_native_pair_probability=0.2` = **30%** 训练样本被增强成"无修正/近 native"。这直接稀释主干 `pred→native` 修正信号。提案 §4.6 要求纳入 identity pair(正确),但 0.3 占比偏大,可能使模型在大误差样本上训练不足。**建议消融 0.1/0.05 等更小值,并观察 `flow_matching_loss` 收敛。**

**Bug-6 `CosineAnnealingWarmRestarts` gamma=1.0**(`base.py:120-139`)
`gamma=1.0` 表示每个 cycle 的 max_lr 不衰减(通常取 0.7-0.9 渐减)。属超参问题,建议确认。

### 3.4 次要 / 工程问题

- **重复死代码**:`etflow/schedulers/torchmd_net/model_dynamics.py`(620 行)是一份放错目录的旧版副本,真正被 import 的是 `networks/torchmd_net/model_dynamics.py`(760 行)。此副本不会被 import,建议删除,避免混淆。
- **`testDataloader.py:7` 硬编码旧路径** `DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"`,与当前 `~/Data_V1` 布局不符,易误导。
- **`dataset.py:117-127` `clash_exclusion_index` fallback**:旧数据无 1-3 排除对时退回 `geometry_bond_index`(仅含 1-2 键)。会把共价键对当成 clash 排除对,削弱 clash loss 处罚,但不造成泄漏。新 `.pt` 已生成 1-3 对,不受影响。
- **`smoke_test_synthetic.py:301-306` 测试覆盖缺口**:仅测 `atom_plddt[0]=1.5`(超上限)触发报错,未测有效边界 `max=1.0` 与 `min<0`。

---

## 4. 与提案报告对齐检查

对照提案 §6 逐条核对当前代码。**标记 ✅ 表示已按报告修复/实现;⚠️ 表示部分;❌ 表示未实现。**

| 提案条目 | 判定 | 现状 |
|---|---|---|
| §6.1 dataset 强制读 `sequence/edge_attr/node_attr` | ✅ 修复 | `dataset.py:78-88` 正确处理 |
| §6.1 逗号使 `sequence` 变 tuple | ✅ 修复 | `dataset.py:78` 无逗号 |
| §6.1 config `node_attr_dim:0, edge_attr_dim:0` 忽略 RNA-FM/edge type | ⚠️ 修复 | config 现为 `674/7` 正确;`confidence_conditioning:true` 时网络侧 675/11(pt.md 指出 YAML 应保持 674/7,正确) |
| §6.1 `.pt` builder 半径图未存 residue index/atom name/covalent type | ✅ 修复 | `build_refinement_pt.py` 存 `residue_index`/`atom_name_id`/`geometry_bond_index`+`ideal_bond_length`+`clash_exclusion_index` |
| §6.2 `batched_data[" atomic_numbers"]` 多空格 | ✅ 修复 | `model.py:895` 用 `"atomic_numbers"` |
| §6.2 `evaluate.py` `sample(edge_index=...)` 参数名错误 | ✅ 移除 | `evaluate.py` 在本 repo 不存在 |
| §6.2 `evaluate.py` 给 `BaseDataModule` 传 `sample_files` | ✅ 移除 | 同上 |
| §6.3 edge 只按初始 `pos_pred` 构图并固定 | ✅ 修复 | `model.py:812` 每步按当前 `pos` 重建动态半径图 |
| §6.3 无独立保留 covalent backbone 的通道 | ✅ 修复 | `merge_dynamic_radius_edges` 始终 `concat([bond_index, dynamic])`,静态共价边(类型0,1,2)永久保留 |
| §6.3 `edge_weight=sum(edge_vec**2)` 平方距离 | ✅ 修复 | `model_dynamics.py:458` 用 `vector_norm`;`edge_vec/edge_weight` 为单位向量(508行) |
| §6.4 无 pLDDT/PAE/PDE | ✅ 实现 | config `confidence_conditioning:true`;`.pt` 存 `atom_plddt`/`token_pair_pae,pde`/`contact_probs`,dataset 拼入节点(1维)+边(4维) |
| §6.4 RNA-FM 被 config 关闭 | ✅ 开启 | `node_attr_dim:674`,RNA-FM 640 维进节点 |
| §6.4 无 base pair/stacking、torsion/pucker/suite/clash loss | ⚠️ 部分 | 有 bond/clash/base-plane loss(`loss.py`);**无 base-pair/stacking 图、无 torsion/pucker 层**(属提案 Phase 2/3 规划,预期未实现) |
| §6.4 仅 flow vector MSE;eval 仅 all-atom Kabsch RMSD | ⚠️ 部分 | 训练有几何 loss;评估有 C3'RMSD + lDDT + TM + GDT + DockQ(FoldBench 全套);**无 ΔTM/ΔlDDT/INF/stacking 等 refinement-specific 指标**(提案 §7.3 主表要求) |

### 4.1 报告未写的新差距

1. **QA accept / reject 未实现**(差距最大)。提案 §4.6 / Phase 1 step 5 的 `X_final = refined if ΔQ>τ else pred`(no-regret 输出选择)**在代码中不存在**;`no_regret_loss` 仅为训练项。推理端无 accept/reject。
2. **mobility gate 的"误差定位"监督为自指**。提案 §4.3 要求门控学习"Protenix confidence 在 RNA 上是否可信",用真实局部误差(lDDT error / frame error)监督 `m_i`。而当前 `mobility_target` 用 `raw_residue_error`(`model.py:1099-1102`),这是模型自身预测与 native 的误差,非"Protenix confidence 的误差"。不对 native 泄漏(仅作监督),但门控学到的是"我对当前结构有多错",而非"pLDDT 有多可信",**与报告机制略有偏离**。
3. **动态边 confidence 轻微不一致**。`build_confidence_edge_attr`(`model.py:832-851`)对所有边(静态+动态)取 token-pair confidence;动态边按当前坐标随时间重建,而 confidence 来自预测时 pair 矩阵,不随坐标更新。影响小,但严格说动态边应重算或标记。
4. **审计字段已存但未利用**。`native_to_prediction_rotation/translation`、`pre_refinement_aligned_rmsd` 已写入 `.pt`,未用于"per-sample 难度分层"或"refinement 前后对比"。建议利用。

---

## 5. 建议优先级

**P0(必须做,否则影响结论可信度)**
1. 补上 **QA accept/reject**(提案 Phase 1 step 5)。当前最大"报告承诺但代码未做"点。
2. **test 结构相似度去冗余**(RNA-align TM-score)或至少在报告中做 hard/easy 分层。提案 §7.3 要求按 `max-train-structure-similarity`、initial RMSD、length 分层,当前均未做。
3. **锁定 `protenix_base_default_v1.0.0`**,加断言/文档:禁止切到 2025-06-30 模型生成 test 预测(否则 test 直接泄漏)。

**P1(强烈建议,收敛结论)**
4. val 对 train 去冗余,或报告 val 与 train 的 homology 占比。
5. 补 refinement-specific 指标:ΔTM/ΔlDDT/ΔINF/stacking、improvement rate、good-input damage rate(raw RMSD<3Å 后变差>0.5Å 比例)、paired Wilcoxon。
6. 对 `identity_pair_probability`/`near_native_pair_probability` 做消融(30% 占比偏高)。

**P2(建议,不紧急)**
7. 删除 `etflow/schedulers/torchmd_net/model_dynamics.py` 死代码;清理 `testDataloader.py` 脏路径。
8. `sample()` 补回质心,或文档写明"输出为质心对齐,评价一律 Kabsch"。
9. `sigma_dot_t` 加 `clamp_min` 保护;移除或实现 `s_churn`/`std`。

---

## 6. 审计限制与建议

1. 本报告为**静态审读**;本机无数据、conda 环境、GPU,未实际运行训练或采样。结构正确性结论基于 `smoke_test_synthetic.py`(验证 forward/梯度/采样/等变/置信度开关)与逐行读码交叉。
2. "无泄漏"结论依赖三点,均为**代码+报告推断**,实际产物在服务器:
   - pipeline 确实只用 `protenix_base_default_v1.0.0`;
   - 切分日期与模型 cutoff 对齐;
   - `rank1_targets.csv` 确实按 `ranking_score` 排序。
   **建议在服务器上 grep 确认实际模型名与实际排序,再行拍板。**
3. 若需实际运行 smoke test 或 `.pt` 单样本构建,需提供服务器环境或至少一个 `.pt` + 对应 CIF/JSON。

---

*报告结束*
