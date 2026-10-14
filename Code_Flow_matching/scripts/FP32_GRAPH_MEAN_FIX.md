# CUDA BF16 图均值错误修复

## 服务器已经确认的事实

用户于 2026-10-13 在 Torch 2.7.1+cu126、服务器安装的 PyG、CUDA 上运行原检查脚本。
FP32 下计数正确；BF16 下原子数 576、2048、4096、7984、10629、20000 的计数均停在 256。
7984 / 10629 / 20000 个原子的坐标梯度范数分别达到正常理论上界的 14.93 / 19.99 / 38.13 倍。
这是实际复现的数值错误；其对已训练模型的最终精度贡献仍须通过复评确认。

BF16 在 256 附近的相邻数是 256、258，逐次加 1 可能舍入回 256。
原归约的求和与计数都使用输入 dtype，两者均有精度风险。
`mean_ones=1` 不能证明均值正确：分子分母可以同时出错。
修复必须覆盖求和、计数以及去质心的反向广播归约，不能只修分母。

## 修改内容

- 新增共享 `etflow/commons/reductions.py`：FP16/BF16 提升为 FP32 后求和，整数计数；FP64 保留。
- 网络原始速度及 gate 后速度的去质心均使用该路径，结果保持 FP32，避免反向归约再次落入 BF16。
- 残基 gate 输入的均值、均值检查、键误差均值及可选输出均值统一使用该路径。
- 坐标损失在至少 FP32 中计算，掩码计数使用整数。
- 参数名称、形状未修改；同一份旧 checkpoint 可加载。各项 loss 权重未调整。
- 检查脚本区分 `backend_*` 与项目的 `production_*`，并新增 `--assert-fixed`。

修复项目调用后，原始 backend 的 BF16 `backend_count` 在服务器上仍可能显示 256；
它是保留的故障对照。通过条件看 `production_dtype=torch.float32`、
`production_constant_center_error` 接近零、`gate_RMS_ratio≤1`、
`coordinate_gradient_bound_ratio≤1`，以及最后的 `FP32_GRAPH_MEAN_CHECK_PASSED`。

## 上传与执行

将 `rna_fp32_graph_mean_fix_20261013_v2.tar.gz` 上传到服务器 `$HOME/`。v2 已补齐完整模型检查文件 `smoke_test_synthetic.py`。在原 conda 环境中执行：

```bash
cd "$HOME/Code_Flow_matching"
tar -czf "$HOME/rna_before_fp32_mean_$(date +%Y%m%d_%H%M%S).tar.gz" etflow scripts
tar -xzf "$HOME/rna_fp32_graph_mean_fix_20261013_v2.tar.gz" -C "$HOME/Code_Flow_matching"
CUDA_VISIBLE_DEVICES=3 bash scripts/run_fp32_graph_mean_checks.sh
```

脚本依次执行实际 CUDA 数值检查、CPU/CUDA 回归、完整模型 BF16 合成 smoke test、
同一 epoch 49 权重的 5 个 PDB 探针（BF16、FP32 各一次）。无优化器更新。
结果写入新目录 `evaluation/rank1_fp32mean_fix_<时间>/`；下载整个目录继续比较。

如果已完成数值检查和 18 项回归，只因原包缺少 `smoke_test_synthetic.py` 停住，
解压 v2 后可从完整模型检查继续，写入新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=3 START_AT=smoke bash scripts/run_fp32_graph_mean_checks.sh
```

此模式仍校验上传文件、记录配置和权重摘要，然后执行模型检查及两种精度探针。

需要一并重新评估全部验证集和测试集时，第三条命令改为：

```bash
CUDA_VISIBLE_DEVICES=3 FULL_EVAL=1 bash scripts/run_fp32_graph_mean_checks.sh
```

使用 `~/Data_PT_V2` 现有 rank1 数据，测试集结果用于诊断，不据此选择 checkpoint 或调参。
需要更新三态 PyMOL bundle 时，在检查通过后使用现有 `scripts/run_rank1_pymol_diagnostics.sh`，
其输出目录默认带新时间戳，加载的代码会使用修复后的均值。

## 如何理解复评

1. 先确认数值检查通过。随后 BF16 和 FP32 探针的 gate/运动量/梯度应更可比，仍允许网络其他 BF16 运算带来差异。
2. 将修复后旧权重的结果与原 probe 和原 PyMOL bundle 比较，判断推理路径错误造成多少变化。
3. 旧 checkpoint 已经过原计算路径训练；修复推理不能消除训练历史。若长链仍小幅移动，不能据此排除该错误对训练的影响。检验训练影响需要受控的重新训练。
4. 先重新计算梯度，再判断 no-regret 等损失的冲突；原有大图梯度分析可能混入数值异常。

错误均值在反向传播中的常量模式系数可包含 `1-N/256`，而正确去质心应消除常量模式。
这为随长度增加的训练异常提供具体机制，但“模型学会压低长链输出”目前仍是待检验的推断。

单独减去同一个向量不会改变原子间距离，因此错误均值本身不能直接解释全部断键。
不同残基 gate、错误训练梯度和独立原子位移会影响结构，需在修复后继续检查局部几何。

本地只有 CPU Torch，已通过大型混合图的前向/反向、FP64 gradcheck、gate 收缩、坐标梯度上界检查。
实际 CUDA 和完整依赖栈由上述服务器命令验证。
