# 修复后的服务器验证与测速

本地已通过 14 项生成器回归、3 项 RMS/梯度回归，以及 2 项输出头 CPU BF16/FP64 回归。
CUDA 输出头回归在本机跳过，需要服务器运行。
服务器已通过此前版本的 FP32 冒烟测试及非零条件位移等变性检查；本次修复其暴露的 BF16 索引赋值冲突，CUDA/BF16/DDP 需要重跑。
上传更新后的 Code_Flow_matching（尤其下列文件）。无需等待 Protenix 全量采样完成。

- scripts/build_refinement_pt.py
- scripts/test_build_refinement_pt.py
- scripts/test_masked_rms.py
- scripts/test_vector_output_amp.py
- scripts/smoke_test_synthetic.py
- scripts/check_training_runtime.py
- etflow/models/model.py
- etflow/networks/torchmd_net/utils.py

config/RNA_test.yaml 中用户已设置的 clip_during_norm: false 保留；本次没有切换训练目标。

## 1. 基础回归和单卡合成数据测试

选择一张当时确认空闲的 GPU。下面以物理 GPU 0 为例；不是根据旧 nvidia-smi 截图认定它现在空闲。

```bash
conda activate protenix-1.0.5
cd ~/Code_Flow_matching
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/test_build_refinement_pt.py -v
python scripts/test_masked_rms.py -v
CUDA_VISIBLE_DEVICES=0 python scripts/test_vector_output_amp.py -v
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_synthetic.py --device cuda --bf16
CUDA_VISIBLE_DEVICES=0 python scripts/check_training_runtime.py --config config/RNA_test.yaml --devices 1
```

smoke_test_synthetic 使用小网络检查不同训练模式、缺失标签、非零 source 位移、任意轴旋转和共同平移；--bf16 额外检查 CUDA autocast 的前向与反向。
check_training_runtime 使用 YAML 中实际的 128 通道/6 层网络、optimizer、梯度累积和 BF16 设置，跑有界的训练和验证。
默认使用自动生成、退出后清理的合成 pt，不读取真实数据，不保存 checkpoint，不创建 WandB run。
如有依赖错误，请保留具体报错和 torch/CUDA 版本，再选择匹配的 PyG/torch_cluster 安装包。

## 2. 八卡运行检查（八卡可用时）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/check_training_runtime.py \
  --config config/RNA_test.yaml --devices 8
```

由 Lightning 启动 DDP，不要再在外面套一次 torchrun。
默认每个进程执行 12 个训练 microbatch 和 2 个验证 batch，训练前 3 个不计入测速。
预期最后出现 BOUNDED TRAINING RUNTIME CHECK PASSED。
合成数据只有 2/3 个简化残基，适合检查运行链路，不能用于推断真实 RNA 的耗时、显存上限或训练效果。

## 3. 真实数据生成后的短程测速

可以先生成一部分长度具有代表性的 train/val pt，避免只选最短的 RNA。
新 generator_version 是 2.2-strict-complete-sequence-mapping，schema 仍为 2。
建议新输出目录，例如 ~/Data_Refinement_PT_v2_2；默认续跑拒绝旧版文件，不会自动覆盖。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/check_training_runtime.py \
  --config config/RNA_test.yaml --devices 8 \
  --data-dir ~/Data_Refinement_PT_v2_2 \
  --train-batches 100 --warmup-batches 10 --val-batches 20
```

此命令不保存权重，用新建模型跑一次短训练，以测量当前配置、DDP、数据加载和几何 loss 的总开销。
需要至少 1600 个 train pt、320 个 val pt（当前每卡 batch_size=2）。不需要 test pt。
数据不够可降低 batches，但应保持 train-batches > warmup-batches，并最好使 train-batches 是梯度累积数 3 的倍数。
RUNTIME_RESULT 中 train/val.seconds_per_microbatch 是完整分布式 batch 的平均时间，不是单样本时间，也不是三次梯度累积后的 optimizer step 时间。
测量含每批 GPU 同步，排除训练开头 warmup，不包括完整模型评估、checkpoint 和 WandB 开销。

## 4. 10 epochs 时长公式

当前数据量：train=774×200=154800，val=117×200=23400，test=97×200=19400。
总数 197600 正确；test 不参与常规训练/每轮验证。
当前每卡 batch_size=2、8 GPU：

- 每轮 train microbatch：154800/(8×2)=9675。
- 每轮 val batch：ceil(23400/(8×2))=1463（DDP 补齐导致末批细节差异）。
- 梯度累积 3：每轮 optimizer updates=3225，10 轮为 32250。

设实测 train 秒/批为 t_train，val 秒/批为 t_val：

    10 epochs 小时数 ≈ 10 × (9675 × t_train + 1463 × t_val) / 3600

| 假设 t_train / t_val | 10 epochs 耗时，不含额外开销 |
|---|---|
| 0.5 s / 0.2 s | 14.3 小时 |
| 1 s / 0.4 s | 28.5 小时 |
| 2 s / 0.8 s | 57.0 小时 |
| 5 s / 2 s | 142.5 小时 |

这些是条件算例，不是 H20 实测预测。RNA 原子/残基数、图边数、CPU 几何 loss 循环与数据存储速度都会改变时间。Early stopping 也可能减少 epoch 数。

## 5. 切换 flow 模式

现有代码支持通过 YAML 切换；目前 mobility_v1 只支持 residual，所以必须一起调整：

```yaml
training_objective: flow
flow_path: deterministic
use_mobility_v1: false
```

若用随机插值路径，将 flow_path 改为 stochastic 并保持 sigma > 0。
真正的多步 flow 推理还需要调用 sample 时传入大于 1 的 n_timesteps；train.py 本身不运行 YAML eval_args。
同一训练 batch 通常只采一个 t 并前向一次，不会因为推理用 50 步，就把训练开销直接乘 50。
数据生成已改为保守的完整序列精确匹配；不接受序列错配、gap 对齐或观察序列代替完整序列。
完整序列相同但坐标缺失的原子/残基仍通过 target_mask 支持。被拒绝样本进入 generation_issues.tsv。
