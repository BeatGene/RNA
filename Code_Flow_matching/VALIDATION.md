# 验证记录（2026-11-18）

本机CPU，Python3.9，临时隔离依赖：torch2.8.0、torch-geometric2.6.1、torch-cluster1.6.3、e3nn0.5.1、lightning2.6.0。依赖未写入项目或用户全局环境。

`tests/test_modular_architecture.py`：12项测试通过：

1. 两种表征、五种合法输出头组合的前向和有限梯度反向传播，包含flow+mobility。
2. identity增强及空选择保护项。
3. 基础loss的有效原子mask和按图归约语义。
4. 关闭置信度后输出不读取置信度字段，缺失/required模式检查。
5. 拒绝跨图边及验证不同图的token-pair偏移。
6. 原始PT→数据检查→674维原子特征→PyG batch→训练loss；实际Lightning执行一个训练与验证step，包括优化器及梯度裁剪。
7. 默认lmax2的旋转等变性、无噪声采样逐位重复一致、恢复原训练状态。
8. lmax1的旋转/平移等变性与固定source敏感性。
9. deterministic/stochastic训练路径及residual模式。
10. Euler/Heun/SDE调用次数、固定source和非法步数拒绝。
11. 静态图不提供基础边属性时，仅融合成对置信度的前向，以及一维时间输入。
12. 整张图无有效目标原子时mobility/验证指标仍保持有限（batch其他图有效）。

原配置RNA_test、RNA_train_flow_deterministic_v1、RNA_train_residual_v1、RNA_train_residual_v3和新增RNA_train_modular均成功构造。新示例模型参数量6,165,508。

源码编译检查及git diff --check通过。ABRefine原算子有CUDA autocast弃用提示；无logger的Lightning测试有日志器提示，不影响测试结果。

尚未进行真实数据训练、服务器环境验证、GPU混合精度和多卡DDP，未比较真实RNA的精修RMSD。新增参数量及算子不同，训练速度/显存和精修质量需要服务器实测。
