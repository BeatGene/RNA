# 本次重构文件清单

修改现有5个文件：

- `etflow/data/dataset.py`
- `etflow/data/datamodule.py`
- `etflow/models/model.py`
- `etflow/models/loss.py`
- `etflow/networks/torchmd_net/model_dynamics.py`

新增运行模块（不能直接删除，否则对应import或后端不可用）：

- `etflow/data/confidence.py`
- `etflow/data/features.py`
- `etflow/data/validation.py`
- `etflow/models/conditioning/__init__.py`
- `etflow/models/conditioning/confidence.py`
- `etflow/models/conditioning/inputs.py`
- `etflow/models/conditioning/mobility.py`
- `etflow/models/conditioning/network_features.py`
- `etflow/models/construction.py`
- `etflow/models/losses/__init__.py`
- `etflow/models/losses/bond.py`
- `etflow/models/losses/clash.py`
- `etflow/models/losses/composition.py`
- `etflow/models/losses/mobility.py`
- `etflow/models/losses/plane.py`
- `etflow/models/losses/regression.py`
- `etflow/models/losses/safeguards.py`
- `etflow/models/sampling/__init__.py`
- `etflow/models/sampling/evaluation.py`
- `etflow/models/sampling/integrators.py`
- `etflow/models/sampling/solvers.py`
- `etflow/models/training/__init__.py`
- `etflow/models/training/augmentation.py`
- `etflow/models/training/metrics.py`
- `etflow/models/training/paths.py`
- `etflow/models/training/step.py`
- `etflow/networks/factory.py`
- `etflow/networks/heads/__init__.py`
- `etflow/networks/heads/abrefine.py`
- `etflow/networks/heads/linear.py`
- `etflow/networks/representations/PROVENANCE.md`
- `etflow/networks/representations/__init__.py`
- `etflow/networks/representations/abrefine.py`
- `etflow/networks/representations/abrefine_core.py`
- `etflow/networks/representations/abrefine_ops/Jd.pt`
- `etflow/networks/representations/abrefine_ops/__init__.py`
- `etflow/networks/representations/abrefine_ops/activation.py`
- `etflow/networks/representations/abrefine_ops/drop.py`
- `etflow/networks/representations/abrefine_ops/edge_rot_mat.py`
- `etflow/networks/representations/abrefine_ops/gaussian_rbf.py`
- `etflow/networks/representations/abrefine_ops/input_block.py`
- `etflow/networks/representations/abrefine_ops/layer_norm.py`
- `etflow/networks/representations/abrefine_ops/module_list.py`
- `etflow/networks/representations/abrefine_ops/radial_function.py`
- `etflow/networks/representations/abrefine_ops/so2_ops.py`
- `etflow/networks/representations/abrefine_ops/so3.py`
- `etflow/networks/representations/abrefine_ops/transformer_block.py`
- `etflow/networks/representations/abrefine_ops/wigner.py`

新增配置/测试（不参与模型前向；删除将失去示例、说明或验证入口，建议保留）：

- `config/MODEL_FIELDS.yaml`
- `config/RNA_train_modular.yaml`
- `tests/test_modular_architecture.py`

新增交付文档与包（核验/上传后可删除，不影响模型）：

- `ARCHITECTURE.md`：ABCDE接口、目录职责、公式、切换方法和运行命令。
- `VALIDATION.md`：测试范围、版本及尚未验证项。
- `REFACTOR_FILES.md`：本清单。
- `BUNDLE_SHA256.txt`：压缩包内逐文件sha256核验清单。
- `etflow_modular_bundle.tar.gz`：可上传的完整源码包。

`etflow/networks/representations/PROVENANCE.md`记录ABRefine本地来源和修改，仅文档，可删除。包内原有未修改模块、原配置和train.py/utils.py为运行依赖或兼容入口。工作区原有的审计脚本、删除的缓存和其他用户修改不属于本次交付。
