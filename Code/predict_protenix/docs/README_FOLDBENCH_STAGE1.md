# Protenix RNA 第一阶段：FoldBench-style prep + pred

## 固定实验口径

- 目标：`pdb_cif_manifest.csv` 中 `CURRENT_TARGET=True` 的 2241 个 PDB；
- 排除：`3OK2`、`3OK4`、`5EME`、`176D`、`5EMF`；
- 模型：`protenix_base_default_v1.0.0`；
- Protenix：1.0.5；
- 推理：seed `42,66,101,2024,8888`，每 seed 5 samples，200 steps，10 cycles；
- 特征：RNA MSA 开，template 关，BF16，cache/fusion/TF32 开；
- 总产物：2241 x 5 x 5 = 56,025 个主预测 CIF；
- prep 目录：`~/Json_data/Complex_json`；
- 新预测目录：`~/Json_data/Foldbench_predictions`，不读取或覆盖旧 4x50 输出；
- 报告目录：`~/Code/pipeline_reports/FOLDBENCH_STAGE1`。

四个 prep 数据库和 checkpoint 均通过固定绝对路径使用。运行脚本会设置：

```text
PROTENIX_ROOT_DIR=/storage9920/home/tinghao.xia/protenix_data
```

## 上传本地脚本

在本地项目根目录执行：

```powershell
scp -P 30063 Code/predict_protenix/stage2_decoy_pipeline.py `
  Code/predict_protenix/cached_rna_prep.py `
  Code/predict_protenix/resident_protenix_pred.py `
  Code/predict_protenix/run_foldbench_pred.sh `
  Code/predict_protenix/start_foldbench_pred.sh `
  Code/predict_protenix/monitor_foldbench_pred.sh `
  Code/predict_protenix/check_pred_environment.sh `
  Code/predict_protenix/start_foldbench_prep.sh `
  Code/predict_protenix/monitor_foldbench_prep.sh `
  Code/predict_protenix/run_foldbench_stage1.sh `
  tinghao.xia@dubhe.lglab.ac.cn:~/Code/predict_protenix/
```

## 运行顺序

服务器宿主机先准备报告目录：

```bash
mkdir -p ~/Code/pipeline_reports/FOLDBENCH_STAGE1
chmod +x ~/Code/predict_protenix/run_foldbench_stage1.sh
```

### 1. 预检

```bash
docker exec protenix_test bash \
  /storage9920/home/tinghao.xia/Code/predict_protenix/run_foldbench_stage1.sh \
  preflight
```

成功标准：命令退出码为 0，并且：

```bash
cat ~/Code/pipeline_reports/FOLDBENCH_STAGE1/preflight.exit_code
```

输出 `0`。

### 2. 按旧数据库口径增量补 prep

新版 prep 按唯一 RNA 序列解析 MSA。为了与既有自定义数据库结果保持一致，
默认优先级为：已有有效 MSA、一次新的 nhmmer 搜索；官方包只有显式传入
`--allow-official-msa` 才会启用。每个 PDB 仍会生成独立的
`*-final-updated.json`，并尽可能在 `prep_output_<pdb>` 下建立指向共享 A3M 的
兼容符号链接。不会删除或覆盖任何已通过审计的完整 prep。

已有完整 prep 会被审计后跳过，不会删除或覆盖。中断后重复同一命令即可；
已经进入序列缓存的搜索不会重跑。

使用宿主机启动与监控脚本。每次运行写入独立时间戳目录，并每 30 秒更新
heartbeat；即使容器停止，宿主机 launcher 也会记录 `docker exec` 的退出码：

```bash
chmod +x ~/Code/predict_protenix/{run_foldbench_stage1,start_foldbench_prep,monitor_foldbench_prep}.sh
bash ~/Code/predict_protenix/start_foldbench_prep.sh
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_prep.sh
```

监控输出会同时显示宿主机 launcher、容器状态、prep/nhmmer 进程、heartbeat
新鲜度和最新 30 行日志。单次运行的全部日志位于：

```text
~/Code/pipeline_reports/FOLDBENCH_STAGE1/prep_runs/<UTC时间戳>/
```

其中 `launcher.exit_code=RUNNING` 表示仍在运行，`0` 表示正常完成，非零表示
异常或任务失败；`heartbeat.json` 超过 120 秒未更新会显示 `STALE`。重复启动
只会补失败或不完整项。完成后再运行一次 `audit`，确认
`prep_complete=2241`、`need_json=0`、`need_prep=0`。

### 3. 八卡运行 FoldBench-style pred

先执行只读资源/API 检查：

```bash
bash ~/Code/predict_protenix/check_pred_environment.sh
```

再用 GPU 1 对当前最短且 prep 完成的 PDB 做完整 5 seeds x 5 samples 冒烟；
产生的 25 个 CIF 会计入正式结果：

```bash
bash ~/Code/predict_protenix/start_foldbench_pred.sh pred-smoke
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_pred.sh
```

冒烟完成且 `LAUNCHER_EXIT=0`、`PRED_EXIT=0` 后启动正式八卡增量任务：

```bash
bash ~/Code/predict_protenix/start_foldbench_pred.sh pred
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_pred.sh
```

每张 GPU 启动一个常驻 worker，checkpoint 只加载一次，然后连续运行分配给该
GPU 的 PDB 和 seeds。中断后重复启动即可断点续跑；只有通过 CIF 数量和基本
可读性校验的 seed 才会跳过。尚未完成 prep 的条目写入每次运行目录下的
`pred_deferred_prep.csv`，待 prep 完成后再次启动即可补算。

```bash
tail -f ~/Code/pipeline_reports/FOLDBENCH_STAGE1/pred_console.log
nvidia-smi
cat ~/Code/pipeline_reports/FOLDBENCH_STAGE1/pred.exit_code
cat ~/Code/pipeline_reports/FOLDBENCH_STAGE1/summary.json
```

最终成功标准：

```text
target_count=2241
raw_json_valid=2241
prep_complete=2241
all_seeds_complete=2241
overall_complete=2241
valid_decoy_count=56025
expected_decoy_count=56025
all_complete=true
pred.exit_code=0
```

## 日志和断点规则

- 每次 Protenix 子进程的完整命令、GPU、时间和退出码写入
  `run_events.jsonl`；
- 每个 prep 和 seed 都有独立日志；
- `*_wounresol.cif` 不计入 56,025 个主 CIF；
- 不完整 seed 会整 seed 重跑，但不会删除旧输出；
- 旧的 `~/Json_data/Complex_json/pred_output_*` 不参与本实验审计。
