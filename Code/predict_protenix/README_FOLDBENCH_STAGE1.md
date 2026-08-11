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

### 2. 增量补 prep

当前预计从 1974/2241 补到 2241/2241，即执行约 267 个目标。已有完整 prep
会被审计后跳过，不会删除或覆盖。

```bash
docker exec -d protenix_test bash -lc \
  '/storage9920/home/tinghao.xia/Code/predict_protenix/run_foldbench_stage1.sh prep \
  > /storage9920/home/tinghao.xia/Code/pipeline_reports/FOLDBENCH_STAGE1/prep_console.log 2>&1'
```

查看进度：

```bash
tail -f ~/Code/pipeline_reports/FOLDBENCH_STAGE1/prep_console.log
cat ~/Code/pipeline_reports/FOLDBENCH_STAGE1/prep.exit_code
cat ~/Code/pipeline_reports/FOLDBENCH_STAGE1/summary.json
```

开始 pred 前必须满足 `prep_complete=2241`、`need_json=0`、`need_prep=0`，且
`prep.exit_code` 为 `0`。prep 命令退出码为 2 通常表示至少一个目标失败；重复
同一命令只会补失败或不完整项。

### 3. 八卡运行 FoldBench-style pred

```bash
docker exec -d protenix_test bash -lc \
  '/storage9920/home/tinghao.xia/Code/predict_protenix/run_foldbench_stage1.sh pred \
  > /storage9920/home/tinghao.xia/Code/pipeline_reports/FOLDBENCH_STAGE1/pred_console.log 2>&1'
```

每张 GPU 同时只运行一个 `(PDB, seed)` 任务，每个任务生成 5 个样本。中断后
重复同一命令即可断点续跑；只有通过 CIF 数量和基本可读性校验的 seed 才会
跳过。

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
