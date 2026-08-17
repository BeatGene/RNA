# Protenix 1.0.5 持久化 Conda 运行说明

本管线直接使用：

```text
/storage9920/home/tinghao.xia/miniconda3/envs/protenix-1.0.5
```

后台启动器会 source `protenix_env.sh`，无需依赖交互式终端中的
`conda activate`，也不再依赖 `protenix_test` Docker-in-Docker 容器。

默认资源策略：

- prep：4 workers × 8 nhmmer CPUs，使用自定义四数据库，官方 RNA MSA
  包保持关闭；
- pred：GPU 0–3，5 seeds × 5 samples，200 diffusion steps，10 cycles，
  BF16，短目标优先；
- `pred_oom_quarantine.txt` 中的已知 OOM 目标不进入普通队列；
- 启动时 cgroup 内存必须低于 70%；prep 在 75% 暂停启动新搜索；pred
  worker 在每个 PDB 前达到 80% 时安全退出；
- prep 会预加载数据库集合对应的序列缓存，并在 PDB 所需 MSA 齐全后
  立即物化 final-updated JSON 和兼容 prep 目录。

主要命令：

```bash
bash ~/Code/predict_protenix/check_pred_environment.sh
bash ~/Code/predict_protenix/run_foldbench_stage1.sh preflight
bash ~/Code/predict_protenix/run_foldbench_stage1.sh audit

PREP_WORKERS=4 NHMMER_CPUS=8 \
  bash ~/Code/predict_protenix/start_foldbench_prep.sh

SMOKE_GPU=0 \
  bash ~/Code/predict_protenix/start_foldbench_pred.sh pred-smoke

PRED_GPUS=0,1,2,3 \
  bash ~/Code/predict_protenix/start_foldbench_pred.sh pred
```

监控：

```bash
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_prep.sh
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_pred.sh
```

重复启动同一类任务会被 `flock` 拒绝。外层容器重建会终止进程，但环境、
缓存、输出和日志都位于 `/storage9920`；重新运行审计与启动命令即可增量续跑。
