# Second RNA refinement training run

Run every command below on the laboratory server from:

```bash
cd /storage9920/home/tinghao.xia/Code_Flow_matching
```

## 1. Preflight

These commands do not start a persistent experiment or connect to WandB:

```bash
python -m compileall -q etflow scripts
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_synthetic.py --device cuda --bf16
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config config/RNA_train_residual_v1.yaml \
  --debug --no_logger
```

Do not start the formal run unless all three commands exit with status 0.

## 2. Measure batch size on the six selected GPUs

`batch_size` is per GPU. The following three combinations all preserve the
first run's effective global batch size of 48 graphs, so the learning rate and
the approximately 3,034 optimizer steps per epoch remain comparable:

| Per-GPU batch | Gradient accumulation | Effective global batch |
|---:|---:|---:|
| 2 | 4 | 48 |
| 4 | 2 | 48 |
| 8 | 1 | 48 |

Run the bounded real-data checks separately:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/check_training_runtime.py \
  --config config/RNA_train_residual_v1.yaml \
  --devices 6 --data-dir /storage9920/home/tinghao.xia/Data_PT_V1 \
  --batch-size 2 --accumulate-grad-batches 4 --num-workers 2 \
  --train-batches 24 --warmup-batches 4 --val-batches 4 --largest-files

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/check_training_runtime.py \
  --config config/RNA_train_residual_v1.yaml \
  --devices 6 --data-dir /storage9920/home/tinghao.xia/Data_PT_V1 \
  --batch-size 4 --accumulate-grad-batches 2 --num-workers 2 \
  --train-batches 24 --warmup-batches 4 --val-batches 4 --largest-files

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/check_training_runtime.py \
  --config config/RNA_train_residual_v1.yaml \
  --devices 6 --data-dir /storage9920/home/tinghao.xia/Data_PT_V1 \
  --batch-size 8 --accumulate-grad-batches 1 --num-workers 2 \
  --train-batches 24 --warmup-batches 4 --val-batches 4 --largest-files
```

`--largest-files` sorts by PT file size as a conservative proxy for graph size;
the scan may take a while on shared storage. Each successful command prints one
`RUNTIME_RESULT` JSON object containing throughput and per-GPU peak
allocated/reserved memory. Choose the fastest
configuration that has no OOM and keeps `max_reserved_fraction` at or below
about 0.80. RNA graph sizes vary, so do not select a setting that only fits
with negligible headroom. Then copy its batch size and accumulation value into
`config/RNA_train_residual_v1.yaml`.

If GPUs wait on data while memory usage is low, repeat the chosen batch setting
with `--num-workers 4`, then 8. Keep the worker count that increases global
samples/second without making the shared storage unstable, and copy it to the
YAML.

## 3. Start a fresh formal run

For the second experiment, leave both `pretrained_ckpt` and `ckpt_path` as
`null`. The V2 task name and checkpoint directory prevent mixing with the first
run.

```bash
mkdir -p training_logs
run_stamp=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u scripts/train.py \
  --config config/RNA_train_residual_v1.yaml \
  > "training_logs/train_residual_v2_6gpu_${run_stamp}.log" 2>&1 &
train_pid=$!
echo "PID=${train_pid} LOG=training_logs/train_residual_v2_6gpu_${run_stamp}.log"
```

Monitor without changing the run:

```bash
tail -f "training_logs/train_residual_v2_6gpu_${run_stamp}.log"
nvidia-smi dmon -i 0,1,2,3,4,5 -s pucm
```

The 30 epochs are a maximum budget. The 9,000-step learning-rate cycle is
approximately three epochs when the effective global batch remains 48.
Training metrics are logged by step and
validation metrics once per epoch. Select the model by the minimum
`val/refined_rmsd` among the five retained checkpoints, not by the last epoch.
Do not inspect the test split to choose a checkpoint; run the locked test only
after the validation-based choice is final.

## 4. Resume only after an interruption

To continue the same V2 run, set:

```yaml
ckpt_path: /storage9920/home/tinghao.xia/Code_Flow_matching/checkpoints/rna_refinement_v2_residual_mobility/last.ckpt
```

Then use the same formal launch command. `ckpt_path` restores model, optimizer,
scheduler, epoch, and global-step state. Do not use `pretrained_ckpt` for a true
resume.
