#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
PYTHON="$PROTENIX_ENV_PREFIX/bin/python"

echo '=== HOST LOAD / MEMORY ==='
uptime
free -h

echo '=== CGROUP / PIPELINE ==='
protenix_memory_snapshot
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
pgrep -af 'cached_rna_prep|stage2_decoy_pipeline|resident_protenix_pred|protenix prep|nhmmer' \
  || echo 'NO_ACTIVE_PIPELINE_PROCESS'

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
nvidia-smi pmon -c 1

echo '=== PROTENIX PYTHON API ==='
"$PYTHON" -c \
  'import importlib.metadata as m; import protenix, runner.batch_inference, runner.inference; print("IMPORT_OK", m.version("protenix"))'

echo '=== TOOLCHAIN ==='
"$PYTHON" -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "CUDA_HOME", CUDA_HOME)'
command -v nhmmer
command -v kalign
command -v nvcc
command -v flock

echo '=== CHECKPOINT ==='
ls -lh "$USER_ROOT/protenix_data/checkpoint/protenix_base_default_v1.0.0.pt"

echo '=== WRITABILITY ==='
for path in \
  "$USER_ROOT/Json_data/Foldbench_predictions" \
  "$USER_ROOT/protenix_data/rna_msa_sequence_cache" \
  "$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"
do
  test -w "$path" || { echo "NOT_WRITABLE $path"; exit 1; }
  echo "WRITABLE $path"
done
