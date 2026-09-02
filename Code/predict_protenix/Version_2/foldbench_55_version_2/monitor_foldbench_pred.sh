#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"
run_id="${1:-}"
if [[ -z "$run_id" ]]; then
  run_id="$(cat "$BASE_REPORT/pred_runs/current_run.txt")"
fi
run_dir="$BASE_REPORT/pred_runs/$run_id"

echo "RUN_ID=$run_id MODE=$(cat "$run_dir/mode.txt" 2>/dev/null || echo UNKNOWN)"
echo "RUN_DIR=$run_dir"
launcher_pid="$(cat "$run_dir/launcher.pid" 2>/dev/null || true)"
if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
  echo "HOST_LAUNCHER=ALIVE pid=$launcher_pid"
else
  echo "HOST_LAUNCHER=NOT_RUNNING pid=${launcher_pid:-unknown}"
fi
echo "LAUNCHER_EXIT=$(cat "$run_dir/launcher.exit_code" 2>/dev/null || echo UNKNOWN)"
echo "PRED_EXIT=$(cat "$run_dir/pred.exit_code" 2>/dev/null || echo UNKNOWN)"

echo 'PRED_PROCESSES:'
pgrep -af 'stage2_decoy_pipeline.py pred|resident_protenix_pred.py' \
  || echo 'NO_PRED_PROCESS'

echo 'MEMORY:'
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
protenix_memory_snapshot
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo 'GPU_PROCESSES:'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader \
  2>/dev/null || true
echo 'GPU_UTILIZATION:'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader \
  2>/dev/null || true

echo 'WORKER_HEARTBEATS:'
now_epoch="$(date +%s)"
fresh=0
stale=0
missing=0
shopt -s nullglob
heartbeats=("$run_dir"/heartbeats/*.json)
if (( ${#heartbeats[@]} == 0 )); then
  echo 'HEARTBEAT_STATUS=MISSING (模型可能仍在首次加载)'
  missing=1
else
  for path in "${heartbeats[@]}"; do
    age=$((now_epoch - $(stat -c %Y "$path")))
    worker_state="$(sed -n 's/.*"status": "\([^"]*\)".*/\1/p' "$path" | head -1)"
    if [[ "$worker_state" == "COMPLETE" ]]; then
      status=COMPLETE
    elif [[ "$worker_state" == "FAILED" ]]; then
      status=FAILED
      stale=$((stale + 1))
    elif (( age > 120 )); then
      status=STALE
      stale=$((stale + 1))
    else
      status=FRESH
      fresh=$((fresh + 1))
    fi
    echo "$(basename "$path") age=${age}s worker=$worker_state status=$status"
  done
fi
echo "HEARTBEAT_SUMMARY fresh=$fresh stale=$stale missing=$missing"

echo 'OUTPUT_COUNTS:'
find "$USER_ROOT/Json_data/Foldbench_predictions" -type f \
  -name '*_sample_*.cif' ! -name '*_wounresol.cif' 2>/dev/null | wc -l
find "$USER_ROOT/Json_data/Foldbench_predictions" -type d -name 'seed_*' \
  2>/dev/null | wc -l | awk '{print "seed_directories=" $1}'
echo 'DEFERRED_PREP_COUNT:'
if [[ -f "$run_dir/pred_deferred_prep.csv" ]]; then
  tail -n +2 "$run_dir/pred_deferred_prep.csv" | wc -l
else
  echo UNKNOWN
fi
echo 'CONSOLE_TAIL:'
tail -30 "$run_dir/console.log" 2>/dev/null || true
