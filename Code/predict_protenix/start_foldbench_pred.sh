#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
RUNNER="$USER_ROOT/Code/predict_protenix/run_foldbench_pred.sh"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"

worker() {
  local run_id="$1"
  local mode="$2"
  local run_dir="$BASE_REPORT/pred_runs/$run_id"
  exec 9> "$BASE_REPORT/pred_runs/.pred.lock"
  if ! flock -n 9; then
    echo '另一个 pred launcher 正在运行，拒绝重复启动' >&2
    protenix_atomic_write "$run_dir/launcher.exit_code" 75
    return 75
  fi
  protenix_atomic_write "$run_dir/launcher.exit_code" RUNNING
  set +e
  PRED_RUN_DIR="$run_dir" bash "$RUNNER" "$mode" \
    > "$run_dir/console.log" 2>&1
  local code=$?
  set -e
  protenix_atomic_write "$run_dir/launcher.exit_code" "$code"
  protenix_atomic_write "$run_dir/launcher.finished_at_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$code"
}

if [[ "${1:-}" == "__worker" ]]; then
  worker "$2" "$3"
  exit $?
fi

mode="${1:-pred-smoke}"
if [[ "$mode" != "pred-smoke" && "$mode" != "pred" ]]; then
  echo "Usage: $0 {pred-smoke|pred} [run_id]" >&2
  exit 64
fi
run_id="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PRED_GPUS="${PRED_GPUS:-0,1,2,3}"
export SMOKE_GPU="${SMOKE_GPU:-0}"
protenix_require_memory_below "${START_MEMORY_LIMIT_PERCENT:-70}"
run_dir="$BASE_REPORT/pred_runs/$run_id"
mkdir -p "$run_dir"
protenix_atomic_write "$BASE_REPORT/pred_runs/current_run.txt" "$run_id"
protenix_atomic_write "$run_dir/mode.txt" "$mode"
protenix_atomic_write "$run_dir/launcher.started_at_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

nohup bash "$0" __worker "$run_id" "$mode" > "$run_dir/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
protenix_atomic_write "$run_dir/launcher.pid" "$launcher_pid"

echo "PRED_RUN_ID=$run_id"
echo "MODE=$mode"
echo "RUN_DIR=$run_dir"
echo "LAUNCHER_PID=$launcher_pid"
echo "PRED_GPUS=$PRED_GPUS SMOKE_GPU=$SMOKE_GPU"
echo "监控：watch -n 30 bash $USER_ROOT/Code/predict_protenix/monitor_foldbench_pred.sh"
