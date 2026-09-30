#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
RUNNER="$USER_ROOT/Code/predict_protenix/run_foldbench_pred.sh"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"

worker() {
  local run_id="$1"
  local mode="$2"
  local run_dir="$BASE_REPORT/pred_runs/$run_id"
  printf 'RUNNING\n' > "$run_dir/launcher.exit_code"
  set +e
  docker exec \
    -e PRED_RUN_DIR="$run_dir" \
    -e PYTHONUNBUFFERED=1 \
    -e PRED_GPUS="${PRED_GPUS:-0,1,2,3,4,5,6,7}" \
    -e SMOKE_GPU="${SMOKE_GPU:-1}" \
    protenix_test bash "$RUNNER" "$mode" \
    > "$run_dir/console.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$run_dir/launcher.exit_code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/launcher.finished_at_utc"
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
export PRED_GPUS="${PRED_GPUS:-0,1,2,3,4,5,6,7}"
export SMOKE_GPU="${SMOKE_GPU:-1}"
run_dir="$BASE_REPORT/pred_runs/$run_id"
mkdir -p "$run_dir"
printf '%s\n' "$run_id" > "$BASE_REPORT/pred_runs/current_run.txt"
printf '%s\n' "$mode" > "$run_dir/mode.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/launcher.started_at_utc"

nohup bash "$0" __worker "$run_id" "$mode" > "$run_dir/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$run_dir/launcher.pid"

echo "PRED_RUN_ID=$run_id"
echo "MODE=$mode"
echo "RUN_DIR=$run_dir"
echo "LAUNCHER_PID=$launcher_pid"
echo "监控：watch -n 30 bash $USER_ROOT/Code/predict_protenix/monitor_foldbench_pred.sh"
