#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
RUNNER="$USER_ROOT/Code/predict_protenix/run_foldbench_stage1.sh"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"

worker() {
  local run_id="$1"
  local run_dir="$BASE_REPORT/prep_runs/$run_id"
  printf 'RUNNING\n' > "$run_dir/launcher.exit_code"
  set +e
  docker exec \
    -e PREP_RUN_DIR="$run_dir" \
    -e PYTHONUNBUFFERED=1 \
    protenix_test bash "$RUNNER" prep \
    > "$run_dir/console.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$run_dir/launcher.exit_code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/launcher.finished_at_utc"
  return "$code"
}

if [[ "${1:-}" == "__worker" ]]; then
  worker "$2"
  exit $?
fi

run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$BASE_REPORT/prep_runs/$run_id"
mkdir -p "$run_dir"
printf '%s\n' "$run_id" > "$BASE_REPORT/prep_runs/current_run.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/launcher.started_at_utc"

nohup bash "$0" __worker "$run_id" > "$run_dir/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$run_dir/launcher.pid"

echo "PREP_RUN_ID=$run_id"
echo "RUN_DIR=$run_dir"
echo "LAUNCHER_PID=$launcher_pid"
echo "监控：bash $USER_ROOT/Code/predict_protenix/monitor_foldbench_prep.sh"
