#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
RUNNER="$USER_ROOT/Code/predict_protenix/run_foldbench_stage1.sh"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"

worker() {
  local run_id="$1"
  local run_dir="$BASE_REPORT/prep_runs/$run_id"
  exec 9> "$BASE_REPORT/prep_runs/.prep.lock"
  if ! flock -n 9; then
    echo '另一个 prep launcher 正在运行，拒绝重复启动' >&2
    protenix_atomic_write "$run_dir/launcher.exit_code" 75
    return 75
  fi
  protenix_atomic_write "$run_dir/launcher.exit_code" RUNNING
  set +e
  PREP_RUN_DIR="$run_dir" bash "$RUNNER" prep \
    > "$run_dir/console.log" 2>&1
  local code=$?
  set -e
  protenix_atomic_write "$run_dir/launcher.exit_code" "$code"
  protenix_atomic_write "$run_dir/launcher.finished_at_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$code"
}

if [[ "${1:-}" == "__worker" ]]; then
  worker "$2"
  exit $?
fi

run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
protenix_require_memory_below "${START_MEMORY_LIMIT_PERCENT:-70}"
run_dir="$BASE_REPORT/prep_runs/$run_id"
mkdir -p "$run_dir"
protenix_atomic_write "$BASE_REPORT/prep_runs/current_run.txt" "$run_id"
protenix_atomic_write "$run_dir/launcher.started_at_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

nohup bash "$0" __worker "$run_id" > "$run_dir/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
protenix_atomic_write "$run_dir/launcher.pid" "$launcher_pid"

echo "PREP_RUN_ID=$run_id"
echo "RUN_DIR=$run_dir"
echo "LAUNCHER_PID=$launcher_pid"
echo "PREP_WORKERS=${PREP_WORKERS:-4} NHMMER_CPUS=${NHMMER_CPUS:-8}"
echo "监控：bash $USER_ROOT/Code/predict_protenix/monitor_foldbench_prep.sh"
