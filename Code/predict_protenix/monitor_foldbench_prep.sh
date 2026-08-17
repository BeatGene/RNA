#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"
run_id="${1:-}"
if [[ -z "$run_id" ]]; then
  run_id="$(cat "$BASE_REPORT/prep_runs/current_run.txt")"
fi
run_dir="$BASE_REPORT/prep_runs/$run_id"

echo "RUN_ID=$run_id"
echo "RUN_DIR=$run_dir"

launcher_pid="$(cat "$run_dir/launcher.pid" 2>/dev/null || true)"
if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
  echo "HOST_LAUNCHER=ALIVE pid=$launcher_pid"
else
  echo "HOST_LAUNCHER=NOT_RUNNING pid=${launcher_pid:-unknown}"
fi

echo "LAUNCHER_EXIT=$(cat "$run_dir/launcher.exit_code" 2>/dev/null || echo UNKNOWN)"
echo "PREP_EXIT=$(cat "$run_dir/prep.exit_code" 2>/dev/null || echo UNKNOWN)"

echo 'PIPELINE_PROCESSES:'
pgrep -af 'cached_rna_prep|protenix prep|nhmmer|hmmalign|hmmbuild' \
  || echo 'NO_PREP_PROCESS'

echo 'MEMORY:'
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
protenix_memory_snapshot
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

if [[ -f "$run_dir/heartbeat.json" ]]; then
  heartbeat_epoch="$(stat -c %Y "$run_dir/heartbeat.json")"
  now_epoch="$(date +%s)"
  age=$((now_epoch - heartbeat_epoch))
  echo "HEARTBEAT_AGE_SECONDS=$age"
  if (( age > 120 )); then
    echo 'HEARTBEAT_STATUS=STALE'
  else
    echo 'HEARTBEAT_STATUS=FRESH'
  fi
  cat "$run_dir/heartbeat.json"
else
  echo 'HEARTBEAT_STATUS=MISSING'
fi

echo 'CACHE_AND_OUTPUT_COUNTS:'
find "$USER_ROOT/protenix_data/rna_msa_sequence_cache/legacy_rna_databases_20260807/entries" \
  -mindepth 2 -maxdepth 2 -type f -name 'rna_msa.a3m' -size +0c \
  2>/dev/null | wc -l | awk '{print "sequence_cache_entries=" $1}'
find "$USER_ROOT/Json_data/Simple_json" -maxdepth 1 -type f \
  -name '*-final-updated.json' 2>/dev/null | wc -l \
  | awk '{print "physical_final_updated_json=" $1 " (includes 5 excluded residuals)"}'

echo 'CONSOLE_TAIL:'
tail -30 "$run_dir/console.log" 2>/dev/null || true
