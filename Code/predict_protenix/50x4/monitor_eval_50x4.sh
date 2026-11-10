#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_EVAL"
PYTHON="$USER_ROOT/miniconda3/envs/protenix-1.0.5/bin/python"

run_id="${1:-}"
if [[ -z "$run_id" ]]
then
    run_id="$(cat "$BASE_REPORT/pred_runs/current_run.txt")"
fi
run_dir="$BASE_REPORT/pred_runs/$run_id"
split_name="$(cat "$run_dir/current_split.txt" 2>/dev/null || true)"
round_label="$(cat "$run_dir/current_round.txt" 2>/dev/null || true)"
round_dir=""
if [[ -n "$split_name" && -n "$round_label" ]]
then
    round_dir="$run_dir/$split_name/rounds/$round_label"
fi

echo "RUN_ID=$run_id"
echo "MODE=$(cat "$run_dir/mode.txt" 2>/dev/null || echo UNKNOWN)"
echo "RUN_DIR=$run_dir"
echo "VAL_OUTPUT_DIR=$USER_ROOT/Json_data/Foldbench_predictions_50x4_val"
echo "TEST_OUTPUT_DIR=$USER_ROOT/Json_data/Foldbench_predictions_50x4_test"
echo "CURRENT_SPLIT=${split_name:-NOT_STARTED}"
echo "CURRENT_ROUND=${round_label:-NOT_STARTED}"
echo "LAUNCHER_EXIT=$(cat "$run_dir/launcher.exit_code" 2>/dev/null || echo UNKNOWN)"
echo "PRED_EXIT=$(cat "$run_dir/pred.exit_code" 2>/dev/null || echo UNKNOWN)"

launcher_pid="$(cat "$run_dir/launcher.pid" 2>/dev/null || true)"
pipeline_pid="$(cat "$run_dir/pipeline.pid" 2>/dev/null || true)"
if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null
then
    echo "LAUNCHER=ALIVE pid=$launcher_pid"
else
    echo "LAUNCHER=NOT_RUNNING pid=${launcher_pid:-UNKNOWN}"
fi
if [[ -n "$pipeline_pid" ]] && kill -0 "$pipeline_pid" 2>/dev/null
then
    echo "PIPELINE=ALIVE pid=$pipeline_pid"
else
    echo "PIPELINE=NOT_RUNNING pid=${pipeline_pid:-UNKNOWN}"
fi

echo '=== SELECTION ==='
if [[ -f "$run_dir/selection_summary.json" ]]
then
    "$PYTHON" - "$run_dir/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in (
    "source_train_run_id",
    "excluded_total_count",
    "val_count",
    "excluded_in_val",
    "eligible_val_count",
    "expected_val_seed_tasks",
    "expected_val_decoys",
    "test_count",
    "excluded_in_test",
    "eligible_test_count",
    "expected_test_seed_tasks",
    "expected_test_decoys",
):
    print(f"{key}={data.get(key)}")
PY
else
    echo 'SELECTION_SUMMARY=MISSING'
fi

echo '=== SPLIT EXIT CODES ==='
for item in val test
do
    echo "$item=$(cat "$run_dir/$item/pred.exit_code" 2>/dev/null || echo NOT_STARTED)"
done

echo '=== THIS RUN PROCESSES ==='
pgrep -af 'stage2_decoy_pipeline.py pred|resident_protenix_pred.py' |
    grep -F "$run_dir" || echo 'NO_EVAL_50X4_PROCESS'

echo '=== MEMORY ==='
"$PYTHON" - <<'PY'
from pathlib import Path

gib = 1024**3
current = int(Path("/sys/fs/cgroup/memory.current").read_text())
maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
available_kib = 0
for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
        available_kib = int(line.split()[1])
        break
print(f"CGROUP_CURRENT={current / gib:.1f} GiB")
if maximum != "max":
    print(f"CGROUP_MAX={int(maximum) / gib:.1f} GiB")
print(f"HOST_AVAILABLE={available_kib * 1024 / gib:.1f} GiB")
PY
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo '=== GPU ==='
nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader

echo '=== CURRENT STAGE HEARTBEATS ==='
if [[ -n "$round_dir" && -d "$round_dir" ]]
then
    RUN_DIR="$round_dir" "$PYTHON" - <<'PY'
import json
import os
import time
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
paths = sorted((run_dir / "heartbeats").glob("*.json"))
if not paths:
    print("NO_HEARTBEAT_YET")
else:
    now = time.time()
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(path.name, "READ_ERROR", exc)
            continue
        age = int(now - path.stat().st_mtime)
        print(
            path.name,
            f"age={age}s",
            f"status={data.get('status')}",
            f"current={data.get('current_target')}",
            f"target={data.get('target_index')}/{data.get('target_total')}",
            f"finished={data.get('targets_finished', 0)}",
        )
PY
    echo '=== CURRENT STAGE SUMMARY ==='
    if [[ -f "$round_dir/summary.json" ]]
    then
        "$PYTHON" - "$round_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in (
    "target_count",
    "all_seeds_complete",
    "need_pred",
    "valid_decoy_count",
    "expected_decoy_count",
):
    print(f"{key}={data.get(key)}")
PY
    else
        echo 'SUMMARY=NOT_WRITTEN_YET'
    fi
    echo '=== CURRENT STAGE CONSOLE TAIL ==='
    tail -40 "$round_dir/console.log" 2>/dev/null || true
else
    echo 'STAGE=NOT_STARTED'
fi

echo '=== MEMORY GUARD ==='
cat "$run_dir/memory_guard_stop.txt" 2>/dev/null || echo 'NOT_TRIGGERED'
