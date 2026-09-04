#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/DATA_V1_50X4_CONFIDENCE"
DATA_ROOT="$USER_ROOT/Data_V1"
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
echo "RUN_DIR=$run_dir"
echo "CURRENT_SPLIT=${split_name:-NOT_STARTED}"
echo "CURRENT_ROUND=${round_label:-NOT_STARTED}"
echo "LAUNCHER_EXIT=$(cat "$run_dir/launcher.exit_code" 2>/dev/null || echo UNKNOWN)"
echo "PRED_EXIT=$(cat "$run_dir/pred.exit_code" 2>/dev/null || echo UNKNOWN)"

launcher_pid="$(cat "$run_dir/launcher.pid" 2>/dev/null || true)"
pipeline_pid="$(cat "$run_dir/pipeline.pid" 2>/dev/null || true)"
if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null
then echo "LAUNCHER=ALIVE pid=$launcher_pid"
else echo "LAUNCHER=NOT_RUNNING pid=${launcher_pid:-UNKNOWN}"
fi
if [[ -n "$pipeline_pid" ]] && kill -0 "$pipeline_pid" 2>/dev/null
then echo "PIPELINE=ALIVE pid=$pipeline_pid"
else echo "PIPELINE=NOT_RUNNING pid=${pipeline_pid:-UNKNOWN}"
fi

echo '=== CONFIG / EXPECTED OUTPUTS ==='
"$PYTHON" - "$run_dir/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in (
    "train_count", "val_count", "test_count", "total_target_count",
    "expected_total_seed_tasks", "expected_total_decoys",
    "expected_total_full_data_json", "model_name", "diffusion_steps",
    "pairformer_cycles", "dtype", "need_atom_confidence",
):
    print(f"{key}={data.get(key)}")
PY

echo '=== RESOURCES ==='
"$PYTHON" - "$DATA_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

gib = 1024 ** 3
current = int(Path("/sys/fs/cgroup/memory.current").read_text())
available = next(
    int(line.split()[1]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
)
free = shutil.disk_usage(sys.argv[1]).free
print(f"CGROUP_CURRENT={current / gib:.1f} GiB")
print(f"HOST_AVAILABLE={available / gib:.1f} GiB")
print(f"DATA_DISK_FREE={free / gib:.1f} GiB")
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader

echo '=== SPLIT STATUS ==='
for split in train val test
do
    echo "$split=$(cat "$run_dir/$split/pred.exit_code" 2>/dev/null || echo NOT_STARTED)"
done

echo '=== CURRENT ROUND HEARTBEATS ==='
if [[ -n "$round_dir" && -d "$round_dir" ]]
then
    RUN_DIR="$round_dir" "$PYTHON" - <<'PY'
import json
import os
import time
from pathlib import Path

paths = sorted((Path(os.environ["RUN_DIR"]) / "heartbeats").glob("*.json"))
if not paths:
    print("NO_HEARTBEAT_YET")
for path in paths:
    try:
        data = json.loads(path.read_text())
        age = int(time.time() - path.stat().st_mtime)
        print(
            path.name, f"age={age}s", f"status={data.get('status')}",
            f"current={data.get('current_target')}",
            f"target={data.get('target_index')}/{data.get('target_total')}",
            f"finished={data.get('targets_finished', 0)}",
            f"failed={data.get('failed_targets', 0)}",
        )
    except Exception as exc:
        print(path.name, "READ_ERROR", exc)
PY
    echo '=== CURRENT ROUND SUMMARY ==='
    if [[ -f "$round_dir/summary.json" ]]
    then
        "$PYTHON" - "$round_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in (
    "target_count", "all_seeds_complete", "need_json", "need_prep",
    "need_pred", "valid_decoy_count", "expected_decoy_count",
    "need_atom_confidence", "all_complete",
):
    print(f"{key}={data.get(key)}")
PY
    else
        echo SUMMARY=NOT_WRITTEN_YET
    fi
    echo '=== CONSOLE TAIL ==='
    tail -40 "$round_dir/console.log" 2>/dev/null || true
else
    echo ROUND=NOT_STARTED
fi

echo '=== RESOURCE GUARD ==='
cat "$run_dir/resource_guard_stop.txt" 2>/dev/null || echo NOT_TRIGGERED
