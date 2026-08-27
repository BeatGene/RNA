#!/usr/bin/env bash
set -euo pipefail

umask 0002

USER_ROOT="/storage9920/home/tinghao.xia"
PROTENIX_ENV_PREFIX="$USER_ROOT/miniconda3/envs/protenix-1.0.5"
PYTHON="$PROTENIX_ENV_PREFIX/bin/python"
PROTENIX="$PROTENIX_ENV_PREFIX/bin/protenix"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_EVAL"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="$(paste -sd, "$RUN_DIR/seeds.txt")"

GPU_LIST="2,3,4,5,6,7"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-700}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"

source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"

exec 9> "$BASE_REPORT/pred_runs/.eval50x4.lock"
if ! flock -n 9
then
    printf '%s\n' 75 > "$RUN_DIR/pred.exit_code"
    printf '%s\n' 75 > "$RUN_DIR/launcher.exit_code"
    echo '另一个验证/测试集50x4任务正在运行' >&2
    exit 75
fi

cgroup_guard_bytes=$((CGROUP_GUARD_GIB * 1024 * 1024 * 1024))
host_guard_bytes=$((HOST_AVAILABLE_GUARD_GIB * 1024 * 1024 * 1024))
final_status=0

for split_name in val test
do
    manifest="$RUN_DIR/${split_name}_50x4_manifest.csv"
    if [[ "$split_name" == val ]]
    then
        output_dir="$USER_ROOT/Json_data/Foldbench_predictions_50x4_val"
    else
        output_dir="$USER_ROOT/Json_data/Foldbench_predictions_50x4_test"
    fi

    split_dir="$RUN_DIR/$split_name"
    mkdir -p "$split_dir/rounds"
    printf '%s\n' "$split_name" > "$RUN_DIR/current_split.txt"
    printf '%s\n' 'RUNNING' > "$split_dir/pred.exit_code"
    round=0

    while true
    do
        round=$((round + 1))
        round_label="$(printf '%03d' "$round")"
        round_dir="$split_dir/rounds/$round_label"
        mkdir -p "$round_dir"
        printf '%s\n' "$round_label" > "$RUN_DIR/current_round.txt"
        printf '%s\n' 'RUNNING' > "$round_dir/pred.exit_code"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$round_dir/started_at_utc"

        setsid "$PYTHON" "$PIPELINE" pred \
            --manifest "$manifest" \
            --chain-manifest "$USER_ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv" \
            --cif-dir "$USER_ROOT/pdb_data" \
            --simple-json-dir "$USER_ROOT/Json_data/Simple_json" \
            --complex-json-dir "$USER_ROOT/Json_data/Complex_json" \
            --pred-output-dir "$output_dir" \
            --report-dir "$round_dir" \
            --seeds "$SEEDS" \
            --samples 4 \
            --cif-validation quick \
            --protenix "$PROTENIX" \
            --gpus "$GPU_LIST" \
            --max-targets "$ROUND_MAX_TARGETS" \
            --prefer-shortest \
            --cpu-threads-per-gpu 2 \
            --memory-stop-percent "$MEMORY_STOP_PERCENT" \
            > "$round_dir/console.log" 2>&1 &

        pred_pid=$!
        printf '%s\n' "$pred_pid" > "$round_dir/pipeline.pid"
        printf '%s\n' "$pred_pid" > "$RUN_DIR/pipeline.pid"
        guard_reason=""

        while kill -0 "$pred_pid" 2>/dev/null
        do
            sleep 10
            cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)"
            host_available="$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)"
            if (( cgroup_current >= cgroup_guard_bytes ))
            then
                guard_reason="CGROUP_MEMORY_REACHED_${CGROUP_GUARD_GIB}_GIB"
            elif (( host_available <= host_guard_bytes ))
            then
                guard_reason="HOST_AVAILABLE_BELOW_${HOST_AVAILABLE_GUARD_GIB}_GIB"
            fi

            if [[ -n "$guard_reason" ]]
            then
                {
                    echo "reason=$guard_reason"
                    echo "time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                    echo "split=$split_name"
                    echo "round=$round_label"
                    echo "cgroup_current=$cgroup_current"
                    echo "host_available=$host_available"
                } > "$RUN_DIR/memory_guard_stop.txt"
                kill -TERM -- "-$pred_pid" 2>/dev/null \
                    || kill -TERM "$pred_pid" 2>/dev/null \
                    || true
                for _ in $(seq 1 12)
                do
                    kill -0 "$pred_pid" 2>/dev/null || break
                    sleep 5
                done
                if kill -0 "$pred_pid" 2>/dev/null
                then
                    kill -KILL -- "-$pred_pid" 2>/dev/null \
                        || kill -KILL "$pred_pid" 2>/dev/null \
                        || true
                fi
                break
            fi
        done

        set +e
        wait "$pred_pid"
        round_rc=$?
        set -e
        date -u +%Y-%m-%dT%H:%M:%SZ > "$round_dir/finished_at_utc"

        if [[ -n "$guard_reason" ]]
        then
            printf '%s\n' 'MEMORY_GUARD_STOP' > "$round_dir/pred.exit_code"
            printf '%s\n' 'MEMORY_GUARD_STOP' > "$split_dir/pred.exit_code"
            final_status='MEMORY_GUARD_STOP'
            break 2
        fi

        printf '%s\n' "$round_rc" > "$round_dir/pred.exit_code"
        if (( round_rc != 0 ))
        then
            printf '%s\n' "$round_rc" > "$split_dir/pred.exit_code"
            final_status="$round_rc"
            break 2
        fi

        need_pred="$($PYTHON - "$round_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("MISSING_SUMMARY")
print(int(json.loads(path.read_text())["need_pred"]))
PY
)"
        printf '%s\n' "$need_pred" > "$round_dir/need_pred_after_round.txt"
        if (( need_pred == 0 ))
        then
            printf '%s\n' 0 > "$split_dir/pred.exit_code"
            break
        fi

        for _ in $(seq 1 60)
        do
            cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)"
            if (( cgroup_current < 350 * 1024 * 1024 * 1024 ))
            then
                break
            fi
            sleep 10
        done
    done

    for _ in $(seq 1 60)
    do
        cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)"
        if (( cgroup_current < 350 * 1024 * 1024 * 1024 ))
        then
            break
        fi
        sleep 10
    done
done

printf '%s\n' "$final_status" > "$RUN_DIR/pred.exit_code"
printf '%s\n' "$final_status" > "$RUN_DIR/launcher.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/launcher.finished_at_utc"
exit 0
