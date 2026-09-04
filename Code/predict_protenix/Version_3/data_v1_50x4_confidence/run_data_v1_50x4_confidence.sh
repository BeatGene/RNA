#!/usr/bin/env bash
set -euo pipefail

umask 0002

USER_ROOT="/storage9920/home/tinghao.xia"
PYTHON="$USER_ROOT/miniconda3/envs/protenix-1.0.5/bin/python"
PROTENIX="$USER_ROOT/miniconda3/envs/protenix-1.0.5/bin/protenix"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/DATA_V1_50X4_CONFIDENCE"
DATA_ROOT="$USER_ROOT/Data_V1"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
SEEDS="$(paste -sd, "$RUN_DIR/seeds.txt")"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SPLIT_ORDER="${SPLIT_ORDER:-train val test}"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-800}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-200}"

source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"

exec 9> "$BASE_REPORT/pred_runs/.data_v1_50x4_confidence.lock"
if ! flock -n 9
then
    printf '%s\n' 75 > "$RUN_DIR/pred.exit_code"
    printf '%s\n' 75 > "$RUN_DIR/launcher.exit_code"
    echo '另一个 Data_V1 50x4 confidence 任务正在运行' >&2
    exit 75
fi

cgroup_guard_bytes=$((CGROUP_GUARD_GIB * 1024 * 1024 * 1024))
host_guard_bytes=$((HOST_AVAILABLE_GUARD_GIB * 1024 * 1024 * 1024))
disk_guard_bytes=$((MIN_FREE_DISK_GIB * 1024 * 1024 * 1024))
final_status=0

for split_name in $SPLIT_ORDER
do
    manifest="$RUN_DIR/${split_name}_50x4_confidence_manifest.csv"
    output_dir="$DATA_ROOT/$split_name"
    split_dir="$RUN_DIR/$split_name"
    mkdir -p "$split_dir/rounds"
    printf '%s\n' "$split_name" > "$RUN_DIR/current_split.txt"
    printf '%s\n' RUNNING > "$split_dir/pred.exit_code"
    round=0

    while true
    do
        round=$((round + 1))
        round_label="$(printf '%03d' "$round")"
        round_dir="$split_dir/rounds/$round_label"
        mkdir -p "$round_dir"
        printf '%s\n' "$round_label" > "$RUN_DIR/current_round.txt"
        printf '%s\n' RUNNING > "$round_dir/pred.exit_code"
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
            --need-atom-confidence \
            --prediction-layout dataset \
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
            disk_available="$(df -PB1 "$DATA_ROOT" | awk 'NR==2 {print $4}')"
            if (( cgroup_current >= cgroup_guard_bytes ))
            then
                guard_reason="CGROUP_MEMORY_REACHED_${CGROUP_GUARD_GIB}_GIB"
            elif (( host_available <= host_guard_bytes ))
            then
                guard_reason="HOST_AVAILABLE_BELOW_${HOST_AVAILABLE_GUARD_GIB}_GIB"
            elif (( disk_available <= disk_guard_bytes ))
            then
                guard_reason="DISK_AVAILABLE_BELOW_${MIN_FREE_DISK_GIB}_GIB"
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
                    echo "disk_available=$disk_available"
                } > "$RUN_DIR/resource_guard_stop.txt"
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
            printf '%s\n' RESOURCE_GUARD_STOP > "$round_dir/pred.exit_code"
            printf '%s\n' RESOURCE_GUARD_STOP > "$split_dir/pred.exit_code"
            final_status=RESOURCE_GUARD_STOP
            break 2
        fi
        printf '%s\n' "$round_rc" > "$round_dir/pred.exit_code"
        if (( round_rc != 0 ))
        then
            printf '%s\n' "$round_rc" > "$split_dir/pred.exit_code"
            final_status="$round_rc"
            break 2
        fi

        read -r all_complete need_pred < <(
            "$PYTHON" - "$round_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(int(bool(data["all_complete"])), int(data["need_pred"]))
PY
        )
        printf '%s\n' "$need_pred" > "$round_dir/need_pred_after_round.txt"
        if (( all_complete == 1 ))
        then
            printf '%s\n' 0 > "$split_dir/pred.exit_code"
            break
        fi
        if (( need_pred == 0 ))
        then
            printf '%s\n' 3 > "$split_dir/pred.exit_code"
            echo "$split_name 未完成且已无可调度预测；检查 NEED_JSON/NEED_PREP" >&2
            final_status=3
            break 2
        fi

        for _ in $(seq 1 60)
        do
            cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)"
            (( cgroup_current < 350 * 1024 * 1024 * 1024 )) && break
            sleep 10
        done
    done
done

printf '%s\n' "$final_status" > "$RUN_DIR/pred.exit_code"
printf '%s\n' "$final_status" > "$RUN_DIR/launcher.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/launcher.finished_at_utc"
[[ "$final_status" == 0 ]]
