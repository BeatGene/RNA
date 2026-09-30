#!/usr/bin/env bash
set -euo pipefail

umask 0002

USER_ROOT="/storage9920/home/tinghao.xia"
PROTENIX_ENV_PREFIX="$USER_ROOT/miniconda3/envs/protenix-1.0.5"
PYTHON="$PROTENIX_ENV_PREFIX/bin/python"
PROTENIX="$PROTENIX_ENV_PREFIX/bin/protenix"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"

MASTER_MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv"
SPLIT_MANIFEST="$USER_ROOT/Code/pipeline_reports/DATA_SPLIT_2241_CHAINMASK_20260807T114307Z_EXECUTE/final_manifest.tsv"
TRAIN_BASE="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_TRAIN"
TRAIN_RUN_ID="$(cat "$TRAIN_BASE/pred_runs/current_run.txt" 2>/dev/null || true)"
TRAIN_RUN_DIR="$TRAIN_BASE/pred_runs/$TRAIN_RUN_ID"
EXCLUDED_IDS="$TRAIN_RUN_DIR/all_excluded_ids.txt"

BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_EVAL"
VAL_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions_50x4_val"
TEST_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions_50x4_test"
RUN_ID="${RUN_ID:-eval50x4_6gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$BASE_REPORT/pred_runs/$RUN_ID"

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
SPLIT_ORDER="${SPLIT_ORDER:-val test}"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-700}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"
START_CGROUP_MAX_GIB="${START_CGROUP_MAX_GIB:-500}"
START_HOST_AVAILABLE_MIN_GIB="${START_HOST_AVAILABLE_MIN_GIB:-700}"
GPU_START_MAX_USED_MIB="${GPU_START_MAX_USED_MIB:-8192}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -n "$TRAIN_RUN_ID" ]] || die "找不到训练集50x4 current_run.txt"
for required_path in \
    "$PYTHON" \
    "$PROTENIX" \
    "$PIPELINE" \
    "$MASTER_MANIFEST" \
    "$SPLIT_MANIFEST" \
    "$EXCLUDED_IDS" \
    "$USER_ROOT/Data/val" \
    "$USER_ROOT/Data/test"
do
    [[ -e "$required_path" ]] || die "缺少必要文件：$required_path"
done

command -v flock >/dev/null 2>&1 || die "找不到 flock"
command -v setsid >/dev/null 2>&1 || die "找不到 setsid"
command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"

mkdir -p \
    "$RUN_DIR" \
    "$VAL_OUTPUT_DIR" \
    "$TEST_OUTPUT_DIR" \
    "$BASE_REPORT/pred_runs"

case "$SPLIT_ORDER" in
    val|test|'val test') ;;
    *) die "SPLIT_ORDER只能是val、test或'val test'" ;;
esac

echo "=== 检查所选GPU（$GPU_LIST）的启动占用 ==="
gpu_too_busy=0
while IFS=',' read -r gpu_index gpu_memory gpu_util
do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_memory="${gpu_memory//[^0-9]/}"
    gpu_util="${gpu_util//[^0-9]/}"
    case ",$GPU_LIST," in
        *",$gpu_index,"*)
            echo "GPU=$gpu_index memory=${gpu_memory}MiB util=${gpu_util}%"
            if (( gpu_memory > GPU_START_MAX_USED_MIB ))
            then
                echo "GPU $gpu_index 已使用超过 ${GPU_START_MAX_USED_MIB}MiB" >&2
                gpu_too_busy=1
            fi
            ;;
    esac
done < <(
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader
)
(( gpu_too_busy == 0 )) || die "所选GPU中至少一张占用过高，未启动"

echo '=== 检查启动前主机内存 ==='
cgroup_current="$(cat /sys/fs/cgroup/memory.current)"
host_available="$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)"
"$PYTHON" - \
    "$cgroup_current" \
    "$host_available" \
    "$START_CGROUP_MAX_GIB" \
    "$START_HOST_AVAILABLE_MIN_GIB" <<'PY'
import sys

current = int(float(sys.argv[1]))
available = int(float(sys.argv[2]))
max_current_gib = int(sys.argv[3])
min_available_gib = int(sys.argv[4])
gib = 1024**3

print(f"CGROUP_CURRENT={current / gib:.1f} GiB")
print(f"HOST_AVAILABLE={available / gib:.1f} GiB")
if current >= max_current_gib * gib:
    raise SystemExit(f"cgroup当前内存达到{max_current_gib}GiB，拒绝启动")
if available <= min_available_gib * gib:
    raise SystemExit(f"宿主机可用内存低于{min_available_gib}GiB，拒绝启动")
PY

echo '=== 生成并审计验证集/测试集50x4清单 ==='
"$PYTHON" - \
    "$MASTER_MANIFEST" \
    "$SPLIT_MANIFEST" \
    "$EXCLUDED_IDS" \
    "$USER_ROOT/Data/val" \
    "$USER_ROOT/Data/test" \
    "$USER_ROOT/Json_data/Simple_json" \
    "$USER_ROOT/Json_data/Complex_json" \
    "$RUN_DIR" \
    "$TRAIN_RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

(
    master_path,
    split_path,
    excluded_path,
    val_dir,
    test_dir,
    simple_dir,
    complex_dir,
    run_dir,
) = map(Path, sys.argv[1:9])
train_run_id = sys.argv[9]


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized(series):
    return series.astype(str).str.strip().str.upper()


def folder_ids(path):
    return {
        child.name.strip().upper()
        for child in path.iterdir()
        if child.is_dir()
    }


master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
split = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False)
for frame, label, required in (
    (master, "master manifest", {"PDB_ID", "CURRENT_TARGET"}),
    (split, "split manifest", {"PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"}),
):
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{label} 缺少列：{sorted(missing)}")

master["__PDB_ID"] = normalized(master["PDB_ID"])
split["__PDB_ID"] = normalized(split["PDB_ID"])
split_name = normalized(split["FINAL_SPLIT"])
split_status = normalized(split["FINAL_STATUS"])

current_ids = set(
    master.loc[master["CURRENT_TARGET"].map(truth), "__PDB_ID"]
)
val_log_ids = set(
    split.loc[
        split_name.eq("VAL") & split_status.eq("KEPT"),
        "__PDB_ID",
    ]
)
test_log_ids = set(
    split.loc[
        split_name.eq("TEST") & split_status.str.startswith("KEEP"),
        "__PDB_ID",
    ]
)
if len(val_log_ids) != 116:
    raise SystemExit(f"划分日志中的验证集不是116个，而是{len(val_log_ids)}个")
if len(test_log_ids) != 101:
    raise SystemExit(f"划分日志中的测试集不是101个，而是{len(test_log_ids)}个")

val_folder_ids = folder_ids(val_dir)
test_folder_ids = folder_ids(test_dir)
if val_folder_ids != val_log_ids:
    raise SystemExit(
        "Data/val与划分日志不一致："
        f" missing={sorted(val_log_ids - val_folder_ids)[:20]}"
        f" extra={sorted(val_folder_ids - val_log_ids)[:20]}"
    )
if test_folder_ids != test_log_ids:
    raise SystemExit(
        "Data/test与划分日志不一致："
        f" missing={sorted(test_log_ids - test_folder_ids)[:20]}"
        f" extra={sorted(test_folder_ids - test_log_ids)[:20]}"
    )

excluded = {
    line.strip().upper()
    for line in excluded_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
eligible_val = val_log_ids - excluded
eligible_test = test_log_ids - excluded

for label, eligible in (("val", eligible_val), ("test", eligible_test)):
    outside = eligible - current_ids
    if outside:
        raise SystemExit(f"{label}候选不在当前PDB清单：{sorted(outside)}")

suffix = "-final-updated.json"
updated_ids = {
    path.name[:-len(suffix)].upper()
    for path in simple_dir.glob("*-final-updated.json")
    if path.name.lower().endswith(suffix)
}
prep_ids = {
    path.name[len("prep_output_"):].upper()
    for path in complex_dir.glob("prep_output_*")
    if path.is_dir()
}
for label, eligible in (("val", eligible_val), ("test", eligible_test)):
    missing_updated = sorted(eligible - updated_ids)
    missing_prep_dir = sorted(eligible - prep_ids)
    if missing_updated or missing_prep_dir:
        raise SystemExit(
            f"{label}候选prep文件不完整："
            f" missing_updated={missing_updated[:30]}"
            f" missing_prep_dir={missing_prep_dir[:30]}"
        )

for label, eligible in (("val", eligible_val), ("test", eligible_test)):
    output = master[
        master["CURRENT_TARGET"].map(truth)
        & master["__PDB_ID"].isin(eligible)
    ].copy()
    if output["__PDB_ID"].nunique() != len(eligible):
        raise SystemExit(
            f"{label}输出manifest唯一PDB数异常："
            f" {output['__PDB_ID'].nunique()} != {len(eligible)}"
        )
    output.drop(columns=["__PDB_ID"]).to_csv(
        run_dir / f"{label}_50x4_manifest.csv",
        index=False,
    )
    (run_dir / f"eligible_{label}_pdb_ids.txt").write_text(
        "\n".join(sorted(eligible)) + "\n",
        encoding="utf-8",
    )

seeds = list(range(300, 350))
(run_dir / "seeds.txt").write_text(
    "\n".join(map(str, seeds)) + "\n",
    encoding="utf-8",
)
(run_dir / "all_excluded_ids.txt").write_text(
    "\n".join(sorted(excluded)) + "\n",
    encoding="utf-8",
)

summary = {
    "source_train_run_id": train_run_id,
    "excluded_total_count": len(excluded),
    "val_count": len(val_log_ids),
    "excluded_in_val": sorted(val_log_ids & excluded),
    "eligible_val_count": len(eligible_val),
    "test_count": len(test_log_ids),
    "excluded_in_test": sorted(test_log_ids & excluded),
    "eligible_test_count": len(eligible_test),
    "seeds": seeds,
    "samples_per_seed": 4,
    "expected_val_seed_tasks": len(eligible_val) * len(seeds),
    "expected_val_decoys": len(eligible_val) * len(seeds) * 4,
    "expected_test_seed_tasks": len(eligible_test) * len(seeds),
    "expected_test_decoys": len(eligible_test) * len(seeds) * 4,
}
(run_dir / "selection_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '%s\n' "$RUN_ID" > "$BASE_REPORT/pred_runs/current_run.txt"
printf '%s\n' 'eval-50x4-6gpu-val-then-test' > "$RUN_DIR/mode.txt"
printf '%s\n' 'RUNNING' > "$RUN_DIR/launcher.exit_code"
printf '%s\n' 'RUNNING' > "$RUN_DIR/pred.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/launcher.started_at_utc"

cat > "$RUN_DIR/launch_worker.sh" <<'WORKER'
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

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
SPLIT_ORDER="${SPLIT_ORDER:-val test}"
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

for split_name in $SPLIT_ORDER
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
WORKER

chmod 0755 "$RUN_DIR/launch_worker.sh"

nohup env \
    GPU_LIST="$GPU_LIST" \
    SPLIT_ORDER="$SPLIT_ORDER" \
    ROUND_MAX_TARGETS="$ROUND_MAX_TARGETS" \
    MEMORY_STOP_PERCENT="$MEMORY_STOP_PERCENT" \
    CGROUP_GUARD_GIB="$CGROUP_GUARD_GIB" \
    HOST_AVAILABLE_GUARD_GIB="$HOST_AVAILABLE_GUARD_GIB" \
    bash "$RUN_DIR/launch_worker.sh" \
    > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"

echo '=== EVAL 50x4 STARTED ==='
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "VAL_OUTPUT_DIR=$VAL_OUTPUT_DIR"
echo "TEST_OUTPUT_DIR=$TEST_OUTPUT_DIR"
echo "LAUNCHER_PID=$launcher_pid"
echo "GPUS=$GPU_LIST"
echo "ORDER=$SPLIT_ORDER"
echo 'SEEDS=300..349'
echo 'SAMPLES_PER_SEED=4'
echo "MEMORY_STOP_PERCENT=$MEMORY_STOP_PERCENT"
echo "CGROUP_GUARD_GIB=$CGROUP_GUARD_GIB"
echo "监控：bash $USER_ROOT/Code/predict_protenix/monitor_eval_50x4.sh"

sleep 10
bash "$USER_ROOT/Code/predict_protenix/monitor_eval_50x4.sh"
