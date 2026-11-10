#!/usr/bin/env bash
set -euo pipefail

umask 0002

USER_ROOT="/storage9920/home/tinghao.xia"
PROTENIX_ENV_PREFIX="$USER_ROOT/miniconda3/envs/protenix-1.0.5"
PYTHON="$PROTENIX_ENV_PREFIX/bin/python"
PROTENIX="$PROTENIX_ENV_PREFIX/bin/protenix"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"

MASTER_MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv"
FILTERED_2227="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest_2227_abandon14.csv"
SPLIT_MANIFEST="$USER_ROOT/Code/pipeline_reports/DATA_SPLIT_2241_CHAINMASK_20260807T114307Z_EXECUTE/final_manifest.tsv"
RETRY_RUNS="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1/pred_runs"

BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_TRAIN"
PRED_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions_50x4_train"
RUN_ID="${RUN_ID:-train50x4_8gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$BASE_REPORT/pred_runs/$RUN_ID"

GPU_LIST="0,1,2,3,4,5,6,7"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-800}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"
START_CGROUP_MAX_GIB="${START_CGROUP_MAX_GIB:-500}"
START_HOST_AVAILABLE_MIN_GIB="${START_HOST_AVAILABLE_MIN_GIB:-700}"
GPU_START_MAX_USED_MIB="${GPU_START_MAX_USED_MIB:-8192}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

for required_path in \
    "$PYTHON" \
    "$PROTENIX" \
    "$PIPELINE" \
    "$MASTER_MANIFEST" \
    "$FILTERED_2227" \
    "$SPLIT_MANIFEST"
do
    [[ -e "$required_path" ]] || die "缺少必要文件：$required_path"
done

command -v flock >/dev/null 2>&1 || die "找不到 flock"
command -v setsid >/dev/null 2>&1 || die "找不到 setsid"
command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"

retry_run_dir="$(
    find "$RETRY_RUNS" -mindepth 1 -maxdepth 1 -type d \
        -name 'retry10_2gpu_*' -print 2>/dev/null |
        sort |
        tail -1
)"
[[ -n "$retry_run_dir" ]] || die "找不到 retry10_2gpu_* 补跑目录"
RETRY_MANIFEST="$retry_run_dir/decoy_manifest.csv"
[[ -f "$RETRY_MANIFEST" ]] || die "找不到补跑审计：$RETRY_MANIFEST"

echo "RETRY_RUN_DIR=$retry_run_dir"
echo "RETRY_MANIFEST=$RETRY_MANIFEST"

mkdir -p "$RUN_DIR" "$RUN_DIR/rounds" "$PRED_OUTPUT_DIR" "$BASE_REPORT/pred_runs"

echo '=== 检查8张GPU的启动占用 ==='
gpu_too_busy=0
while IFS=',' read -r gpu_index gpu_memory gpu_util
do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_memory="${gpu_memory//[^0-9]/}"
    gpu_util="${gpu_util//[^0-9]/}"
    echo "GPU=$gpu_index memory=${gpu_memory}MiB util=${gpu_util}%"
    if (( gpu_memory > GPU_START_MAX_USED_MIB ))
    then
        echo "GPU $gpu_index 已使用超过 ${GPU_START_MAX_USED_MIB}MiB" >&2
        gpu_too_busy=1
    fi
done < <(
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader
)
(( gpu_too_busy == 0 )) || die "至少一张GPU占用过高，未启动"

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
    raise SystemExit(
        f"cgroup当前内存达到{max_current_gib}GiB，拒绝启动"
    )
if available <= min_available_gib * gib:
    raise SystemExit(
        f"宿主机可用内存低于{min_available_gib}GiB，拒绝启动"
    )
PY

echo '=== 生成并审计训练集50x4清单 ==='
"$PYTHON" - \
    "$MASTER_MANIFEST" \
    "$FILTERED_2227" \
    "$SPLIT_MANIFEST" \
    "$RETRY_MANIFEST" \
    "$USER_ROOT/Data/train" \
    "$USER_ROOT/Json_data/Simple_json" \
    "$USER_ROOT/Json_data/Complex_json" \
    "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

(
    master_path,
    filtered_path,
    split_path,
    retry_manifest_path,
    train_dir,
    simple_dir,
    complex_dir,
    run_dir,
) = map(Path, sys.argv[1:])


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized(series):
    return series.astype(str).str.strip().str.upper()


master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
filtered = pd.read_csv(filtered_path, dtype=str, keep_default_na=False)
split = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False)
retry = pd.read_csv(retry_manifest_path, dtype=str, keep_default_na=False)

for frame, label, required in (
    (master, "master manifest", {"PDB_ID", "CURRENT_TARGET"}),
    (filtered, "2227 manifest", {"PDB_ID", "CURRENT_TARGET"}),
    (split, "split manifest", {"PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"}),
    (retry, "retry manifest", {"PDB_ID", "OVERALL_STATUS"}),
):
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{label} 缺少列：{sorted(missing)}")

master["__PDB_ID"] = normalized(master["PDB_ID"])
filtered["__PDB_ID"] = normalized(filtered["PDB_ID"])
split["__PDB_ID"] = normalized(split["PDB_ID"])
retry["__PDB_ID"] = normalized(retry["PDB_ID"])

current_ids = set(
    master.loc[master["CURRENT_TARGET"].map(truth), "__PDB_ID"]
)
filtered_ids = set(
    filtered.loc[filtered["CURRENT_TARGET"].map(truth), "__PDB_ID"]
)
abandon14 = current_ids - filtered_ids
if len(abandon14) != 14:
    raise SystemExit(
        f"2241与2227清单差集不是14个，而是{len(abandon14)}个："
        f"{sorted(abandon14)}"
    )

data_ccd_12 = {
    "21ET", "2LVY", "2M39", "5TGP", "9J9X", "9Q0I",
    "9Q0J", "9RB4", "9VMZ", "25SM", "8SKQ", "9O47",
}
oom_13 = {
    "9SD9", "9ZCB", "9ZCC", "9R7W", "9J3T", "9L0R",
    "9LEE", "9LEM", "9LHL", "9LMF", "9MDS", "9MM6", "9MME",
}
abandoned25 = data_ccd_12 | oom_13
if len(abandoned25) != 25:
    raise SystemExit("12+13排除集合内部存在重复")

retry_ids = set(retry["__PDB_ID"])
retry_complete = set(
    retry.loc[normalized(retry["OVERALL_STATUS"]).eq("COMPLETE"), "__PDB_ID"]
)
retry_failed = retry_ids - retry_complete
if len(retry_ids) != 10 or len(retry_complete) != 6 or len(retry_failed) != 4:
    raise SystemExit(
        "补跑审计不符合10个目标、6成功、4失败："
        f" total={len(retry_ids)} complete={len(retry_complete)}"
        f" failed={len(retry_failed)}"
    )

train_ids = set(
    split.loc[
        normalized(split["FINAL_SPLIT"]).eq("TRAIN")
        & normalized(split["FINAL_STATUS"]).eq("KEPT"),
        "__PDB_ID",
    ]
)
if len(train_ids) != 1795:
    raise SystemExit(
        f"训练集数量不是预期1795，而是{len(train_ids)}"
    )

folder_ids = {
    path.name.strip().upper()
    for path in train_dir.iterdir()
    if path.is_dir()
}
if folder_ids != train_ids:
    raise SystemExit(
        "Data/train与划分日志不一致："
        f" missing={sorted(train_ids - folder_ids)[:20]}"
        f" extra={sorted(folder_ids - train_ids)[:20]}"
    )

excluded = abandon14 | abandoned25 | retry_failed
eligible = train_ids - excluded
expected_eligible = len(train_ids) - len(excluded & train_ids)
if len(eligible) != expected_eligible:
    raise SystemExit("候选集合计数内部不一致")

not_in_master = eligible - current_ids
if not_in_master:
    raise SystemExit(
        f"训练集候选不在当前PDB清单：{sorted(not_in_master)}"
    )

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
missing_updated = sorted(eligible - updated_ids)
missing_prep_dir = sorted(eligible - prep_ids)
if missing_updated or missing_prep_dir:
    raise SystemExit(
        "候选目标prep文件不完整："
        f" missing_updated={missing_updated[:30]}"
        f" missing_prep_dir={missing_prep_dir[:30]}"
    )

output = master[
    master["CURRENT_TARGET"].map(truth)
    & master["__PDB_ID"].isin(eligible)
].copy()
if output["__PDB_ID"].nunique() != len(eligible):
    raise SystemExit(
        "输出manifest唯一PDB数异常："
        f" {output['__PDB_ID'].nunique()} != {len(eligible)}"
    )
output = output.drop(columns=["__PDB_ID"])

manifest_path = run_dir / "train_50x4_manifest.csv"
output.to_csv(manifest_path, index=False)

# Keep the seed label below Linux NAME_MAX when the pipeline embeds all seeds
# in batch/log/heartbeat filenames.  These 50 seeds are deterministic and do
# not overlap the earlier 5x5 seeds: 42, 66, 101, 2024, and 8888.
seeds = list(range(300, 350))
(run_dir / "seeds.txt").write_text(
    "\n".join(map(str, seeds)) + "\n",
    encoding="utf-8",
)
(run_dir / "eligible_pdb_ids.txt").write_text(
    "\n".join(sorted(eligible)) + "\n",
    encoding="utf-8",
)
(run_dir / "retry_failed4.txt").write_text(
    "\n".join(sorted(retry_failed)) + "\n",
    encoding="utf-8",
)
(run_dir / "all_excluded_ids.txt").write_text(
    "\n".join(sorted(excluded)) + "\n",
    encoding="utf-8",
)

summary = {
    "train_count": len(train_ids),
    "abandon_prep_14": sorted(abandon14),
    "data_ccd_12": sorted(data_ccd_12),
    "oom_13": sorted(oom_13),
    "retry_complete_6": sorted(retry_complete),
    "retry_failed_4": sorted(retry_failed),
    "retry_failed_in_train": sorted(retry_failed & train_ids),
    "excluded_in_train_count": len(excluded & train_ids),
    "eligible_train_count": len(eligible),
    "seed_strategy": "fixed consecutive integers 300 through 349",
    "seeds": seeds,
    "samples_per_seed": 4,
    "expected_seed_tasks": len(eligible) * len(seeds),
    "expected_decoys": len(eligible) * len(seeds) * 4,
}
(run_dir / "selection_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"MANIFEST={manifest_path}")
PY

printf '%s\n' "$RUN_ID" > "$BASE_REPORT/pred_runs/current_run.txt"
printf '%s\n' 'train-50x4-8gpu-chunked' > "$RUN_DIR/mode.txt"
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
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_50X4_TRAIN"
PRED_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions_50x4_train"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$RUN_DIR/train_50x4_manifest.csv"
SEEDS="$(paste -sd, "$RUN_DIR/seeds.txt")"

GPU_LIST="0,1,2,3,4,5,6,7"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-800}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"

source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"

exec 9> "$BASE_REPORT/pred_runs/.train50x4.lock"
if ! flock -n 9
then
    printf '%s\n' 75 > "$RUN_DIR/pred.exit_code"
    printf '%s\n' 75 > "$RUN_DIR/launcher.exit_code"
    echo '另一个训练集50x4任务正在运行' >&2
    exit 75
fi

cgroup_guard_bytes=$((CGROUP_GUARD_GIB * 1024 * 1024 * 1024))
host_guard_bytes=$((HOST_AVAILABLE_GUARD_GIB * 1024 * 1024 * 1024))
round=0
final_status=0

while true
do
    round=$((round + 1))
    round_label="$(printf '%03d' "$round")"
    round_dir="$RUN_DIR/rounds/$round_label"
    mkdir -p "$round_dir"
    printf '%s\n' "$round_label" > "$RUN_DIR/current_round.txt"
    printf '%s\n' 'RUNNING' > "$round_dir/pred.exit_code"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$round_dir/started_at_utc"

    setsid "$PYTHON" "$PIPELINE" pred \
        --manifest "$MANIFEST" \
        --chain-manifest "$USER_ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv" \
        --cif-dir "$USER_ROOT/pdb_data" \
        --simple-json-dir "$USER_ROOT/Json_data/Simple_json" \
        --complex-json-dir "$USER_ROOT/Json_data/Complex_json" \
        --pred-output-dir "$PRED_OUTPUT_DIR" \
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
        host_available="$(
            awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo
        )"

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
        final_status='MEMORY_GUARD_STOP'
        break
    fi

    printf '%s\n' "$round_rc" > "$round_dir/pred.exit_code"
    if (( round_rc != 0 ))
    then
        final_status="$round_rc"
        break
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
        final_status=0
        break
    fi

    # Every completed round destroys all resident workers.  Wait for their
    # host allocations to be returned before starting another bounded round.
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
    ROUND_MAX_TARGETS="$ROUND_MAX_TARGETS" \
    MEMORY_STOP_PERCENT="$MEMORY_STOP_PERCENT" \
    CGROUP_GUARD_GIB="$CGROUP_GUARD_GIB" \
    HOST_AVAILABLE_GUARD_GIB="$HOST_AVAILABLE_GUARD_GIB" \
    bash "$RUN_DIR/launch_worker.sh" \
    > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"

echo '=== 50x4 STARTED ==='
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_DIR=$PRED_OUTPUT_DIR"
echo "LAUNCHER_PID=$launcher_pid"
echo "GPUS=$GPU_LIST"
echo "SEEDS=300..349"
echo "SAMPLES_PER_SEED=4"
echo "ROUND_MAX_TARGETS=$ROUND_MAX_TARGETS"
echo "MEMORY_STOP_PERCENT=$MEMORY_STOP_PERCENT"
echo "CGROUP_GUARD_GIB=$CGROUP_GUARD_GIB"
echo "HOST_AVAILABLE_GUARD_GIB=$HOST_AVAILABLE_GUARD_GIB"
echo "监控：bash $USER_ROOT/Code/predict_protenix/monitor_train_50x4.sh"

sleep 10
bash "$USER_ROOT/Code/predict_protenix/monitor_train_50x4.sh"
