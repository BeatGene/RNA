#!/usr/bin/env bash
set -euo pipefail

umask 0002

USER_ROOT="/storage9920/home/tinghao.xia"
PYTHON="$USER_ROOT/miniconda3/envs/protenix-1.0.5/bin/python"
PROTENIX="$USER_ROOT/miniconda3/envs/protenix-1.0.5/bin/protenix"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
RESIDENT="$USER_ROOT/Code/predict_protenix/resident_protenix_pred.py"
SCRIPT_DIR="$USER_ROOT/Code/predict_protenix/Version_3/data_v1_50x4_confidence"
PREPARE="$SCRIPT_DIR/prepare_data_v1_confidence_run.py"
WORKER="$SCRIPT_DIR/run_data_v1_50x4_confidence.sh"

MASTER_MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv"
SPLIT_MANIFEST="$USER_ROOT/Code/pipeline_reports/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE/final_manifest.tsv"
DATA_ROOT="$USER_ROOT/Data_V1"
BASE_REPORT="$USER_ROOT/Code/pipeline_reports/DATA_V1_50X4_CONFIDENCE"
RUN_ID="${RUN_ID:-data_v1_50x4_conf_8gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$BASE_REPORT/pred_runs/$RUN_ID"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SPLIT_ORDER="${SPLIT_ORDER:-train val test}"
ROUND_MAX_TARGETS="${ROUND_MAX_TARGETS:-640}"
MEMORY_STOP_PERCENT="${MEMORY_STOP_PERCENT:-40}"
CGROUP_GUARD_GIB="${CGROUP_GUARD_GIB:-800}"
HOST_AVAILABLE_GUARD_GIB="${HOST_AVAILABLE_GUARD_GIB:-500}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-200}"
SMOKE_TEST_FIRST="${SMOKE_TEST_FIRST:-1}"
GPU_START_MAX_USED_MIB="${GPU_START_MAX_USED_MIB:-8192}"
START_CGROUP_MAX_GIB="${START_CGROUP_MAX_GIB:-500}"
START_HOST_AVAILABLE_MIN_GIB="${START_HOST_AVAILABLE_MIN_GIB:-700}"

die() { echo "ERROR: $*" >&2; exit 1; }

for required_path in \
    "$PYTHON" "$PROTENIX" "$PIPELINE" "$RESIDENT" "$PREPARE" "$WORKER" \
    "$MASTER_MANIFEST" "$SPLIT_MANIFEST" "$DATA_ROOT/train" \
    "$DATA_ROOT/val" "$DATA_ROOT/test" \
    "$USER_ROOT/Json_data/Simple_json" "$USER_ROOT/Json_data/Complex_json"
do
    [[ -e "$required_path" ]] || die "缺少必要路径：$required_path"
done
command -v flock >/dev/null 2>&1 || die "找不到 flock"
command -v setsid >/dev/null 2>&1 || die "找不到 setsid"
command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"

[[ -n "$SPLIT_ORDER" ]] || die "SPLIT_ORDER 不能为空"
for split_name in $SPLIT_ORDER
do
    case "$split_name" in
        train|val|test) ;;
        *) die "SPLIT_ORDER 含非法划分：$split_name" ;;
    esac
done

echo "=== 检查 GPU：$GPU_LIST ==="
gpu_too_busy=0
while IFS=',' read -r gpu_index gpu_memory gpu_util
do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_memory="${gpu_memory//[^0-9]/}"
    gpu_util="${gpu_util//[^0-9]/}"
    case ",$GPU_LIST," in
        *",$gpu_index,"*)
            echo "GPU=$gpu_index memory=${gpu_memory}MiB util=${gpu_util}%"
            (( gpu_memory <= GPU_START_MAX_USED_MIB )) || gpu_too_busy=1
            ;;
    esac
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
(( gpu_too_busy == 0 )) || die "所选 GPU 中至少一张占用过高"

cgroup_current="$(cat /sys/fs/cgroup/memory.current)"
host_available="$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)"
disk_available="$(df -PB1 "$DATA_ROOT" | awk 'NR==2 {print $4}')"
(( cgroup_current < START_CGROUP_MAX_GIB * 1024 * 1024 * 1024 )) \
    || die "cgroup 当前内存达到 ${START_CGROUP_MAX_GIB} GiB"
(( host_available > START_HOST_AVAILABLE_MIN_GIB * 1024 * 1024 * 1024 )) \
    || die "宿主机可用内存低于 ${START_HOST_AVAILABLE_MIN_GIB} GiB"
(( disk_available > MIN_FREE_DISK_GIB * 1024 * 1024 * 1024 )) \
    || die "Data_V1 所在文件系统可用空间低于 ${MIN_FREE_DISK_GIB} GiB"

mkdir -p "$RUN_DIR" "$BASE_REPORT/pred_runs"
seq 300 349 > "$RUN_DIR/seeds.txt"
SEEDS="$(paste -sd, "$RUN_DIR/seeds.txt")"

echo '=== 生成并审计 Data_V1 50x4 清单 ==='
"$PYTHON" "$PREPARE" \
    --master-manifest "$MASTER_MANIFEST" \
    --split-manifest "$SPLIT_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --simple-json-dir "$USER_ROOT/Json_data/Simple_json" \
    --complex-json-dir "$USER_ROOT/Json_data/Complex_json" \
    --run-dir "$RUN_DIR" \
    --seeds "$SEEDS" \
    --samples 4

source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"
if [[ "$SMOKE_TEST_FIRST" == 1 ]]
then
    echo '=== 先运行 1 target x 1 seed x 4 samples 完整置信度冒烟测试 ==='
    smoke_gpu="${GPU_LIST%%,*}"
    "$PYTHON" "$PIPELINE" pred \
        --manifest "$RUN_DIR/smoke_manifest.csv" \
        --chain-manifest "$USER_ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv" \
        --cif-dir "$USER_ROOT/pdb_data" \
        --simple-json-dir "$USER_ROOT/Json_data/Simple_json" \
        --complex-json-dir "$USER_ROOT/Json_data/Complex_json" \
        --pred-output-dir "$RUN_DIR/smoke_output" \
        --report-dir "$RUN_DIR/smoke_report" \
        --seeds 300 \
        --samples 4 \
        --need-atom-confidence \
        --prediction-layout dataset \
        --cif-validation quick \
        --protenix "$PROTENIX" \
        --gpus "$smoke_gpu" \
        --max-targets 1 \
        --prefer-shortest \
        --cpu-threads-per-gpu 2 \
        --memory-stop-percent "$MEMORY_STOP_PERCENT"
    "$PYTHON" - "$RUN_DIR/smoke_report/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary.get("all_complete") or not summary.get("need_atom_confidence"):
    raise SystemExit(f"完整置信度冒烟测试未通过：{summary}")
print("完整置信度冒烟测试通过")
PY
fi

printf '%s\n' "$RUN_ID" > "$BASE_REPORT/pred_runs/current_run.txt"
printf '%s\n' data-v1-50x4-confidence-8gpu > "$RUN_DIR/mode.txt"
printf '%s\n' RUNNING > "$RUN_DIR/launcher.exit_code"
printf '%s\n' RUNNING > "$RUN_DIR/pred.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/launcher.started_at_utc"

nohup env \
    RUN_DIR="$RUN_DIR" \
    GPU_LIST="$GPU_LIST" \
    SPLIT_ORDER="$SPLIT_ORDER" \
    ROUND_MAX_TARGETS="$ROUND_MAX_TARGETS" \
    MEMORY_STOP_PERCENT="$MEMORY_STOP_PERCENT" \
    CGROUP_GUARD_GIB="$CGROUP_GUARD_GIB" \
    HOST_AVAILABLE_GUARD_GIB="$HOST_AVAILABLE_GUARD_GIB" \
    MIN_FREE_DISK_GIB="$MIN_FREE_DISK_GIB" \
    bash "$WORKER" > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"

echo '=== DATA_V1 50x4 CONFIDENCE STARTED ==='
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "LAUNCHER_PID=$launcher_pid"
echo "GPUS=$GPU_LIST"
echo "SPLIT_ORDER=$SPLIT_ORDER"
echo 'MODEL=protenix_base_default_v1.0.0'
echo 'SEEDS=300..349; SAMPLES_PER_SEED=4; N_STEP=200; N_CYCLE=10; DTYPE=bf16'
echo 'FULL_DATA=atom_plddt,token_pair_pae,token_pair_pde,contact_probs,atom_to_token_idx'
echo "监控：bash $SCRIPT_DIR/monitor_data_v1_50x4_confidence.sh"
