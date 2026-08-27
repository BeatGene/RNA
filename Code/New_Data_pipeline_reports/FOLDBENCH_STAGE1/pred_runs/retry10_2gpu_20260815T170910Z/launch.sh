#!/usr/bin/env bash
set -uo pipefail
umask 0002

ROOT="$HOME"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_REPORT="$ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"
PYTHON="$ROOT/miniconda3/envs/protenix-1.0.5/bin/python"
PROTENIX="$ROOT/miniconda3/envs/protenix-1.0.5/bin/protenix"
PIPELINE="$ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
MANIFEST="$ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest_retry10.csv"

source "$ROOT/Code/predict_protenix/protenix_env.sh"
export PROTENIX_ROOT_DIR="$ROOT/protenix_data"

exec 9> "$BASE_REPORT/pred_runs/.pred.lock"
if ! flock -n 9
then
    printf '%s\n' 75 > "$RUN_DIR/pred.exit_code"
    printf '%s\n' 75 > "$RUN_DIR/launcher.exit_code"
    exit 75
fi

set +e
"$PYTHON" "$PIPELINE" pred \
    --manifest "$MANIFEST" \
    --chain-manifest "$ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv" \
    --cif-dir "$ROOT/pdb_data" \
    --simple-json-dir "$ROOT/Json_data/Simple_json" \
    --complex-json-dir "$ROOT/Json_data/Complex_json" \
    --pred-output-dir "$ROOT/Json_data/Foldbench_predictions" \
    --report-dir "$RUN_DIR" \
    --seeds 42,66,101,2024,8888 \
    --samples 5 \
    --cif-validation quick \
    --protenix "$PROTENIX" \
    --gpus 2,3 \
    --prefer-shortest \
    --cpu-threads-per-gpu 2 \
    --memory-stop-percent 45 \
    > "$RUN_DIR/console.log" 2>&1
rc=$?
set -e

printf '%s\n' "$rc" > "$RUN_DIR/pred.exit_code"
printf '%s\n' "$rc" > "$RUN_DIR/launcher.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/launcher.finished_at_utc"
exit "$rc"
