#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="/storage9920/home/tinghao.xia"
source "$USER_ROOT/Code/predict_protenix/protenix_env.sh"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
CACHED_PREP="$USER_ROOT/Code/predict_protenix/cached_rna_prep.py"
PYTHON="$PROTENIX_ENV_PREFIX/bin/python"
PROTENIX="$PROTENIX_ENV_PREFIX/bin/protenix"

export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"

MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv"
CHAIN_MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv"
CIF_DIR="$USER_ROOT/pdb_data"
SIMPLE_JSON_DIR="$USER_ROOT/Json_data/Simple_json"
PREP_OUTPUT_DIR="$USER_ROOT/Json_data/Complex_json"
PRED_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions"
BASE_REPORT_DIR="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"
REPORT_DIR="${PREP_RUN_DIR:-$BASE_REPORT_DIR}"
REPORT_DIR="${PRED_RUN_DIR:-$REPORT_DIR}"

SEQRES_DB="$USER_ROOT/protenix_data/search_database_legacy/pdb_seqres.txt"
RFAM_DB="$USER_ROOT/protenix_data/search_database_legacy/Rfam.fasta"
RNACENTRAL_DB="$USER_ROOT/protenix_data/search_database_legacy/rnacentral_active.fasta"
NTRNA_DB="$USER_ROOT/protenix_data/search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
RNA_MSA_SEQUENCE_CACHE="$USER_ROOT/protenix_data/rna_msa_sequence_cache"

FOLDBENCH_SEEDS="42,66,101,2024,8888"
FOLDBENCH_SAMPLES="5"
PRED_GPUS="${PRED_GPUS:-0,1,2,3}"
SMOKE_GPU="${SMOKE_GPU:-0}"
PREP_WORKERS="${PREP_WORKERS:-4}"
NHMMER_CPUS="${NHMMER_CPUS:-8}"
PREP_MEMORY_PAUSE_PERCENT="${PREP_MEMORY_PAUSE_PERCENT:-75}"
PRED_OOM_QUARANTINE="${PRED_OOM_QUARANTINE:-$USER_ROOT/Code/predict_protenix/pred_oom_quarantine.txt}"

mkdir -p "$PRED_OUTPUT_DIR" "$REPORT_DIR"

COMMON_ARGS=(
  --manifest "$MANIFEST"
  --chain-manifest "$CHAIN_MANIFEST"
  --cif-dir "$CIF_DIR"
  --simple-json-dir "$SIMPLE_JSON_DIR"
  --complex-json-dir "$PREP_OUTPUT_DIR"
  --pred-output-dir "$PRED_OUTPUT_DIR"
  --report-dir "$REPORT_DIR"
  --seeds "$FOLDBENCH_SEEDS"
  --samples "$FOLDBENCH_SAMPLES"
  --cif-validation quick
  --protenix "$PROTENIX"
)

DATABASE_ARGS=(
  --seqres-database "$SEQRES_DB"
  --ntrna-database "$NTRNA_DB"
  --rfam-database "$RFAM_DB"
  --rnacentral-database "$RNACENTRAL_DB"
)

run_and_record() {
  local stage="$1"
  shift
  protenix_atomic_write "$REPORT_DIR/${stage}.exit_code" RUNNING
  set +e
  "$@"
  local code=$?
  set -e
  protenix_atomic_write "$REPORT_DIR/${stage}.exit_code" "$code"
  return "$code"
}

case "${1:-}" in
  preflight)
    run_and_record preflight "$PYTHON" "$PIPELINE" preflight \
      "${COMMON_ARGS[@]}" "${DATABASE_ARGS[@]}" --gpus "$PRED_GPUS"
    ;;
  audit)
    run_and_record audit "$PYTHON" "$PIPELINE" audit "${COMMON_ARGS[@]}"
    ;;
  prep)
    run_and_record prep "$PYTHON" "$CACHED_PREP" \
      --manifest "$MANIFEST" \
      --simple-json-dir "$SIMPLE_JSON_DIR" \
      --complex-json-dir "$PREP_OUTPUT_DIR" \
      --report-dir "$REPORT_DIR" \
      --cache-dir "$RNA_MSA_SEQUENCE_CACHE" \
      --database-set-id legacy_rna_databases_20260807 \
      --protenix "$PROTENIX" \
      "${DATABASE_ARGS[@]}" \
      --workers "$PREP_WORKERS" --nhmmer-cpus "$NHMMER_CPUS" \
      --memory-pause-percent "$PREP_MEMORY_PAUSE_PERCENT"
    ;;
  pred)
    run_and_record pred "$PYTHON" "$PIPELINE" pred \
      "${COMMON_ARGS[@]}" --gpus "$PRED_GPUS" \
      --prefer-shortest --exclude-pdb-file "$PRED_OOM_QUARANTINE"
    ;;
  pred-smoke)
    run_and_record pred "$PYTHON" "$PIPELINE" pred \
      "${COMMON_ARGS[@]}" --gpus "$SMOKE_GPU" --max-targets 1 \
      --prefer-shortest --exclude-pdb-file "$PRED_OOM_QUARANTINE"
    ;;
  *)
    echo "Usage: $0 {preflight|audit|prep|pred-smoke|pred}" >&2
    exit 64
    ;;
esac
