#!/usr/bin/env bash
set -euo pipefail

# Run inside the existing protenix_test container. /storage9920 is mounted at
# the same path on the host and in the container.
USER_ROOT="/storage9920/home/tinghao.xia"
PIPELINE="$USER_ROOT/Code/predict_protenix/stage2_decoy_pipeline.py"
PYTHON="/root/miniconda3/bin/python"
PROTENIX="/root/miniconda3/bin/protenix"

export PROTENIX_ROOT_DIR="$USER_ROOT/protenix_data"

MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv"
CHAIN_MANIFEST="$USER_ROOT/Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv"
CIF_DIR="$USER_ROOT/pdb_data"
SIMPLE_JSON_DIR="$USER_ROOT/Json_data/Simple_json"
PREP_OUTPUT_DIR="$USER_ROOT/Json_data/Complex_json"
PRED_OUTPUT_DIR="$USER_ROOT/Json_data/Foldbench_predictions"
REPORT_DIR="$USER_ROOT/Code/pipeline_reports/FOLDBENCH_STAGE1"

SEQRES_DB="$USER_ROOT/protenix_data/search_database_legacy/pdb_seqres.txt"
RFAM_DB="$USER_ROOT/protenix_data/search_database_legacy/Rfam.fasta"
RNACENTRAL_DB="$USER_ROOT/protenix_data/search_database_legacy/rnacentral_active.fasta"
NTRNA_DB="$USER_ROOT/protenix_data/search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"

FOLDBENCH_SEEDS="42,66,101,2024,8888"
FOLDBENCH_SAMPLES="5"

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
  printf 'RUNNING\n' > "$REPORT_DIR/${stage}.exit_code"
  set +e
  "$@"
  local code=$?
  set -e
  printf '%s\n' "$code" > "$REPORT_DIR/${stage}.exit_code"
  return "$code"
}

case "${1:-}" in
  preflight)
    run_and_record preflight "$PYTHON" "$PIPELINE" preflight \
      "${COMMON_ARGS[@]}" "${DATABASE_ARGS[@]}"
    ;;
  audit)
    run_and_record audit "$PYTHON" "$PIPELINE" audit "${COMMON_ARGS[@]}"
    ;;
  prep)
    run_and_record prep "$PYTHON" "$PIPELINE" prep \
      "${COMMON_ARGS[@]}" "${DATABASE_ARGS[@]}" \
      --workers 4 --nhmmer-cpus 8
    ;;
  pred)
    run_and_record pred "$PYTHON" "$PIPELINE" pred \
      "${COMMON_ARGS[@]}" --gpus 0,1,2,3,4,5,6,7
    ;;
  *)
    echo "Usage: $0 {preflight|audit|prep|pred}" >&2
    exit 64
    ;;
esac
