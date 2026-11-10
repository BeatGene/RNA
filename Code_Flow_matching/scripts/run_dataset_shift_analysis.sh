#!/usr/bin/env bash
# CPU only; reads existing data, writes only to a fresh analysis directory.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
OUT="${1:-$PROJECT_ROOT/evaluation/dataset_shift_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$HOME/Data_PT_V2}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/Data_PT_V2_all_candidates_20261013}"
LENGTH_TABLE="${LENGTH_TABLE:-$HOME/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE/selection_audit.tsv}"
python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib; print('Dataset analysis dependencies OK; CUDA is not required.')"
args=(--data-root "$DATA_ROOT" --archive-root "$ARCHIVE_ROOT"
      --prediction-root "${PREDICTION_ROOT:-$HOME/Data_V2}" --output-dir "$OUT"
      --workers "${WORKERS:-4}" --long-factor "${LONG_FACTOR:-2.0}"
      --long-threshold "${LONG_THRESHOLD:-50}" --bootstrap "${BOOTSTRAP:-500}")
if [[ "${INCLUDE_ARCHIVE:-1}" == 1 && -d "$ARCHIVE_ROOT" ]]; then
    args+=(--include-archive)
fi
if [[ -f "$ARCHIVE_ROOT/rank1_selection.tsv" ]]; then
    args+=(--score-table "$ARCHIVE_ROOT/rank1_selection.tsv")
fi
if [[ -f "$LENGTH_TABLE" ]]; then
    args+=(--train-length-table "$LENGTH_TABLE")
else
    echo "WARNING: training length audit missing; effective exposure will use PT sequence length."
fi
if [[ -n "${TRAIN_SCORE_TABLE:-}" ]]; then
    args+=(--score-table "$TRAIN_SCORE_TABLE")
fi
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ "${UMAP:-0}" == 1 ]]; then args+=(--umap); fi
if [[ "${REPORT_ONLY:-0}" == 1 ]]; then args+=(--report-only); fi
if [[ -n "${LIMIT_PER_PDB:-}" ]]; then args+=(--limit-per-pdb "$LIMIT_PER_PDB"); fi
mkdir -p "$OUT"
printf 'OUTPUT %s\n' "$OUT"
python -u scripts/analyze_dataset_shift.py "${args[@]}" 2>&1 | tee "$OUT/run.log"
printf 'DONE: download %s\n' "$OUT"
