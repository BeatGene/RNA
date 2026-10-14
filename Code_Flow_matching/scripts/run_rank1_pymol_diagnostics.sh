#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2}"
archive_root="${RANK1_ARCHIVE_ROOT:-$HOME/Data_PT_V2_all_candidates_20261013}"
selection_file="${SELECTION_FILE:-$archive_root/rank1_selection.tsv}"
prediction_root="${PREDICTION_ROOT:-$HOME/Data_V2}"
native_root="${NATIVE_ROOT:-$HOME/pdb_data}"
output_root="${PYMOL_DIAG_ROOT:-$project_root/evaluation/rank1_pymol_diagnostics_$(date +%Y%m%d_%H%M%S)}"
checkpoint="${CHECKPOINT:-$project_root/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt}"
config="${CONFIG:-$project_root/config/RNA_train_residual_v3.yaml}"
gpu_count="${EVAL_GPUS:-1}"
expected_val="${EXPECTED_VAL:-103}"
expected_test="${EXPECTED_TEST:-67}"

for file in "$selection_file" "$checkpoint" "$config" \
  "$project_root/scripts/evaluate_refinement_v3.py" \
  "$project_root/scripts/export_rank1_pymol_v3.py" \
  "$project_root/scripts/build_rank1_pymol_diagnostics.py" \
  "$project_root/scripts/repack_rank1_pymol_bundle.py" \
  "$project_root/scripts/rna_pymol_viewer.py" \
  "$project_root/scripts/RANK1_PYMOL_DIAGNOSTICS.md"; do
  [[ -f "$file" ]] || { echo "missing file: $file" >&2; exit 1; }
done
[[ -d "$prediction_root" && -d "$native_root" ]] || { echo "missing source data roots" >&2; exit 1; }
[[ ! -e "$output_root" ]] || { echo "choose a fresh PYMOL_DIAG_ROOT: $output_root" >&2; exit 1; }

python "$project_root/scripts/build_rank1_pymol_diagnostics.py" \
  --check-data "$data_root" --selection-file "$selection_file" \
  --prediction-root "$prediction_root" --native-root "$native_root" \
  --expected-val "$expected_val" --expected-test "$expected_test"
[[ "${CHECK_ONLY:-0}" != 1 ]] || exit 0

mkdir -p "$output_root"
cp "$selection_file" "$output_root/selection_snapshot.tsv"
sha256sum "$selection_file" > "$output_root/rank1_selection.sha256"
sha256sum "$checkpoint" > "$output_root/checkpoint.sha256"
cp "$config" "$output_root/config_snapshot.yaml"
printf 'OUTPUT_ROOT=%s\n' "$output_root"

for split in val test; do
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$checkpoint" --data-dir "$data_root" \
    --split "$split" --output-dir "$output_root/$split" \
    --batch-size 1 --num-timesteps 1 --precision bf16 --progress-every 5 \
    --prediction-root "$prediction_root" --native-root "$native_root" \
    --physics --export-structures
done

cmp "$selection_file" "$output_root/selection_snapshot.tsv"
python "$project_root/scripts/build_rank1_pymol_diagnostics.py" \
  --val-dir "$output_root/val" --test-dir "$output_root/test" \
  --selection-file "$output_root/selection_snapshot.tsv" \
  --expected-val "$expected_val" --expected-test "$expected_test" \
  --output-dir "$output_root/pymol_bundle" --zip

echo "DOWNLOAD_THIS_ZIP $output_root/pymol_bundle.zip"
