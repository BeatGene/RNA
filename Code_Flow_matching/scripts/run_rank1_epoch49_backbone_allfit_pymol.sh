#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2}"
archive_root="${RANK1_ARCHIVE_ROOT:-$HOME/Data_PT_V2_all_candidates_20261013}"
selection_file="$archive_root/rank1_selection.tsv"
summary_file="$archive_root/rank1_summary.json"
prediction_root="${PREDICTION_ROOT:-$HOME/Data_V2}"
native_root="${NATIVE_ROOT:-$HOME/pdb_data}"
output_root="${BACKBONE_PYMOL_ROOT:-$project_root/evaluation/v3_rank1_epoch49_backbone_allfit_pymol_v2promoted_20261013}"
checkpoint="$project_root/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt"
config="$project_root/config/RNA_train_residual_v3.yaml"
gpu_count="${EVAL_GPUS:-4}"

for file in "$selection_file" "$summary_file" "$checkpoint" "$config" \
  "$project_root/scripts/evaluate_refinement_v3.py" \
  "$project_root/scripts/export_rank1_pymol_v3.py" \
  "$project_root/scripts/summarize_backbone_alignment_v3.py"; do
  [[ -f "$file" ]] || { echo "missing file: $file" >&2; exit 1; }
done
[[ -d "$prediction_root" ]] || { echo "missing prediction root: $prediction_root" >&2; exit 1; }
[[ -d "$native_root" ]] || { echo "missing native root: $native_root" >&2; exit 1; }
[[ ! -e "$output_root" ]] || { echo "output root exists; choose a fresh BACKBONE_PYMOL_ROOT: $output_root" >&2; exit 1; }

mkdir -p "$output_root"
selection_hash="$(sha256sum "$selection_file")"
selection_hash="${selection_hash%% *}"
printf '%s  %s\n' "$selection_hash" "$selection_file" > "$output_root/rank1_selection.sha256"

for split in val test; do
  [[ -d "$data_root/$split" ]] || { echo "missing split: $data_root/$split" >&2; exit 1; }
  expected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_count"][sys.argv[2]])' "$summary_file" "$split")"
  actual="$(find "$data_root/$split" -name '*.pt' | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "$split: $actual .pt, expected $expected" >&2; exit 1; }
  while IFS= read -r -d '' link; do
    [[ -e "$link" ]] || { echo "broken .pt link: $link" >&2; exit 1; }
  done < <(find "$data_root/$split" -name '*.pt' -type l -print0)
  echo "EVALUATE_AND_EXPORT split=$split candidates=$expected"
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$checkpoint" \
    --data-dir "$data_root" --split "$split" \
    --output-dir "$output_root/$split" --batch-size 1 --num-timesteps 1 \
    --prediction-root "$prediction_root" --native-root "$native_root" \
    --export-structures
done

python "$project_root/scripts/summarize_backbone_alignment_v3.py" \
  --val-dir "$output_root/val" --test-dir "$output_root/test" \
  --output-dir "$output_root/backbone_comparison"

python "$project_root/scripts/export_rank1_pymol_v3.py" \
  --val-dir "$output_root/val" --test-dir "$output_root/test" \
  --output-dir "$output_root/pymol_bundle" --zip

current_hash="$(sha256sum "$selection_file")"
[[ "${current_hash%% *}" == "$selection_hash" ]] || { echo "rank-1 selection changed" >&2; exit 1; }
echo "BACKBONE_ALLFIT_PYMOL_COMPLETE output_root=$output_root"
