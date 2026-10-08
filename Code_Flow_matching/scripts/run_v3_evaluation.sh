#!/usr/bin/env bash
set -euo pipefail

# Run from the activated protenix-1.0.5 environment on the laboratory server.
# screen: compare checkpoints on validation, without physical-loss overhead.
# finalize: evaluate the chosen checkpoint on validation and locked test.

stage="${1:-}"
if [[ "$stage" != "screen" && "$stage" != "finalize" ]]; then
  echo "usage: bash scripts/run_v3_evaluation.sh screen|finalize" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
checkpoint_root="${CHECKPOINT_ROOT:-$project_root/checkpoints/rna_refinement_v3_residual_mobility}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2}"
output_root="${EVAL_ROOT:-$project_root/evaluation/v3_20261008}"
config="$project_root/config/RNA_train_residual_v3.yaml"
targets="${FOLDBENCH_TARGETS:-$project_root/config/foldbench_monomer_rna_targets.csv}"
gpu_count="${EVAL_GPUS:-4}"
batch_size="${EVAL_BATCH_SIZE:-1}"

for path in "$config" "$project_root/scripts/evaluate_refinement_v3.py" "$targets"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done
for split in val test; do
  if [[ ! -d "$data_root/$split" ]]; then
    echo "missing data split: $data_root/$split" >&2
    exit 1
  fi
done
command -v python >/dev/null
command -v torchrun >/dev/null

if [[ "$stage" == "screen" ]]; then
  mkdir -p "$output_root/screen"
  if [[ -f "$output_root/code_sha256.txt" ]]; then
    sha256sum -c "$output_root/code_sha256.txt"
  else
    sha256sum \
    "$project_root/scripts/evaluate_refinement.py" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    "$project_root/scripts/analyze_refinement_v3.py" \
    "$project_root/scripts/select_refinement_v3_checkpoint.py" \
    "$project_root/scripts/export_foldbench_v3.py" \
    "$project_root/scripts/summarize_foldbench_v3.py" \
    "$config" > "$output_root/code_sha256.txt"
  fi
  if [[ ! -f "$output_root/preflight_val_8/summary.json" ]]; then
    echo "PREFLIGHT_VALIDATION_8 checkpoint=epoch=88-step=280439.ckpt"
    torchrun --standalone --nproc_per_node="$gpu_count" \
      "$project_root/scripts/evaluate_refinement_v3.py" \
      --config "$config" \
      --checkpoint "$checkpoint_root/epoch=88-step=280439.ckpt" \
      --data-dir "$data_root" --split val \
      --output-dir "$output_root/preflight_val_8" \
      --batch-size "$batch_size" --num-timesteps 1 --limit 8 --physics
  fi
  for name in \
    'epoch=49-step=157550.ckpt' \
    'epoch=68-step=217419.ckpt' \
    'epoch=69-step=220570.ckpt' \
    'epoch=83-step=264684.ckpt' \
    'epoch=88-step=280439.ckpt' \
    'last.ckpt' \
    'last-v1.ckpt'; do
    checkpoint="$checkpoint_root/$name"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    stem="${name%.ckpt}"
    if [[ -f "$output_root/screen/$stem/summary.json" ]]; then
      echo "VALIDATION_SCREEN_SKIP completed=$stem"
      continue
    fi
    echo "VALIDATION_SCREEN checkpoint=$checkpoint"
    torchrun --standalone --nproc_per_node="$gpu_count" \
      "$project_root/scripts/evaluate_refinement_v3.py" \
      --config "$config" --checkpoint "$checkpoint" \
      --data-dir "$data_root" --split val \
      --output-dir "$output_root/screen/$stem" \
      --batch-size "$batch_size" --num-timesteps 1
  done
  python "$project_root/scripts/select_refinement_v3_checkpoint.py" \
    --screen-root "$output_root/screen" --output-dir "$output_root"
  echo "SCREEN_COMPLETE output_root=$output_root"
  exit 0
fi

if [[ ! -f "$output_root/selected_checkpoint.txt" ]]; then
  echo "run screen first; selected checkpoint file is missing" >&2
  exit 1
fi
selected_checkpoint="${SELECTED_CHECKPOINT:-$(<"$output_root/selected_checkpoint.txt")}"
if [[ ! -f "$selected_checkpoint" ]]; then
  echo "selected checkpoint does not exist: $selected_checkpoint" >&2
  exit 1
fi
for split_dir in val_final test_locked; do
  if [[ -f "$output_root/$split_dir/summary.json" ]]; then
    recorded_checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$output_root/$split_dir/summary.json")"
    if [[ "$recorded_checkpoint" != "$selected_checkpoint" ]]; then
      echo "$split_dir was evaluated with another checkpoint: $recorded_checkpoint" >&2
      exit 1
    fi
  fi
done
if [[ ! -f "$output_root/val_final/summary.json" ]]; then
  echo "FINAL_VALIDATION checkpoint=$selected_checkpoint"
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$selected_checkpoint" \
    --data-dir "$data_root" --split val \
    --output-dir "$output_root/val_final" \
    --batch-size "$batch_size" --num-timesteps 1 --physics
fi

if [[ ! -f "$output_root/test_locked/summary.json" ]]; then
  echo "LOCKED_TEST checkpoint=$selected_checkpoint"
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$selected_checkpoint" \
    --data-dir "$data_root" --split test \
    --output-dir "$output_root/test_locked" \
    --batch-size "$batch_size" --num-timesteps 1 --physics
fi

python "$project_root/scripts/analyze_refinement_v3.py" \
  --val-dir "$output_root/val_final" --test-dir "$output_root/test_locked" \
  --foldbench-targets "$targets" --output-dir "$output_root/analysis"

if [[ "${FOLDBENCH_EXPORT:-1}" == "1" && ! -f "$output_root/foldbench/selected_candidates.tsv" ]]; then
  python "$project_root/scripts/export_foldbench_v3.py" \
    --test-dir "$output_root/test_locked" --targets "$targets" \
    --config "$config" --checkpoint "$selected_checkpoint" \
    --data-dir "$data_root" --output-dir "$output_root/foldbench"
fi
echo "FINALIZE_COMPLETE output_root=$output_root"
