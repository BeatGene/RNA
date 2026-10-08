#!/usr/bin/env bash
set -euo pipefail

# Evaluate only the fixed epoch-49 checkpoint on globally ranked rank-1 PTs.
project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2_rank1}"
output_root="${EVAL_ROOT:-$project_root/evaluation/v3_rank1_epoch49_20261008}"
checkpoint="$project_root/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt"
config="$project_root/config/RNA_train_residual_v3.yaml"
targets="$project_root/config/foldbench_monomer_rna_targets.csv"
gpu_count="${EVAL_GPUS:-4}"
batch_size="${EVAL_BATCH_SIZE:-1}"

for path in "$data_root/summary.json" "$data_root/selection.tsv" "$checkpoint" "$config" \
  "$project_root/scripts/evaluate_refinement_v3.py" "$project_root/scripts/analyze_refinement_v3.py"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done
command -v python >/dev/null
command -v torchrun >/dev/null

for split in val test; do
  if [[ ! -d "$data_root/$split" ]]; then
    echo "missing rank-1 split: $data_root/$split" >&2
    exit 1
  fi
  actual="$(find "$data_root/$split" -name '*.pt' | wc -l)"
  expected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_count"][sys.argv[2]])' "$data_root/summary.json" "$split")"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "$split: expected $expected .pt links, found $actual" >&2
    exit 1
  fi
done
while IFS= read -r -d '' link; do
  if [[ ! -e "$link" ]]; then
    echo "broken rank-1 .pt link: $link" >&2
    exit 1
  fi
done < <(find "$data_root/val" "$data_root/test" -name '*.pt' -type l -print0)

mkdir -p "$output_root"
if [[ -f "$output_root/rank1_selection.sha256" ]]; then
  sha256sum -c "$output_root/rank1_selection.sha256" >/dev/null
else
  sha256sum "$data_root/selection.tsv" > "$output_root/rank1_selection.sha256"
fi

run_split() {
  local split="$1"
  local output_name="$2"
  local output_dir="$output_root/$output_name"
  local expected
  expected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_count"][sys.argv[2]])' "$data_root/summary.json" "$split")"

  if [[ -f "$output_dir/summary.json" ]]; then
    python -c 'import json,pathlib,sys; s=json.load(open(sys.argv[1])); assert s["split"]==sys.argv[2]; assert pathlib.Path(s["checkpoint"]).resolve()==pathlib.Path(sys.argv[3]).resolve(); assert pathlib.Path(s["data_dir"]).resolve()==pathlib.Path(sys.argv[4]).resolve(); assert s["candidate_count"]==int(sys.argv[5]); print("EVALUATION_EXISTS",sys.argv[2],sys.argv[1])' \
      "$output_dir/summary.json" "$split" "$checkpoint" "$data_root" "$expected"
    return
  fi
  if [[ -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]]; then
    echo "incomplete output exists; inspect and move it before retrying: $output_dir" >&2
    exit 1
  fi
  echo "EVALUATE_RANK1 split=$split checkpoint=$checkpoint samples=$expected"
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$checkpoint" \
    --data-dir "$data_root" --split "$split" \
    --output-dir "$output_dir" --batch-size "$batch_size" \
    --num-timesteps 1 --physics
}

run_split val val_final
run_split test test_locked

analysis_args=(
  --val-dir "$output_root/val_final"
  --test-dir "$output_root/test_locked"
  --output-dir "$output_root/analysis"
)
if [[ -f "$targets" ]]; then
  analysis_args+=(--foldbench-targets "$targets")
fi
python "$project_root/scripts/analyze_refinement_v3.py" "${analysis_args[@]}"
echo "RANK1_EPOCH49_COMPLETE output_root=$output_root"
