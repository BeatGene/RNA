#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2_rank1}"
output_root="${BACKBONE_EVAL_ROOT:-$project_root/evaluation/v3_rank1_epoch49_backbone_20261008}"
checkpoint="$project_root/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt"
config="$project_root/config/RNA_train_residual_v3.yaml"
gpu_count="${EVAL_GPUS:-4}"

for file in "$data_root/summary.json" "$data_root/selection.tsv" "$checkpoint" "$config" \
  "$project_root/scripts/evaluate_refinement_v3.py" \
  "$project_root/scripts/summarize_backbone_rmsd_v3.py"; do
  [[ -f "$file" ]] || { echo "missing file: $file" >&2; exit 1; }
done

mkdir -p "$output_root"
if [[ -f "$output_root/rank1_selection.sha256" ]]; then
  sha256sum -c "$output_root/rank1_selection.sha256" >/dev/null
else
  sha256sum "$data_root/selection.tsv" > "$output_root/rank1_selection.sha256"
fi

for split in val test; do
  [[ -d "$data_root/$split" ]] || { echo "missing split: $split" >&2; exit 1; }
  expected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_count"][sys.argv[2]])' "$data_root/summary.json" "$split")"
  actual="$(find "$data_root/$split" -name '*.pt' | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "$split: $actual .pt, expected $expected" >&2; exit 1; }
  while IFS= read -r -d '' link; do
    [[ -e "$link" ]] || { echo "broken .pt link: $link" >&2; exit 1; }
  done < <(find "$data_root/$split" -name '*.pt' -type l -print0)
  output_dir="$output_root/$split"
  if [[ -f "$output_dir/summary.json" ]]; then
    python - "$output_dir" "$expected" "$checkpoint" "$data_root" <<'PY'
import csv, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
summary = json.loads((directory / "summary.json").read_text())
with (directory / "samples.tsv").open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
assert summary["candidate_count"] == int(sys.argv[2]) == len(rows)
assert pathlib.Path(summary["checkpoint"]).resolve() == pathlib.Path(sys.argv[3]).resolve()
assert pathlib.Path(summary["data_dir"]).resolve() == pathlib.Path(sys.argv[4]).resolve()
assert "input_backbone_aligned_rmsd" in rows[0]
print("EVALUATION_EXISTS", directory)
PY
    continue
  fi
  if [[ -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]]; then
    echo "incomplete output exists; inspect before rerunning: $output_dir" >&2
    exit 1
  fi
  echo "BACKBONE_EVALUATE split=$split candidates=$expected"
  torchrun --standalone --nproc_per_node="$gpu_count" \
    "$project_root/scripts/evaluate_refinement_v3.py" \
    --config "$config" --checkpoint "$checkpoint" \
    --data-dir "$data_root" --split "$split" \
    --output-dir "$output_dir" --batch-size 1 --num-timesteps 1
done

python "$project_root/scripts/summarize_backbone_rmsd_v3.py" \
  --val-dir "$output_root/val" --test-dir "$output_root/test" \
  --output-dir "$output_root/backbone_analysis"
echo "BACKBONE_EVALUATION_COMPLETE output_root=$output_root"
