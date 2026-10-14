#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$HOME/Code_Flow_matching}"
output_root="${MEAN_FIX_OUTPUT:-$project_root/evaluation/rank1_fp32mean_fix_$(date +%Y%m%d_%H%M%S)}"
checkpoint="${CHECKPOINT:-$project_root/checkpoints/rna_refinement_v3_residual_mobility/epoch=49-step=157550.ckpt}"
config="${CONFIG:-$project_root/config/RNA_train_residual_v3.yaml}"
data_root="${DATA_ROOT:-$HOME/Data_PT_V2}"
start_at="${START_AT:-checks}"
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
[[ "$start_at" == checks || "$start_at" == smoke ]] || { echo "START_AT must be checks or smoke" >&2; exit 1; }
[[ ! -e "$output_root" ]] || { echo "choose a fresh MEAN_FIX_OUTPUT: $output_root" >&2; exit 1; }
[[ -f "$checkpoint" && -f "$config" ]] || { echo "missing checkpoint/config" >&2; exit 1; }
sha256sum -c scripts/FP32_GRAPH_MEAN_SHA256SUMS
for script_name in check_bf16_graph_mean.py smoke_test_synthetic.py probe_rank1_mobility_gradients.py; do
  [[ -f "scripts/$script_name" ]] || { echo "missing scripts/$script_name" >&2; exit 1; }
done
if [[ "${FULL_EVAL:-0}" == 1 ]]; then
  [[ -f scripts/evaluate_refinement_v3.py ]] || { echo "missing scripts/evaluate_refinement_v3.py" >&2; exit 1; }
fi
mkdir -p "$output_root"
cp scripts/FP32_GRAPH_MEAN_SHA256SUMS "$output_root/code.sha256"
cp "$config" "$output_root/config_snapshot.yaml"
sha256sum "$checkpoint" > "$output_root/checkpoint.sha256"
printf 'START_AT=%s\n' "$start_at" | tee "$output_root/run_stage.txt"
if [[ "$start_at" == checks ]]; then
  python scripts/check_bf16_graph_mean.py --device cuda --assert-fixed | tee "$output_root/numerical_check.log"
  for test_name in test_fp32_graph_reductions test_probe_rank1_mobility_gradients test_masked_rms test_vector_output_amp test_plane_loss_amp; do
    python -m unittest discover -s scripts -p "$test_name.py" -v 2>&1 | tee "$output_root/$test_name.log"
  done
fi
python scripts/smoke_test_synthetic.py --device cuda --bf16 2>&1 | tee "$output_root/synthetic_smoke.log"
for precision in bf16 fp32; do
  python scripts/probe_rank1_mobility_gradients.py \
    --pdb 8JHP 9IOS 7POF 7ZJ4 9C6J --parameter-gradients \
    --precision "$precision" --checkpoint "$checkpoint" --config "$config" --data-root "$data_root" \
    --output "$output_root/probe_$precision" 2>&1 | tee "$output_root/probe_$precision.log"
done
if [[ "${FULL_EVAL:-0}" == 1 ]]; then
  for split in val test; do
    python scripts/evaluate_refinement_v3.py \
      --config "$config" --checkpoint "$checkpoint" --data-dir "$data_root" \
      --split "$split" --output-dir "$output_root/eval_$split" \
      --batch-size 1 --num-timesteps 1 --precision bf16 --progress-every 5 --physics \
      2>&1 | tee "$output_root/eval_$split.log"
  done
fi
echo "FP32_MEAN_CHECKS_COMPLETE $output_root"
echo "DOWNLOAD_THIS_DIRECTORY $output_root"
