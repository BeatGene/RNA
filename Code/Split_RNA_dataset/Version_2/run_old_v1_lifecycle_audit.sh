#!/usr/bin/env bash
set -euo pipefail

user_root="${HOME}"
split_report="$user_root/Code/pipeline_reports/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE"
pred_run="$user_root/Code/pipeline_reports/DATA_V1_50X4_CONFIDENCE/pred_runs/data_v1_50x4_conf_8gpu_20260908T072438Z"
pt_log="$user_root/Data_PT_V1/logs/write_v1_rmsd30_20260916"
output="$user_root/Code/pipeline_reports/PDB_LIFECYCLE_V1_REVIEW_20260924"

python3 "$user_root/Code/Split_RNA_dataset/Version_2/audit_pdb_lifecycle.py" \
  --split-report "$split_report" \
  --output-dir "$output" \
  --prediction-root "$user_root/Data_V1" \
  --pt-root "$user_root/Data_PT_V1" \
  --pred-audit "$pred_run/train/rounds/001/decoy_manifest.csv" \
  --pred-audit "$pred_run/val/rounds/001/decoy_manifest.csv" \
  --pred-audit "$pred_run/test/rounds/001/decoy_manifest.csv" \
  --pt-log-dir "$pt_log"

printf 'PDB lifecycle report: %s\n' "$output"
