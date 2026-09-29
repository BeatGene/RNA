#!/usr/bin/env bash
set -euo pipefail

new_run="${1:?Usage: bash run_v2_pt_final_audit.sh <new-train PT run> <retained PT run>}"
retained_run="${2:?Usage: bash run_v2_pt_final_audit.sh <new-train PT run> <retained PT run>}"
user_root="$HOME"
report="$user_root/Code/pipeline_reports/PT_V2_AUDIT_$(date -u +%Y%m%dT%H%M%SZ)"

python3 "$user_root/Code/Split_RNA_dataset/Version_2/audit_v2_pt_samples.py" \
  --split-report "$user_root/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE" \
  --prediction-root "$user_root/Data_V2" \
  --main-pt-root "$user_root/Data_PT_V2" \
  --high-pt-root "$user_root/Data_PT_V2_RMSD_GT30" \
  --exclude-pdb-file "$user_root/Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv" \
  --skip-pdb-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_upstream_skipped_v2.tsv" \
  --new-pt-run "$new_run" \
  --retained-pt-run "$retained_run" \
  --old-pt-run "$user_root/Data_PT_V1/logs/write_v1_rmsd30_20260916" \
  --output-dir "$report"

old_pred="$user_root/Code/pipeline_reports/DATA_V1_50X4_CONFIDENCE/pred_runs/data_v1_50x4_conf_8gpu_20260908T072438Z"
new_pred="$user_root/Code/pipeline_reports/DATA_V2_50X4_CONFIDENCE/pred_runs/data_v2_41_50x4_conf_8gpu_20260928T063400Z"
python3 "$user_root/Code/Split_RNA_dataset/Version_2/audit_pdb_lifecycle.py" \
  --split-report "$user_root/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE" \
  --prediction-root "$user_root/Data_V2" \
  --pt-root "$user_root/Data_PT_V2" \
  --high-pt-root "$user_root/Data_PT_V2_RMSD_GT30" \
  --prep-root "$user_root/Json_data/Complex_json" \
  --pred-audit "$old_pred/train/rounds/001/decoy_manifest.csv" \
  --pred-audit "$old_pred/val/rounds/001/decoy_manifest.csv" \
  --pred-audit "$old_pred/test/rounds/001/decoy_manifest.csv" \
  --pred-audit "$new_pred/train/rounds/001/decoy_manifest.csv" \
  --pred-audit "$new_pred/test/rounds/001/decoy_manifest.csv" \
  --pt-log-dir "$new_run" \
  --pt-log-dir "$retained_run" \
  --skip-pdb-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_upstream_skipped_v2.tsv" \
  --output-dir "$report/lifecycle"

python3 "$user_root/Code_Flow_matching/scripts/analyze_ranking_vs_rmsd.py" \
  --pt-run-dir "$report/combined_pt_log" \
  --output-dir "$report/ranking_vs_rmsd" \
  --rmsd-threshold 30

printf 'V2 PT audit report: %s\n' "$report"
