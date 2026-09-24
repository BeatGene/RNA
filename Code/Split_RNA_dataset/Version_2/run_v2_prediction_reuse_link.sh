#!/usr/bin/env bash
set -euo pipefail

code_dir="$HOME/Code/Split_RNA_dataset/Version_2"
report_root="$HOME/Code/pipeline_reports"
run_dir="$report_root/DATA_V2_PRED_REUSE_AUDIT_$(date -u +%Y%m%dT%H%M%SZ)_LINKS_EXECUTE"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/run.log") 2>&1

printf 'Report directory: %s\n' "$run_dir"
python3 "$code_dir/audit_v2_prediction_reuse.py" \
  --old-split-report "$report_root/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE" \
  --new-split-report "$report_root/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE" \
  --prep-audit "$report_root/DATA_V2_PREP_20260924T161056Z/audit_before/decoy_manifest.csv" \
  --abandoned-file "$code_dir/prep_abandoned_long_chain_v2.tsv" \
  --old-pred-run "$report_root/DATA_V1_50X4_CONFIDENCE/pred_runs/data_v1_50x4_conf_8gpu_20260908T072438Z" \
  --old-data-root "$HOME/Data_V1" \
  --new-data-root "$HOME/Data_V2" \
  --output-dir "$run_dir" \
  --execute-links
