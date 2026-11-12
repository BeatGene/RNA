#!/usr/bin/env bash
set -euo pipefail

mode="${1:-dry-run}"
case "$mode" in
  dry-run) suffix=DRYRUN; extra=() ;;
  execute) suffix=EXECUTE; extra=(--execute-hardlinks) ;;
  *) echo "Usage: bash run_v2_pt_reuse_plan.sh dry-run | execute <dry-run report> <new-train PT run>" >&2; exit 2 ;;
esac
user_root="$HOME"
if [[ "$mode" == execute ]]; then
  dry_run_report="${2:?execute requires the completed PT reuse dry-run report directory}"
  new_pt_run="${3:?execute requires the completed new-train PT run directory}"
  python3 "$user_root/Code/Split_RNA_dataset/Version_2/validate_v2_new_train_pt.py" \
    --new-pt-run "$new_pt_run" \
    --new-pdb-ids "$dry_run_report/new_train_pdb_ids.txt" \
    --unusable-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_unusable_new_train_v2.tsv"
fi
report="$user_root/Code/pipeline_reports/PT_V2_REUSE_$(date -u +%Y%m%dT%H%M%SZ)_$suffix"
echo "PT reuse report: $report"
python3 "$user_root/Code/Split_RNA_dataset/Version_2/plan_v2_pt_reuse.py" \
  --old-split-report "$user_root/Code/pipeline_reports/DATA_SPLIT_V1_SINGLECHAIN_RMSD15A_20260826T100510Z_EXECUTE" \
  --new-split-report "$user_root/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE" \
  --old-pt-root "$user_root/Data_PT_V1" \
  --old-pt-run "$user_root/Data_PT_V1/logs/write_v1_rmsd30_20260916" \
  --new-pt-root "$user_root/Data_PT_V2" \
  --exclude-pdb-file "$user_root/Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv" \
  --skip-pdb-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_upstream_skipped_v2.tsv" \
  --output-dir "$report" \
  "${extra[@]}"
