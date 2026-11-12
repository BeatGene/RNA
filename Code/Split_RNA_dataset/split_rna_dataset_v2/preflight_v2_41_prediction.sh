#!/usr/bin/env bash
set -euo pipefail

link_report="${1:?Usage: bash preflight_v2_41_prediction.sh <completed V2 link report directory>}"
user_root="$HOME"
report_root="$user_root/Code/pipeline_reports"
split_report="$report_root/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"
prepare="$user_root/Code/predict_protenix/Version_3/data_v1_50x4_confidence/prepare_data_v1_confidence_run.py"
run_dir="$report_root/DATA_V2_41_PRED_PREFLIGHT_$(date -u +%Y%m%dT%H%M%SZ)"
seeds="$(seq -s, 300 349)"

python3 "$prepare" \
  --master-manifest "$report_root/PDB_RAW/pdb_cif_manifest.csv" \
  --split-manifest "$split_report/final_manifest.tsv" \
  --data-root "$user_root/Data_V2" \
  --simple-json-dir "$user_root/Json_data/Simple_json" \
  --complex-json-dir "$user_root/Json_data/Complex_json" \
  --run-dir "$run_dir" \
  --seeds "$seeds" \
  --samples 4 \
  --allow-variable-split-counts \
  --target-ids-file "$link_report/predict_all_pdb_ids.txt"

python3 - "$link_report/summary.json" "$run_dir/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

links = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
actions = links["actions"]
linked = actions.get("REUSE_LINKED", 0) + actions.get("REUSE_ALREADY_LINKED", 0)
if links["mode"] != "EXECUTE_LINKS" or linked != 964 or actions.get("NEW_PREDICTION_REQUIRED") != 41 or actions.get("SKIP_PREP_ABANDONED_LONG_CHAIN") != 10 or links["review_blockers"]:
    raise SystemExit(f"Incomplete V2 reuse links: {actions}")
actual = (summary["full_split_target_count"], summary["total_target_count"],
          summary["train_count"], summary["val_count"], summary["test_count"])
if actual != (1015, 41, 40, 0, 1):
    raise SystemExit(f"Unexpected V2 prediction preflight counts: {actual}")
print("V2 prediction preflight passed: full split=1015, scheduled=41 (train=40, val=0, test=1)")
PY

printf 'Preflight report directory: %s\n' "$run_dir"
