#!/usr/bin/env bash
set -euo pipefail

link_report="${1:?Usage: bash start_v2_41_prediction.sh <completed V2 link report directory>}"
user_root="$HOME"
split_report="$user_root/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_20260924T155344Z_EXECUTE"
worklist="$link_report/predict_all_pdb_ids.txt"

python3 - "$link_report/summary.json" "$worklist" "$split_report/final_manifest.tsv" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

summary_path, worklist_path, manifest_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
actions = summary["actions"]
linked = actions.get("REUSE_LINKED", 0) + actions.get("REUSE_ALREADY_LINKED", 0)
if summary["mode"] != "EXECUTE_LINKS" or linked != 964 or actions.get("NEW_PREDICTION_REQUIRED") != 41 or actions.get("SKIP_PREP_ABANDONED_LONG_CHAIN") != 10 or summary["review_blockers"]:
    raise SystemExit(f"V2 reuse link report is not complete: {summary_path}: {actions}")
ids = [line.strip().upper() for line in worklist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(ids) != 41 or len(set(ids)) != 41:
    raise SystemExit(f"Prediction worklist must contain 41 unique PDBs: {worklist_path}")
with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
    selected = {row["PDB_ID"]: row["FINAL_SPLIT"] for row in csv.DictReader(stream, delimiter="\t") if row["FINAL_STATUS"] == "KEPT"}
if not set(ids) <= selected.keys():
    raise SystemExit(f"Worklist contains PDBs absent from V2 split: {sorted(set(ids) - selected.keys())}")
counts = Counter(selected[pdb_id] for pdb_id in ids)
if counts != {"train": 40, "test": 1}:
    raise SystemExit(f"Unexpected V2 prediction worklist split counts: {dict(counts)}")
print(f"Verified V2 link report and 41-target prediction worklist: {dict(counts)}")
PY

export SPLIT_MANIFEST="$split_report/final_manifest.tsv"
export DATA_ROOT="$user_root/Data_V2"
export BASE_REPORT="$user_root/Code/pipeline_reports/DATA_V2_50X4_CONFIDENCE"
export ALLOW_VARIABLE_SPLIT_COUNTS=1
export TARGET_IDS_FILE="$worklist"
export SPLIT_ORDER="train test"
export RUN_ID="data_v2_41_50x4_conf_8gpu_$(date -u +%Y%m%dT%H%M%SZ)"

bash "$user_root/Code/predict_protenix/Version_3/data_v1_50x4_confidence/start_data_v1_50x4_confidence_8gpu.sh"
