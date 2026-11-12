#!/usr/bin/env bash
set -euo pipefail

plan="${1:?Usage: bash run_v2_pt_new_train.sh <V2 PT plan directory>}"
user_root="$HOME"
pred_run="$user_root/Code/pipeline_reports/DATA_V2_50X4_CONFIDENCE/pred_runs/data_v2_41_50x4_conf_8gpu_20260928T063400Z"
run_name="step1_new_train_$(date -u +%Y%m%dT%H%M%SZ)"

python3 - "$plan/summary.json" "$pred_run/train/rounds/001/summary.json" <<'PY'
import json
import sys
from pathlib import Path
plan, prediction = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
if (plan['new_train_eligible'], plan['retained_eligible'], plan['blockers']) != (40, 908, 0):
    raise SystemExit(f"unexpected V2 PT plan: {plan}")
if (prediction['target_count'], prediction['valid_decoy_count'],
        prediction['expected_decoy_count'], prediction['all_complete'],
        prediction['need_atom_confidence']) != (40, 8000, 8000, True, True):
    raise SystemExit(f"new-train prediction audit is incomplete: {prediction}")
PY

mapfile -t ids < "$plan/new_train_pdb_ids.txt"
[[ "${#ids[@]}" == 40 ]] || { echo "Expected 40 new train IDs" >&2; exit 2; }
python3 - "$plan/new_train_pdb_ids.txt" "$user_root/pdb_data" "$user_root/Data_FM/RNA_FM_embeddings" <<'PY'
import sys
from pathlib import Path
ids = [line.strip().upper() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
native, fm = map(Path, sys.argv[2:])
missing_native = [pdb for pdb in ids if not (native / f"{pdb.lower()}.cif").is_file()]
missing_fm = [pdb for pdb in ids if not (fm / pdb / "rnafm_t12_residue_embeddings.pt").is_file()]
if missing_native or missing_fm:
    raise SystemExit(f"PT prerequisites missing: native={missing_native}, RNA-FM={missing_fm}")
print(f"New-train PT prerequisites present for {len(ids)} PDBs")
PY
echo "PT run directory: $user_root/Code/pipeline_reports/PT_V2/$run_name"
python3 "$user_root/Code_Flow_matching/scripts/build_refinement_pt.py" \
  --prediction-root "$user_root/Data_V2" \
  --native-root "$user_root/pdb_data" \
  --rnafm-root "$user_root/Data_FM/RNA_FM_embeddings" \
  --output-root "$user_root/Data_PT_V2" \
  --high-rmsd-root "$user_root/Data_PT_V2_RMSD_GT30" \
  --exclude-pdb-file "$user_root/Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv" \
  --skip-pdb-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_upstream_skipped_v2.tsv" \
  --log-dir "$user_root/Code/pipeline_reports/PT_V2" \
  --run-name "$run_name" \
  --ccd-components-file "$user_root/protenix_data/common/components.cif" \
  --max-pre-refinement-rmsd 30 \
  --expected-samples 8000 \
  --split train \
  --pdb-id "${ids[@]}"
