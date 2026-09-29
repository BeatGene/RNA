#!/usr/bin/env bash
set -euo pipefail

plan="${1:?Usage: bash run_v2_pt_retained.sh <V2 PT hard-link report directory>}"
new_run="${2:?Usage: bash run_v2_pt_retained.sh <V2 PT hard-link report directory> <completed new-train PT run directory>}"
user_root="$HOME"
run_name="step2_retained_$(date -u +%Y%m%dT%H%M%SZ)"

python3 - "$plan/summary.json" <<'PY'
import json
import sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
if (plan['mode'], plan['new_train_eligible'], plan['retained_eligible'],
        plan['blockers']) != ('EXECUTE_HARDLINKS', 40, 908, 0):
    raise SystemExit(f"V2 PT hard-link plan is incomplete: {plan}")
if plan['created_hardlinks'] + plan['already_hardlinked'] != plan['old_main_pt_files']:
    raise SystemExit(f"old PT reuse count mismatch: {plan}")
PY

python3 "$user_root/Code/Split_RNA_dataset/Version_2/validate_v2_new_train_pt.py" \
  --new-pt-run "$new_run" \
  --new-pdb-ids "$plan/new_train_pdb_ids.txt" \
  --unusable-file "$user_root/Code/Split_RNA_dataset/Version_2/pt_unusable_new_train_v2.tsv"

mapfile -t ids < "$plan/retained_eligible_pdb_ids.txt"
[[ "${#ids[@]}" == 908 ]] || { echo "Expected 908 retained eligible IDs" >&2; exit 2; }
python3 - "$plan/retained_eligible_pdb_ids.txt" "$user_root/pdb_data" "$user_root/Data_FM/RNA_FM_embeddings" <<'PY'
import sys
from pathlib import Path
ids = [line.strip().upper() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
native, fm = map(Path, sys.argv[2:])
missing_native = [pdb for pdb in ids if not (native / f"{pdb.lower()}.cif").is_file()]
missing_fm = [pdb for pdb in ids if not (fm / pdb / "rnafm_t12_residue_embeddings.pt").is_file()]
if missing_native or missing_fm:
    raise SystemExit(f"PT prerequisites missing: native={missing_native}, RNA-FM={missing_fm}")
print(f"Retained PT prerequisites present for {len(ids)} PDBs")
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
  --expected-samples 181600 \
  --pdb-id "${ids[@]}"
