#!/usr/bin/env python3
"""Rebuild the two V2 PT upload bundles from the local workspace."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


VERSION = Path(__file__).resolve().parent
WORKSPACE = VERSION.parents[2]
BUNDLES = {
    "V2_PT_CODE_20260929.zip": {
        "scripts/build_refinement_pt.py": WORKSPACE / "Code_Flow_matching/scripts/build_refinement_pt.py",
        "scripts/analyze_ranking_vs_rmsd.py": WORKSPACE / "Code_Flow_matching/scripts/analyze_ranking_vs_rmsd.py",
        "config/refinement_excluded_pdb_ids_v2.tsv": WORKSPACE / "Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv",
    },
    "V2_PT_WORKFLOW_20260929.zip": {
        name: VERSION / name
        for name in (
            "pt_upstream_skipped_v2.tsv", "plan_v2_pt_reuse.py",
            "audit_v2_pt_samples.py", "audit_pdb_lifecycle.py",
            "run_v2_pt_reuse_plan.sh", "run_v2_pt_new_train.sh",
            "run_v2_pt_retained.sh", "run_v2_pt_final_audit.sh",
            "V2_PT_RECOVERY_20260929.md",
        )
    },
}


def main() -> None:
    for bundle_name, files in BUNDLES.items():
        destination = VERSION / bundle_name
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            for member, source in files.items():
                if not source.is_file():
                    raise FileNotFoundError(source)
                archive.write(source, member)
        with ZipFile(destination) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == set(files)
        print(destination)


if __name__ == "__main__":
    main()
