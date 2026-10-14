"""Validate and ZIP an existing completed PyMOL bundle, without inference.

Uses only the Python standard library. Old source timestamps are clamped
inside the ZIP; source files and any previous ZIP are left in place.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import zipfile


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_bundle(bundle):
    summary = json.loads((bundle / "bundle_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ValueError(f"bundle is not marked COMPLETE: {bundle}")
    manifest = read_tsv(bundle / "manifest.tsv")
    selected = read_tsv(bundle / "rank1_selection.tsv")
    expected = {(row["split"], row["pdb_id"].upper(), int(row["seed"]), int(row["sample"]))
                for row in selected}
    identities = {(row["split"], row["pdb_id"].upper(), int(row["protenix_seed"]), int(row["protenix_sample"]))
                  for row in manifest}
    if len(identities) != len(manifest) or identities != expected or len(expected) != len(selected):
        raise ValueError("manifest candidate identities differ from rank-1 selection")
    for split in ("val", "test"):
        rows = [row for row in manifest if row["split"] == split]
        if len(rows) != int(summary["counts"][split]) or len({row["pdb_id"].upper() for row in rows}) != len(rows):
            raise ValueError(f"missing/duplicate {split} PDBs")
    required = [bundle / name for name in (
        "rna_pymol_viewer.py", "README.md", "render_all.pml",
        "inspection_order_val.tsv", "inspection_order_test.tsv",
    )]
    for row in manifest:
        root = bundle / row["split"] / row["pdb_id"].upper()
        required.extend(root / mode / (name + ".cif")
                        for mode in ("all_atom", "backbone", "source_frame")
                        for name in ("native", "before", "after"))
        required.extend(root / name for name in (
            "atom_metrics.tsv", "bond_metrics.tsv", "residue_metrics.tsv", "display.json"))
        required.extend(bundle / row[field] for field in ("view_all_atom", "view_backbone"))
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty bundle file: {path}")
    print(f"BUNDLE_VALIDATED val={summary['counts']['val']} test={summary['counts']['test']} path={bundle}")


def write_bundle_zip(bundle, archive_path):
    """Publish ZIP only after writing and checking it; never overwrite output."""
    bundle, archive_path = Path(bundle), Path(archive_path)
    temporary = archive_path.with_name(archive_path.name + ".partial")
    if archive_path.exists() or temporary.exists():
        raise FileExistsError(f"choose a fresh ZIP path: {archive_path}")
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty bundle: {bundle}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=6,
                             strict_timestamps=False) as archive:
            for path in files:
                archive.write(path, path.relative_to(bundle.parent).as_posix())
        with zipfile.ZipFile(temporary, "r") as archive:
            if len(archive.infolist()) != len(files):
                raise ValueError("ZIP entry count differs from bundle")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP checksum failure: {bad}")
        temporary.rename(archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"PYMOL_DIAGNOSTICS_ZIP {archive_path}")


def latest_bundle(evaluation_root):
    candidates = sorted(evaluation_root.glob("rank1_pymol_diagnostics_*/pymol_bundle"))
    candidates = [path for path in candidates if (path / "bundle_summary.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no completed diagnostic bundle found below {evaluation_root}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, help="existing pymol_bundle directory")
    parser.add_argument("--evaluation-root", type=Path,
                        default=Path.home() / "Code_Flow_matching" / "evaluation")
    parser.add_argument("--output-zip", type=Path)
    args = parser.parse_args()
    bundle = (args.bundle_dir or latest_bundle(args.evaluation_root)).resolve()
    output = args.output_zip or bundle.with_name(bundle.name + "_repacked.zip")
    validate_bundle(bundle)
    write_bundle_zip(bundle, output.resolve())
    print(f"DOWNLOAD_THIS_ZIP {output.resolve()}")


if __name__ == "__main__":
    main()
