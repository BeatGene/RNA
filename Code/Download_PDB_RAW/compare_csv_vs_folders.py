"""
Compare PDB IDs in all_rna_structures.csv vs actual .cif files in classified folders.
Run on the Linux server:
  python compare_csv_vs_folders.py
"""
import os
import csv
import sys

# ── Config ─────────────────────────────────────────────
BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data"
CSV_PATH = os.path.join(BASE_DIR, "all_rna_structures.csv")
FOLDERS = [
    "01_Pure_RNA",
    "02_RNA_Protein_Complex",
    "03_Ribosome_Apo",
    "04_Ribosome_Bound_RNA",
    "05_Others_or_Failed",
]


def get_ids_from_csv(csv_path):
    """Extract unique PDB IDs from first column of CSV (case-insensitive, uppercased)."""
    ids = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if row and row[0].strip():
                ids.add(row[0].strip().upper())
    return ids


def get_ids_from_folders(base_dir, folders):
    """Scan .cif files in each folder, extract PDB ID from filename."""
    ids = set()
    folder_counts = {}
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  [WARN] Folder not found: {folder_path}")
            continue
        cif_files = [f for f in os.listdir(folder_path) if f.endswith(".cif")]
        folder_counts[folder] = len(cif_files)
        for f in cif_files:
            pdb_id = f.replace(".cif", "").upper()
            ids.add(pdb_id)
    return ids, folder_counts


def main():
    print("=" * 60)
    print("PDB ID Cross-check: CSV vs Classified .cif Files")
    print("=" * 60)

    # 1. Read CSV
    print(f"\n[1] Reading CSV: {CSV_PATH}")
    csv_ids = get_ids_from_csv(CSV_PATH)
    print(f"    Unique PDB IDs in CSV: {len(csv_ids)}")

    # 2. Scan folders
    print(f"\n[2] Scanning folders under: {BASE_DIR}")
    cif_ids, folder_counts = get_ids_from_folders(BASE_DIR, FOLDERS)
    print(f"    Unique PDB IDs in .cif files: {len(cif_ids)}")
    for folder, count in folder_counts.items():
        print(f"      {folder}: {count} files")

    # 3. Compare
    print(f"\n[3] Comparison")
    print("=" * 60)

    in_csv_not_cif = csv_ids - cif_ids
    in_cif_not_csv = cif_ids - csv_ids
    in_both = csv_ids & cif_ids

    print(f"\n  In both CSV and folders:        {len(in_both)}")
    print(f"  In CSV but NOT in any folder:   {len(in_csv_not_cif)}")
    print(f"  In folders but NOT in CSV:      {len(in_cif_not_csv)}")

    # 4. Detail
    if in_csv_not_cif:
        print(f"\n  ── IDs in CSV but missing .cif ──")
        for pid in sorted(in_csv_not_cif):
            print(f"    {pid}")

    if in_cif_not_csv:
        print(f"\n  ── IDs in folders but missing from CSV ──")
        for pid in sorted(in_cif_not_csv):
            print(f"    {pid}")

    # 5. Summary
    print(f"\n[4] Summary")
    print("=" * 60)
    total_csv = len(csv_ids)
    total_cif = len(cif_ids)
    print(f"  CSV unique IDs:   {total_csv}")
    print(f"  CIF unique IDs:   {total_cif}")
    print(f"  Match rate:       {len(in_both) / max(total_csv, 1) * 100:.1f}% (CSV -> CIF)")
    print(f"  Orphan CSV rows:  {len(in_csv_not_cif)} IDs (these have analysis rows but no .cif)")
    print(f"  Orphan CIF files: {len(in_cif_not_csv)} IDs (these have .cif but no analysis row)")
    print()


if __name__ == "__main__":
    main()
