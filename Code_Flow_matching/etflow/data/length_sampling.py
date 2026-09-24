"""Build per-PT training weights from a versioned split audit table."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def long_rna_weights(
    data_files: list[Path],
    train_root: Path,
    length_table: Path,
    *,
    threshold_nt: int = 50,
    factor: float = 2.0,
) -> tuple[list[float], dict[str, int]]:
    if threshold_nt < 1 or not math.isfinite(factor) or factor < 1:
        raise ValueError("threshold_nt must be positive and factor must be finite and >= 1")
    lengths: dict[str, int] = {}
    with Path(length_table).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"PDB_ID", "RNA_LENGTH"}.issubset(reader.fieldnames):
            raise ValueError("length table must contain PDB_ID and RNA_LENGTH")
        for row in reader:
            pdb_id = row["PDB_ID"].strip().upper()
            if not pdb_id:
                continue
            raw = row["RNA_LENGTH"].strip()
            if not raw:
                continue
            length = int(raw)
            if length < 1 or (pdb_id in lengths and lengths[pdb_id] != length):
                raise ValueError(f"invalid or conflicting RNA length for {pdb_id}")
            lengths[pdb_id] = length
    weights: list[float] = []
    counts = {"long_pt": 0, "short_pt": 0, "long_pdb": 0, "short_pdb": 0}
    seen_long: set[str] = set()
    seen_short: set[str] = set()
    for path in data_files:
        relative = path.relative_to(train_root)
        if len(relative.parts) < 2:
            raise ValueError(f"PT file is not under a PDB directory: {path}")
        pdb_id = relative.parts[0].upper()
        if pdb_id not in lengths:
            raise ValueError(f"RNA length is missing for training PDB {pdb_id}")
        if lengths[pdb_id] > threshold_nt:
            weights.append(factor)
            counts["long_pt"] += 1
            seen_long.add(pdb_id)
        else:
            weights.append(1.0)
            counts["short_pt"] += 1
            seen_short.add(pdb_id)
    counts["long_pdb"] = len(seen_long)
    counts["short_pdb"] = len(seen_short)
    return weights, counts
