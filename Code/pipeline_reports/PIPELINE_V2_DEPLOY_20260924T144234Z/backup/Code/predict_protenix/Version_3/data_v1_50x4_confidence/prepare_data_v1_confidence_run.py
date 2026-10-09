#!/usr/bin/env python3
"""Build and audit split-specific manifests for the Data_V1 confidence run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 774, "val": 117, "test": 97}


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def pdb_id(value: object) -> str:
    return str(value).strip().upper()


def read_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"清单没有表头：{path}")
        return list(reader.fieldnames), list(reader)


def directory_ids(path: Path) -> set[str]:
    return {child.name.strip().upper() for child in path.iterdir() if child.is_dir()}


def parse_seeds(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--seeds 必须是不重复的逗号分隔整数")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--simple-json-dir", type=Path, required=True)
    parser.add_argument("--complex-json-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    for name in (
        "master_manifest",
        "split_manifest",
        "data_root",
        "simple_json_dir",
        "complex_json_dir",
        "run_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.samples < 1:
        raise ValueError("--samples 必须大于 0")

    master_fields, master_rows = read_rows(args.master_manifest, ",")
    _, split_rows = read_rows(args.split_manifest, "\t")
    for required in ("PDB_ID", "CURRENT_TARGET"):
        if required not in master_fields:
            raise ValueError(f"master manifest 缺少列：{required}")
    if not split_rows or not {"PDB_ID", "FINAL_SPLIT", "FINAL_STATUS"}.issubset(
        split_rows[0]
    ):
        raise ValueError("split manifest 缺少 PDB_ID/FINAL_SPLIT/FINAL_STATUS")

    current_rows: dict[str, dict[str, str]] = {}
    for row in master_rows:
        key = pdb_id(row.get("PDB_ID", ""))
        if key and truth(row.get("CURRENT_TARGET", "")):
            if key in current_rows:
                raise ValueError(f"master manifest 中 CURRENT_TARGET 重复：{key}")
            current_rows[key] = row

    split_ids: dict[str, set[str]] = {name: set() for name in SPLITS}
    for row in split_rows:
        split = str(row.get("FINAL_SPLIT", "")).strip().lower()
        status = str(row.get("FINAL_STATUS", "")).strip().upper()
        if split in split_ids and status == "KEPT":
            key = pdb_id(row.get("PDB_ID", ""))
            if not key:
                raise ValueError("KEPT 行含空 PDB_ID")
            split_ids[split].add(key)

    if any(split_ids[a] & split_ids[b] for a in SPLITS for b in SPLITS if a < b):
        raise ValueError("train/val/test 清单存在交集")

    suffix = "-final-updated.json"
    updated_ids = {
        path.name[: -len(suffix)].upper()
        for path in args.simple_json_dir.glob(f"*{suffix}")
    }
    prep_ids = {
        path.name[len("prep_output_") :].upper()
        for path in args.complex_json_dir.glob("prep_output_*")
        if path.is_dir()
    }

    args.run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "model_name": "protenix_base_default_v1.0.0",
        "seeds": args.seeds,
        "samples_per_seed": args.samples,
        "diffusion_steps": 200,
        "pairformer_cycles": 10,
        "dtype": "bf16",
        "need_atom_confidence": True,
        "required_full_data_keys": [
            "atom_plddt",
            "token_pair_pae",
            "token_pair_pde",
            "contact_probs",
            "atom_to_token_idx",
        ],
        "output_layout": "Data_V1/<split>/<pdb>/seed_<seed>/predictions",
    }

    total_targets = 0
    for split in SPLITS:
        expected = EXPECTED_COUNTS[split]
        selected = split_ids[split]
        if len(selected) != expected:
            raise ValueError(
                f"划分日志中 {split} 的 KEPT 数量为 {len(selected)}，预期 {expected}"
            )
        folders = directory_ids(args.data_root / split)
        if folders != selected:
            raise ValueError(
                f"Data_V1/{split} 与划分日志不一致："
                f"missing={sorted(selected - folders)[:20]} "
                f"extra={sorted(folders - selected)[:20]}"
            )
        missing_master = selected - set(current_rows)
        missing_updated = selected - updated_ids
        missing_prep = selected - prep_ids
        if missing_master or missing_updated or missing_prep:
            raise ValueError(
                f"{split} 输入不完整：missing_master={sorted(missing_master)[:20]} "
                f"missing_updated={sorted(missing_updated)[:20]} "
                f"missing_prep={sorted(missing_prep)[:20]}"
            )

        manifest_path = args.run_dir / f"{split}_50x4_confidence_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=master_fields)
            writer.writeheader()
            writer.writerows(current_rows[key] for key in sorted(selected))
        if split == "train":
            with (args.run_dir / "smoke_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=master_fields)
                writer.writeheader()
                writer.writerow(current_rows[sorted(selected)[0]])
        (args.run_dir / f"eligible_{split}_pdb_ids.txt").write_text(
            "\n".join(sorted(selected)) + "\n", encoding="utf-8"
        )

        target_count = len(selected)
        total_targets += target_count
        summary[f"{split}_count"] = target_count
        summary[f"expected_{split}_seed_tasks"] = target_count * len(args.seeds)
        summary[f"expected_{split}_decoys"] = (
            target_count * len(args.seeds) * args.samples
        )
        summary[f"expected_{split}_full_data_json"] = (
            target_count * len(args.seeds) * args.samples
        )

    summary["total_target_count"] = total_targets
    summary["expected_total_seed_tasks"] = total_targets * len(args.seeds)
    summary["expected_total_decoys"] = total_targets * len(args.seeds) * args.samples
    summary["expected_total_full_data_json"] = summary["expected_total_decoys"]
    (args.run_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
