"""Export the pre-ranked FoldBench RNA monomer subset as input/refined CIFs.

This never chooses candidates by native structure accuracy. Each target uses
the original Protenix ranking_score in its summary confidence JSON, or the
PT build manifest if that JSON was subsequently moved or deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi
import torch
from torch_geometric.data import Batch

import evaluate_refinement as base
from etflow.data.dataset import EuclideanDataset


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_source_cif(recorded: Path, prediction_root: Path, pdb_id: str,
                       seed: int) -> Path:
    """Find a prediction moved from the old val tree into the test links."""
    if recorded.is_file():
        return recorded
    candidates = (
        prediction_root / "test" / pdb_id.lower() / f"seed_{seed}" / "predictions" / recorded.name,
        prediction_root / "val" / pdb_id.lower() / f"seed_{seed}" / "predictions" / recorded.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Original predicted CIF is missing: {recorded}; also searched "
        + ", ".join(str(path) for path in candidates)
    )


def ranking_score_from_source(source_cif: Path, sample_number: int) -> tuple[float, Path]:
    """Read the summary JSON paired with a Protenix sample CIF.

    The .pt source_confidence_json points to full_data JSON, which contains
    atom-wise confidence but does not contain ranking_score.
    """
    suffix = f"_sample_{sample_number}.cif"
    if not source_cif.name.endswith(suffix):
        raise ValueError(f"Sample CIF name does not match sample {sample_number}: {source_cif}")
    prefix = source_cif.name[:-len(suffix)]
    score_path = source_cif.with_name(
        f"{prefix}_summary_confidence_sample_{sample_number}.json"
    )
    try:
        score_data = json.loads(score_path.read_text(encoding="utf-8"))
        score = float(score_data["ranking_score"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Missing or invalid Protenix ranking_score: {score_path}") from exc
    if not math.isfinite(score):
        raise ValueError(f"Nonfinite Protenix ranking_score: {score_path}")
    return score, score_path


def read_manifest_scores(paths: list[Path], target_pdb_ids: set[str]) -> dict[str, tuple[float, Path]]:
    """Load original ranking scores captured when the .pt dataset was built."""
    scores: dict[str, tuple[float, Path]] = {}
    for path in sorted(set(paths)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pdb_id = (row.get("pdb_id") or "").upper()
                if pdb_id not in target_pdb_ids or not row.get("ranking_score"):
                    continue
                try:
                    key = str(Path(row["output"]).resolve())
                    if not row["output"] or not Path(row["output"]).is_absolute():
                        raise ValueError("manifest output path must be absolute")
                    score = float(row["ranking_score"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid ranking score in {path}: {row}") from exc
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite ranking score in {path}: {row}")
                previous = scores.get(key)
                if previous is not None and not math.isclose(previous[0], score, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Conflicting build-manifest ranking scores for {key}: {previous[1]} and {path}")
                scores[key] = score, path
    return scores


def select_candidates(samples: list[dict], target_rows: list[dict],
                      manifest_scores: dict[str, tuple[float, Path]] | None = None,
                      prediction_root: Path | None = None) -> list[dict]:
    target_map = {
        (row["pdb_id"].split("-", 1)[0].upper(), row["chain_id"]): row
        for row in target_rows
    }
    selected: dict[str, dict] = {}
    for sample in samples:
        key = sample["pdb_id"].upper(), sample["predicted_chain_id"]
        target = target_map.get(key)
        if target is None:
            continue
        pt_path = Path(sample["sample_path"])
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        sample_number = int(sample["protenix_sample"])
        seed = int(sample["protenix_seed"])
        recorded_cif = Path(payload["source_predicted_cif"])
        source_cif = resolve_source_cif(
            recorded_cif, prediction_root or Path.home() / "Data_V2", key[0], seed,
        )
        summary_suffix = f"_sample_{sample_number}.cif"
        if not source_cif.name.endswith(summary_suffix):
            raise ValueError(f"Sample CIF name does not match sample {sample_number}: {source_cif}")
        score_path = source_cif.with_name(
            f"{source_cif.name[:-len(summary_suffix)]}_summary_confidence_sample_{sample_number}.json"
        )
        if score_path.is_file():
            ranking_score, score_path = ranking_score_from_source(source_cif, sample_number)
            score_source = str(score_path)
        else:
            manifest_entry = (manifest_scores or {}).get(str(pt_path.resolve()))
            if manifest_entry is None:
                raise ValueError(
                    f"Missing Protenix ranking_score for {key[0]} seed={seed} sample={sample_number}; "
                    f"summary JSON absent: {score_path}; no score for this exact .pt path in the PT build manifest"
                )
            ranking_score, manifest_path = manifest_entry
            score_source = str(manifest_path)
        candidate = {
            "foldbench_target_id": target["pdb_id"],
            "pdb_id": key[0],
            "native_chain_id": sample["native_chain_id"],
            "predicted_chain_id": key[1],
            "sample_path": str(pt_path),
            "source_cif": str(source_cif),
            "ranking_score_json": str(score_path) if score_path.is_file() else "",
            "ranking_score_source": score_source,
            "ranking_score": ranking_score,
            "seed": seed,
            "sample": sample_number,
            "input_rmsd_a": float(sample["input_aligned_rmsd"]),
            "refined_rmsd_a": float(sample["refined_aligned_rmsd"]),
        }
        previous = selected.get(target["pdb_id"])
        if previous is None or (ranking_score, -candidate["seed"], -candidate["sample"]) > (
            previous["ranking_score"], -previous["seed"], -previous["sample"],
        ):
            selected[target["pdb_id"]] = candidate
    return [selected[key] for key in sorted(selected)]


def write_refined_cif(source_path: Path, dest_path: Path, atom_rows, source_pos, refined_pos) -> None:
    doc = gemmi.cif.read_file(str(source_path))
    table = doc.sole_block().find_mmcif_category("_atom_site.")
    tags = list(table.tags)
    columns = [tags.index(f"_atom_site.Cartn_{axis}") for axis in "xyz"]
    if len(atom_rows) != len(source_pos) or len(atom_rows) != len(refined_pos):
        raise ValueError("Atom mapping and coordinate array sizes differ")
    for index, cif_row in enumerate(atom_rows):
        row = table[int(cif_row)]
        cif_source = [float(row[column]) for column in columns]
        if max(abs(cif_source[axis] - float(source_pos[index, axis])) for axis in range(3)) > 0.01:
            raise ValueError(f"Source CIF and .pt atom mapping differ at row {cif_row}: {source_path}")
        for axis, column in enumerate(columns):
            row[column] = f"{float(refined_pos[index, axis]):.4f}"
    doc.write_file(str(dest_path))


def run_inference(selected: list[dict], config: Path, checkpoint: Path, data_dir: Path,
                  output_dir: Path, precision: str) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    model = base.load_model(config, checkpoint, device)
    dataset = EuclideanDataset(data_dir, split="test", include_metadata=True)
    path_to_index = {str(path.resolve()): index for index, path in enumerate(dataset.data_files)}
    amp_enabled = device.type == "cuda" and precision == "bf16"
    for item in selected:
        sample_path = str(Path(item["sample_path"]).resolve())
        graph = dataset.get(path_to_index[sample_path])
        batch = Batch.from_data_list([graph]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled,
        ):
            refined_centered = model.sample(
                z=batch.atomic_numbers, pos_pred=batch.pos_pred,
                bond_index=batch.edge_index, batch=batch.batch,
                node_attr=batch.node_attr, edge_attr=batch.edge_attr,
                atom_plddt=batch.atom_plddt,
                atom_to_token_idx=batch.atom_to_token_idx,
                token_pair_confidence=batch.token_pair_confidence,
                num_tokens=batch.num_tokens,
                atom_mobility_attr=batch.atom_mobility_attr,
                geometry_bond_index=batch.geometry_bond_index,
                ideal_bond_length=batch.ideal_bond_length,
                clash_exclusion_index=batch.clash_exclusion_index,
                n_timesteps=1,
            )
        refined = (refined_centered.float() + batch.pos_pred.float().mean(dim=0)).cpu()
        source = batch.pos_pred.float().cpu()
        destination = output_dir / "predictions" / f"{item['foldbench_target_id']}_refined.cif"
        write_refined_cif(
            Path(item["source_cif"]), destination,
            torch.load(sample_path, map_location="cpu", weights_only=False)["predicted_atom_site_row"],
            source, refined,
        )
        item["refined_cif"] = str(destination.resolve())
        item["input_cif"] = str(Path(item["source_cif"]).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="FoldBench targets/monomer_rna.csv")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--ranking-manifest", type=Path, action="append", default=[],
                        help="Additional PT build manifest.tsv with original ranking_score")
    parser.add_argument("--prediction-root", type=Path, default=Path.home() / "Data_V2",
                        help="Prediction dataset root containing the current test/PDB links")
    args = parser.parse_args()
    summary = json.loads((args.test_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["split"] != "test" or Path(summary["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Export checkpoint differs from locked test evaluation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    with args.targets.open("r", encoding="utf-8", newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    target_pdb_ids = {row["pdb_id"].split("-", 1)[0].upper() for row in target_rows}
    manifest_paths = list((args.data_dir / "logs").glob("*/manifest.tsv")) + args.ranking_manifest
    manifest_scores = read_manifest_scores(manifest_paths, target_pdb_ids)
    selected = select_candidates(
        read_tsv(args.test_dir / "samples.tsv"), target_rows, manifest_scores,
        args.prediction_root,
    )
    if not selected:
        raise SystemExit("No FoldBench RNA monomer PDB+predicted-chain overlaps found")
    (args.output_dir / "predictions").mkdir(parents=True)
    (args.output_dir / "targets").mkdir()
    (args.output_dir / "evaluation" / "ProtenixV3Input").mkdir(parents=True)
    (args.output_dir / "evaluation" / "ProtenixV3Refined").mkdir(parents=True)
    run_inference(
        selected, args.config, args.checkpoint, args.data_dir, args.output_dir, args.precision,
    )
    with (args.output_dir / "targets" / "monomer_rna.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "chain_id"])
        writer.writeheader()
        for item in selected:
            writer.writerow({"pdb_id": item["foldbench_target_id"], "chain_id": item["predicted_chain_id"]})
    for algorithm, field in (("ProtenixV3Input", "input_cif"), ("ProtenixV3Refined", "refined_cif")):
        path = args.output_dir / "evaluation" / algorithm / "prediction_reference.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["pdb_id", "seed", "sample", "ranking_score", "prediction_path"],
            )
            writer.writeheader()
            for item in selected:
                writer.writerow({
                    "pdb_id": item["foldbench_target_id"],
                    "seed": item["seed"], "sample": item["sample"],
                    "ranking_score": item["ranking_score"], "prediction_path": item[field],
                })
    with (args.output_dir / "selected_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)
    print(f"FOLDBENCH_EXPORT_COMPLETE matched_targets={len(selected)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
