"""Export evaluated rank-1 RNA triplets and prepare a PyMOL bundle.

The evaluator calls ``export_sample_trio`` after inference. Run this file as a
script after both splits finish to package all three CIFs per candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import zipfile
from pathlib import Path


MANIFEST_FIELDS = (
    "split", "sample_id", "pdb_id", "native_chain_id", "predicted_chain_id",
    "seed", "sample", "length_nt", "input_all_atom_rmsd_a",
    "refined_all_atom_rmsd_a", "input_backbone_fit_rmsd_a",
    "refined_backbone_fit_rmsd_a", "input_all_fit_backbone_rmsd_a",
    "refined_all_fit_backbone_rmsd_a", "native_cif", "input_cif",
    "refined_cif", "source_native_cif", "source_predicted_cif",
)


def _resolve_source(recorded: str, fallback: tuple[Path, ...]) -> Path:
    candidate = Path(recorded)
    for path in (candidate, *fallback):
        if path.is_file():
            return path
    raise FileNotFoundError(f"source CIF missing: {recorded}; searched {fallback}")


def _source_xyz(table, row: int, columns: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(table[row][column]) for column in columns)


def write_subset_cif(
    source_path: Path, destination: Path, atom_rows, positions,
    *, expected_source=None, native_transform=None,
) -> None:
    """Keep source atom_site identities, replacing coordinates in source order."""
    import gemmi
    import torch

    source = gemmi.cif.read_file(str(source_path))
    source_table = source.sole_block().find_mmcif_category("_atom_site.")
    tags = list(source_table.tags)
    xyz_columns = tuple(tags.index(f"_atom_site.Cartn_{axis}") for axis in "xyz")
    row_ids = torch.as_tensor(atom_rows, dtype=torch.long).view(-1).tolist()
    coordinates = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 3)
    if len(row_ids) != len(coordinates) or not row_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError(f"invalid atom_site mapping for {source_path}")
    if min(row_ids) < 0 or max(row_ids) >= len(source_table):
        raise ValueError(f"atom_site row out of range in {source_path}")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError(f"nonfinite coordinates for {source_path}")

    document = gemmi.cif.Document()
    block = document.add_new_block("rna_refinement")
    loop = block.init_loop("_atom_site.", [tag.split(".", 1)[1] for tag in tags])
    for index, row_id in enumerate(row_ids):
        original_xyz = torch.tensor(_source_xyz(source_table, row_id, xyz_columns), dtype=torch.float64)
        if expected_source is not None:
            expected = torch.as_tensor(expected_source[index], dtype=torch.float64)
            if float((original_xyz - expected).abs().max()) > 0.02:
                raise ValueError(f"source CIF differs from .pt at row {row_id}: {source_path}")
        if native_transform is not None:
            rotation, translation = native_transform
            transformed = original_xyz @ rotation + translation
            if float((transformed - coordinates[index]).abs().max()) > 0.03:
                raise ValueError(f"native CIF differs from aligned .pt at row {row_id}: {source_path}")
        values = [str(source_table[row_id][column]) for column in range(len(tags))]
        for axis, column in enumerate(xyz_columns):
            values[column] = f"{float(coordinates[index, axis]):.4f}"
        loop.add_row(values)
    document.write_file(str(destination))


def export_sample_trio(
    *, payload_path: Path, output_dir: Path, split: str, pdb_id: str,
    sample_id: str, source, native, refined, prediction_root: Path,
    native_root: Path, metrics: dict,
) -> dict[str, str]:
    """Write exactly the atoms used by the model and matched native atoms."""
    import torch

    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError(f"unsafe sample_id: {sample_id}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    atom_count = len(source)
    observed = payload["target_mask"].bool().view(-1)
    predicted_rows = payload["predicted_atom_site_row"].long().view(-1)
    native_rows = payload["native_atom_site_row"].long().view(-1)
    if (len(observed) != atom_count or len(predicted_rows) != atom_count
            or len(native_rows) != atom_count or int(observed.sum()) < 3):
        raise ValueError(f"invalid .pt atom mapping: {payload_path}")
    if bool((native_rows[observed] < 0).any()) or bool((native_rows[~observed] >= 0).any()):
        raise ValueError(f"native row mapping disagrees with target_mask: {payload_path}")
    if not torch.allclose(source, payload["pos_pred"].float(), atol=1e-5, rtol=0):
        raise ValueError(f"inference source differs from .pt: {payload_path}")
    if not torch.allclose(native[observed], payload["pos"].float()[observed], atol=1e-5, rtol=0):
        raise ValueError(f"inference native differs from .pt: {payload_path}")

    seed = int(payload["protenix_seed"])
    source_predicted = _resolve_source(
        payload["source_predicted_cif"],
        tuple(prediction_root / candidate_split / pdb_id.lower() /
              f"seed_{seed}" / "predictions" / Path(payload["source_predicted_cif"]).name
              for candidate_split in (split, "test", "val", "train")),
    )
    source_native = _resolve_source(
        payload["source_native_cif"],
        (native_root / f"{pdb_id.upper()}.cif", native_root / f"{pdb_id.lower()}.cif"),
    )
    sample_dir = output_dir / "structures" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)
    write_subset_cif(source_predicted, sample_dir / "input.cif", predicted_rows, source,
                     expected_source=source)
    write_subset_cif(source_predicted, sample_dir / "refined.cif", predicted_rows, refined,
                     expected_source=source)
    rotation = payload["native_to_prediction_rotation"].double()
    translation = payload["native_to_prediction_translation"].double()
    write_subset_cif(
        source_native, sample_dir / "native.cif", native_rows[observed], native[observed],
        native_transform=(rotation, translation),
    )
    return {
        "split": split, "sample_id": sample_id, "pdb_id": pdb_id,
        "native_chain_id": str(payload["native_chain_id"]),
        "predicted_chain_id": str(payload["predicted_chain_id"]),
        "seed": seed, "sample": int(payload["protenix_sample"]),
        "length_nt": int(metrics["length"]),
        "input_all_atom_rmsd_a": f"{metrics['input_aligned_rmsd']:.8f}",
        "refined_all_atom_rmsd_a": f"{metrics['refined_aligned_rmsd']:.8f}",
        "input_backbone_fit_rmsd_a": f"{metrics['input_backbone_aligned_rmsd']:.8f}",
        "refined_backbone_fit_rmsd_a": f"{metrics['refined_backbone_aligned_rmsd']:.8f}",
        "input_all_fit_backbone_rmsd_a": f"{metrics['input_backbone_after_all_atom_fit_rmsd']:.8f}",
        "refined_all_fit_backbone_rmsd_a": f"{metrics['refined_backbone_after_all_atom_fit_rmsd']:.8f}",
        "native_cif": f"structures/{sample_id}/native.cif",
        "input_cif": f"structures/{sample_id}/input.cif",
        "refined_cif": f"structures/{sample_id}/refined.cif",
        "source_native_cif": str(source_native.resolve()),
        "source_predicted_cif": str(source_predicted.resolve()),
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or ()) != set(MANIFEST_FIELDS):
            raise ValueError(f"unexpected manifest columns: {path}")
        return list(reader)


def _pymol_script(records: list[dict[str, str]]) -> str:
    entries = [(row["split"], row["sample_id"]) for row in records]
    return f'''python
from pathlib import Path
from pymol import cmd

bundle = Path.cwd()
for split, sample_id in {entries!r}:
    cmd.reinitialize()
    root = bundle / split / "structures" / sample_id
    for name in ("native", "input", "refined"):
        path = root / (name + ".cif")
        if not path.is_file():
            raise FileNotFoundError(path)
        cmd.load(str(path), name)
    cmd.hide("everything")
    cmd.set_color("native_gray", [0.48, 0.48, 0.52])
    cmd.set_color("input_orange", [0.92, 0.48, 0.20])
    cmd.set_color("refined_teal", [0.03, 0.56, 0.63])
    for name, color in (("native", "native_gray"), ("input", "input_orange"),
                        ("refined", "refined_teal")):
        cmd.color(color, name)
        cmd.show("cartoon", name)
        cmd.show("spheres", name + " and name P")
        cmd.set("sphere_scale", 0.16, name)
    cmd.set("cartoon_transparency", 0.35, "input")
    cmd.set("cartoon_transparency", 0.25, "native")
    cmd.bg_color("white")
    cmd.set("orthoscopic", 1)
    cmd.orient("native or input or refined")
    cmd.zoom("native or input or refined", buffer=4)
    (bundle / "sessions" / split).mkdir(parents=True, exist_ok=True)
    (bundle / "images" / split).mkdir(parents=True, exist_ok=True)
    cmd.save(str(bundle / "sessions" / split / (sample_id + ".pse")))
    cmd.png(str(bundle / "images" / split / (sample_id + ".png")),
            width=1600, height=1200, dpi=200, ray=0)
    print("PYMOL_READY", split, sample_id)
python end
'''


def package_bundle(val_dir: Path, test_dir: Path, output_dir: Path, make_zip: bool) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {output_dir}")
    split_rows = {}
    for split, directory in (("val", val_dir), ("test", test_dir)):
        rows = _read_manifest(directory / "structure_manifest.tsv")
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError(f"empty or mismatched {split} manifest")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {split} sample IDs")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["candidate_count"] != len(rows):
            raise ValueError(f"{split} manifest does not match evaluation count")
        for row in rows:
            for name in ("native_cif", "input_cif", "refined_cif"):
                source = directory / row[name]
                if not source.is_file() or source.stat().st_size == 0:
                    raise FileNotFoundError(source)
            for name in ("input_all_fit_backbone_rmsd_a", "refined_all_fit_backbone_rmsd_a"):
                if not math.isfinite(float(row[name])):
                    raise ValueError(f"nonfinite metric for {row['sample_id']}")
        split_rows[split] = rows

    output_dir.mkdir(parents=True)
    all_rows = []
    for split, directory in (("val", val_dir), ("test", test_dir)):
        for row in split_rows[split]:
            for name in ("native_cif", "input_cif", "refined_cif"):
                destination = output_dir / split / row[name]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / row[name], destination)
            all_rows.append(row)
            (output_dir / f"view_{split}_{row['sample_id']}.pml").write_text(
                _pymol_script([row]), encoding="utf-8",
            )
    with (output_dir / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "render_all.pml").write_text(_pymol_script(all_rows), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "RNA refinement epoch 49 rank-1: val/test native, input, refined CIF triplets.\n"
        "Native CIF contains only mapped observed atoms, rigidly transformed into the input frame.\n"
        "Input/refined CIFs contain exactly the modelled RNA atoms. No extra PyMOL alignment is applied.\n"
        "In PyMOL, cd to this bundle directory, then run @view_val_SAMPLE_ID.pml\n"
        "or @view_test_SAMPLE_ID.pml. @render_all.pml creates all PNG and PSE files.\n"
        "Gray=native; orange=input; teal=refined. metrics.tsv lists both backbone RMSD definitions.\n"
        "RMSD values are calculated from coordinates before CIF rounding; visualization does not recompute them.\n",
        encoding="utf-8",
    )
    if make_zip:
        zip_path = output_dir.with_suffix(".zip")
        if zip_path.exists():
            raise FileExistsError(zip_path)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(output_dir.parent))
        print(f"PYMOL_BUNDLE_ZIP {zip_path}")
    print(f"PYMOL_BUNDLE_READY val={len(split_rows['val'])} test={len(split_rows['test'])} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    package_bundle(args.val_dir, args.test_dir, args.output_dir, args.zip)


if __name__ == "__main__":
    main()
