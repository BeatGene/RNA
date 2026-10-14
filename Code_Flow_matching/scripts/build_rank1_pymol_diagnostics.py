"""Build mapped, independently fitted RNA triplets for local PyMOL.

Inference and global physical losses come from evaluate_refinement_v3.py.
This packager fits the exported coordinates and writes local diagnostics;
it never selects a candidate or calls PyMOL on the server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

BACKBONE_NAMES = (
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'",
)
BASE_RING_NAMES = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows):
    if not rows:
        raise ValueError(f"empty table: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def array(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def fit_coordinates(mobile, target, mask):
    """Proper row-vector Kabsch fit, applied to ALL mobile atoms."""
    mobile, target = array(mobile, np.float64), array(target, np.float64)
    mask = array(mask, bool)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("coordinate shape mismatch")
    if mask.shape != (len(mobile),) or mask.sum() < 3:
        raise ValueError("fit requires at least three matched atoms")
    if not np.isfinite(mobile).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite coordinates")
    a, b = mobile[mask], target[mask]
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    if min(np.linalg.matrix_rank(a - ac), np.linalg.matrix_rank(b - bc)) < 2:
        raise ValueError("fit atoms are collinear")
    u, _, vh = np.linalg.svd((a - ac).T @ (b - bc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    translation = bc - ac @ rotation
    fitted = mobile @ rotation + translation
    rmsd = float(np.sqrt(np.square(fitted[mask] - target[mask]).sum(axis=1).mean()))
    return fitted, rmsd, {"rotation": rotation.tolist(), "translation": translation.tolist()}


def read_cif(path):
    import gemmi
    block = gemmi.cif.read_file(str(path)).sole_block()
    table = block.find_mmcif_category("_atom_site.")
    tags = list(table.tags)
    if not tags:
        raise ValueError(f"missing atom_site: {path}")
    rows = [[str(value) for value in row] for row in table]
    columns = [tags.index(f"_atom_site.Cartn_{axis}") for axis in "xyz"]
    positions = np.array([[float(row[column]) for column in columns] for row in rows])
    return tags, rows, positions


def cif_value(tags, row, suffix, default=""):
    import gemmi
    tag = "_atom_site." + suffix
    return gemmi.cif.as_string(row[tags.index(tag)]) if tag in tags else default


def atom_name(tags, row):
    name = cif_value(tags, row, "label_atom_id").replace("*", "'")
    return {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3"}.get(name, name)


def write_display_cif(path, tags, rows, positions, atom_indices, residue_index):
    """Use common atom IDs and sequential residues; keep original IDs in TSV."""
    import gemmi
    doc = gemmi.cif.Document()
    loop = doc.add_new_block("rank1_rna").init_loop(
        "_atom_site.", [tag.split(".", 1)[1] for tag in tags],
    )
    if len(rows) != len(positions) or len(rows) != len(atom_indices):
        raise ValueError("CIF and coordinate counts differ")
    columns = {tag.split(".", 1)[1]: i for i, tag in enumerate(tags)}
    for row, xyz, index in zip(rows, positions, atom_indices):
        values = list(row)
        replacements = {
            "id": str(int(index) + 1),
            "label_asym_id": "A", "auth_asym_id": "A",
            "label_seq_id": str(int(residue_index[index]) + 1),
            "auth_seq_id": str(int(residue_index[index]) + 1),
            "label_alt_id": ".", "pdbx_PDB_ins_code": "?",
            "pdbx_PDB_model_num": "1",
            "label_atom_id": gemmi.cif.quote(atom_name(tags, row)),
            "auth_atom_id": gemmi.cif.quote(atom_name(tags, row)),
        }
        replacements.update({f"Cartn_{axis}": f"{float(value):.5f}" for axis, value in zip("xyz", xyz)})
        for key, value in replacements.items():
            if key in columns:
                values[columns[key]] = value
        loop.add_row(values)
    doc.write_file(str(path))


def selections(path):
    records = {}
    for row in read_tsv(path):
        key = row["split"], row["pdb_id"].upper()
        if key in records or key[0] not in ("val", "test"):
            raise ValueError(f"duplicate/invalid selection: {key}")
        if not np.isfinite(float(row["ranking_score"])):
            raise ValueError(f"nonfinite ranking score: {key}")
        records[key] = row
    return records


def validate_candidate(payload, selection, path):
    identity = (str(payload["native_structure_id"]).upper(),
                int(payload["protenix_seed"]), int(payload["protenix_sample"]))
    expected = (selection["pdb_id"].upper(), int(selection["seed"]), int(selection["sample"]))
    if identity != expected:
        raise ValueError(f"rank-1 identity mismatch at {path}: {identity} != {expected}")
    for key in ("target_mask", "residue_index", "predicted_atom_site_row", "native_atom_site_row",
                "native_to_prediction_rotation", "native_to_prediction_translation"):
        if key not in payload:
            raise ValueError(f"missing mapped field {key}: {path}")


def preflight(data_root, selection_file, expected_counts, prediction_root=None, native_root=None):
    import torch
    from export_rank1_pymol_v3 import _resolve_source
    records = selections(selection_file)
    for split, expected_count in expected_counts.items():
        paths = sorted((data_root / split).rglob("*.pt"))
        selected = {pdb: row for (s, pdb), row in records.items() if s == split}
        if len(paths) != expected_count or len(selected) != expected_count:
            raise ValueError(f"{split}: PT={len(paths)}, selection={len(selected)}, expected={expected_count}")
        seen = set()
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            pdb = str(payload["native_structure_id"]).upper()
            if pdb in seen or pdb not in selected:
                raise ValueError(f"duplicate/unselected PDB: {split}/{pdb}")
            validate_candidate(payload, selected[pdb], path)
            if prediction_root is not None and native_root is not None:
                seed = int(payload["protenix_seed"])
                _resolve_source(payload["source_predicted_cif"], tuple(
                    prediction_root / s / pdb.lower() / f"seed_{seed}" / "predictions" /
                    Path(payload["source_predicted_cif"]).name for s in (split, "test", "val", "train")))
                _resolve_source(payload["source_native_cif"], (
                    native_root / f"{pdb.lower()}.cif", native_root / f"{pdb.upper()}.cif"))
            seen.add(pdb)
        if seen != set(selected):
            raise ValueError(f"PDB coverage differs from selection: {split}")
        print(f"RANK1_PREFLIGHT_OK split={split} pdbs={len(seen)}", flush=True)


def plane_thickness(positions, indices):
    if len(indices) < 4:
        return ""
    points = positions[indices]
    _, singular, _ = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return float(singular[-1] / np.sqrt(len(points)))


def local_diagnostics(payload, source, target, refined, fitted_before, fitted_after, names, identities):
    observed = array(payload["target_mask"], bool)
    residue_index = array(payload["residue_index"], int)
    plddt = array(payload["atom_plddt"], float).reshape(-1)
    movement = np.linalg.norm(refined - source, axis=1)
    before_error = np.linalg.norm(fitted_before - target, axis=1)
    after_error = np.linalg.norm(fitted_after - target, axis=1)
    bonds = array(payload["geometry_bond_index"], int).T
    ideals = array(payload["ideal_bond_length"], float).reshape(-1)
    attached_bonds = [[] for _ in payload["sequence"]]
    for i, (a, b) in enumerate(bonds):
        for r in {int(residue_index[a]), int(residue_index[b])}:
            attached_bonds[r].append(i)
    atoms, bond_rows, residues = [], [], []
    for i in range(len(source)):
        atoms.append({
            "atom_id": i + 1, "residue": int(residue_index[i]) + 1,
            "atom_name": names[i], "native_observed": int(observed[i]),
            **identities[i], "plddt_0_1": float(plddt[i]),
            "raw_movement_a": float(movement[i]),
            "input_error_all_atom_fit_a": float(before_error[i]) if observed[i] else "",
            "refined_error_all_atom_fit_a": float(after_error[i]) if observed[i] else "",
            "improvement_a": float(before_error[i] - after_error[i]) if observed[i] else "",
        })
    for (a, b), ideal in zip(bonds, ideals):
        pre = float(np.linalg.norm(source[a] - source[b]))
        post = float(np.linalg.norm(refined[a] - refined[b]))
        native = float(np.linalg.norm(target[a] - target[b])) if observed[a] and observed[b] else ""
        bond_rows.append({
            "atom_id_1": int(a) + 1, "atom_id_2": int(b) + 1,
            "residue_1": int(residue_index[a]) + 1, "residue_2": int(residue_index[b]) + 1,
            "atom_name_1": names[a], "atom_name_2": names[b],
            "ideal_length_a": float(ideal), "native_length_a": native,
            "input_length_a": pre, "refined_length_a": post,
            "input_abs_error_a": abs(pre - ideal), "refined_abs_error_a": abs(post - ideal),
            "error_increase_a": abs(post - ideal) - abs(pre - ideal),
        })
    for r, base in enumerate(payload["sequence"]):
        indices = np.flatnonzero(residue_index == r)
        matched = indices[observed[indices]]
        attached = attached_bonds[r]
        ring = [i for i in indices if names[i] in BASE_RING_NAMES]
        pre = float(np.sqrt(np.square(before_error[matched]).mean())) if len(matched) else ""
        post = float(np.sqrt(np.square(after_error[matched]).mean())) if len(matched) else ""
        residues.append({
            "residue": r + 1, "base": base, "atom_count": len(indices),
            "native_observed_atoms": len(matched),
            "input_rms_error_all_atom_fit_a": pre, "refined_rms_error_all_atom_fit_a": post,
            "improvement_a": pre - post if len(matched) else "",
            "mean_raw_movement_a": float(movement[indices].mean()),
            "mean_plddt_0_1": float(plddt[indices].mean()),
            "input_mean_abs_bond_error_a": float(np.mean([bond_rows[i]["input_abs_error_a"] for i in attached])) if attached else "",
            "refined_mean_abs_bond_error_a": float(np.mean([bond_rows[i]["refined_abs_error_a"] for i in attached])) if attached else "",
            "input_base_plane_rms_a": plane_thickness(source, ring),
            "refined_base_plane_rms_a": plane_thickness(refined, ring),
            "native_base_plane_rms_a": plane_thickness(target, [i for i in ring if observed[i]]),
        })
    return atoms, bond_rows, residues


def prepare_target(evaluation_dir, structure, sample, selection, destination):
    import torch
    payload_path = Path(sample["sample_path"])
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    validate_candidate(payload, selection, payload_path)
    observed = array(payload["target_mask"], bool).reshape(-1)
    residue_index = array(payload["residue_index"], int).reshape(-1)
    tags, input_rows, source = read_cif(evaluation_dir / structure["input_cif"])
    refined_tags, refined_rows, refined = read_cif(evaluation_dir / structure["refined_cif"])
    native_tags, native_rows, native_observed = read_cif(evaluation_dir / structure["native_cif"])
    target = array(payload["pos"], float).copy()
    if len(source) != len(observed) or refined.shape != source.shape or len(native_observed) != observed.sum():
        raise ValueError(f"exported atom counts differ from PT: {payload_path}")
    if not np.allclose(source, array(payload["pos_pred"]), atol=0.00011, rtol=0):
        raise ValueError(f"input CIF differs from PT: {payload_path}")
    if not np.allclose(native_observed, target[observed], atol=0.00011, rtol=0):
        raise ValueError(f"native CIF differs from PT: {payload_path}")
    names = [atom_name(tags, row) for row in input_rows]
    if names != [atom_name(refined_tags, row) for row in refined_rows]:
        raise ValueError("input/refined atom identities differ")
    matched_indices = np.flatnonzero(observed)
    if [names[i] for i in matched_indices] != [atom_name(native_tags, row) for row in native_rows]:
        raise ValueError("native atom names differ from matched prediction atoms")
    target[observed] = native_observed
    backbone = observed & np.isin(names, BACKBONE_NAMES)
    destination.mkdir(parents=True, exist_ok=False)
    transforms = {}
    aligned = {}
    for mode, mask in (("all_atom", observed), ("backbone", backbone)):
        before, pre_rmsd, pre_transform = fit_coordinates(source, target, mask)
        after, post_rmsd, post_transform = fit_coordinates(refined, target, mask)
        aligned[mode] = before, after
        expected_pre = float(sample["input_aligned_rmsd"] if mode == "all_atom" else sample["input_backbone_aligned_rmsd"])
        expected_post = float(sample["refined_aligned_rmsd"] if mode == "all_atom" else sample["refined_backbone_aligned_rmsd"])
        if max(abs(pre_rmsd - expected_pre), abs(post_rmsd - expected_post)) > 0.001:
            raise ValueError(f"exported {mode} RMSD disagrees with evaluation: {payload_path}")
        folder = destination / mode
        folder.mkdir()
        write_display_cif(folder / "native.cif", native_tags, native_rows,
                          native_observed, matched_indices, residue_index)
        write_display_cif(folder / "before.cif", tags, input_rows, before,
                          np.arange(len(source)), residue_index)
        write_display_cif(folder / "after.cif", refined_tags, refined_rows, after,
                          np.arange(len(source)), residue_index)
        transforms[mode] = {"before": pre_transform, "after": post_transform,
                            "input_rmsd_a": pre_rmsd, "refined_rmsd_a": post_rmsd}
    raw_folder = destination / "source_frame"
    raw_folder.mkdir()
    for field, name in (("native_cif", "native"), ("input_cif", "before"), ("refined_cif", "after")):
        shutil.copy2(evaluation_dir / structure[field], raw_folder / (name + ".cif"))
    identities = []
    native_by_index = dict(zip(matched_indices, native_rows))
    for i, row in enumerate(input_rows):
        native_row = native_by_index.get(i)
        identities.append({
            "source_prediction_chain": cif_value(tags, row, "auth_asym_id"),
            "source_prediction_resi": cif_value(tags, row, "auth_seq_id"),
            "source_native_chain": cif_value(native_tags, native_row, "auth_asym_id") if native_row else "",
            "source_native_resi": cif_value(native_tags, native_row, "auth_seq_id") if native_row else "",
            "source_native_insertion": cif_value(native_tags, native_row, "pdbx_PDB_ins_code") if native_row else "",
        })
    atoms, bonds, residues = local_diagnostics(payload, source, target, refined, *aligned["all_atom"], names, identities)
    write_tsv(destination / "atom_metrics.tsv", atoms)
    write_tsv(destination / "bond_metrics.tsv", bonds)
    write_tsv(destination / "residue_metrics.tsv", residues)
    write_json(destination / "display.json", {
        "pdb_id": sample["pdb_id"], "split": sample["split"], "sample_id": sample["sample_id"],
        "ranking_score": float(selection["ranking_score"]),
        "atom_ids": list(range(1, len(source) + 1)),
        "observed_atom_ids": (matched_indices + 1).tolist(),
        "bonds": (array(payload["geometry_bond_index"], int).T + 1).tolist(),
        "transforms": transforms,
        "source_native_cif": structure["source_native_cif"],
        "source_predicted_cif": structure["source_predicted_cif"],
    })
    result = dict(sample)
    result.update({"ranking_score": selection["ranking_score"],
                   "candidate_count": selection["candidate_count"], "complete_50x4": selection["complete_50x4"],
                   "view_all_atom": f"view_{sample['split']}_{sample['pdb_id'].upper()}.pml",
                   "view_backbone": f"view_{sample['split']}_{sample['pdb_id'].upper()}_backbone.pml"})
    return result


def package(val_dir, test_dir, selection_file, output_dir, expected_counts, make_zip):
    records = selections(selection_file)
    inputs, checkpoint = [], None
    for split, directory in (("val", val_dir), ("test", test_dir)):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        current_checkpoint = summary["checkpoint"]
        if checkpoint is not None and current_checkpoint != checkpoint:
            raise ValueError("val/test checkpoint mismatch")
        checkpoint = current_checkpoint
        samples = read_tsv(directory / "samples.tsv")
        structures = read_tsv(directory / "structure_manifest.tsv")
        if len(samples) != expected_counts[split] or len(structures) != len(samples):
            raise ValueError(f"{split}: expected {expected_counts[split]} samples and exported triplets")
        if summary["candidate_count"] != len(samples) or summary["split"] != split:
            raise ValueError(f"evaluation summary mismatch: {split}")
        by_id = {row["sample_id"]: row for row in structures}
        if len(by_id) != len(structures) or set(by_id) != {row["sample_id"] for row in samples}:
            raise ValueError(f"duplicate/missing structure identity: {split}")
        pdbs = set()
        for sample in samples:
            pdb = sample["pdb_id"].upper()
            if sample["split"] != split or pdb in pdbs or (split, pdb) not in records:
                raise ValueError(f"duplicate/missing/mismatched rank-1 PDB: {split}/{pdb}")
            pdbs.add(pdb)
            for metric in ("input_bond_loss_a2", "refined_bond_loss_a2", "input_clash_loss_a2",
                           "refined_clash_loss_a2", "input_plane_loss_a2", "refined_plane_loss_a2"):
                if not np.isfinite(float(sample[metric])):
                    raise ValueError("physical metrics missing; evaluate with --physics")
            inputs.append((split, directory, sample, by_id[sample["sample_id"]], records[split, pdb]))
        if pdbs != {pdb for s, pdb in records if s == split}:
            raise ValueError(f"selection coverage mismatch: {split}")
    if output_dir.exists() or (make_zip and output_dir.with_suffix(".zip").exists()):
        raise FileExistsError(f"choose a fresh output path: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = []
    for split, directory, sample, structure, selection in inputs:
        pdb = sample["pdb_id"].upper()
        result = prepare_target(directory, structure, sample, selection, output_dir / split / pdb)
        manifest.append(result)
        for mode, suffix in (("all_atom", ""), ("backbone", "_backbone")):
            (output_dir / f"view_{split}_{pdb}{suffix}.pml").write_text(
                f"run rna_pymol_viewer.py, module\nrna_open {split}, {pdb}, {mode}\n", encoding="utf-8")
        print(f"PYMOL_TARGET_READY {split}/{pdb}", flush=True)
    write_tsv(output_dir / "manifest.tsv", manifest)
    for split in ("val", "test"):
        subset = [row for row in manifest if row["split"] == split]
        write_tsv(output_dir / f"inspection_order_{split}.tsv",
                  sorted(subset, key=lambda row: float(row["aligned_improvement"])))
    shutil.copy2(selection_file, output_dir / "rank1_selection.tsv")
    shutil.copy2(Path(__file__).with_name("rna_pymol_viewer.py"), output_dir / "rna_pymol_viewer.py")
    shutil.copy2(Path(__file__).with_name("RANK1_PYMOL_DIAGNOSTICS.md"), output_dir / "README.md")
    (output_dir / "render_all.pml").write_text("run rna_pymol_viewer.py, module\nrna_render_all\n", encoding="utf-8")
    for split, directory in (("val", val_dir), ("test", test_dir)):
        shutil.copy2(directory / "summary.json", output_dir / f"evaluation_summary_{split}.json")
    for name in ("config_snapshot.yaml", "checkpoint.sha256"):
        if (val_dir.parent / name).is_file():
            shutil.copy2(val_dir.parent / name, output_dir / name)
    write_json(output_dir / "bundle_summary.json", {
        "status": "COMPLETE", "counts": expected_counts, "checkpoint": checkpoint,
        "selection_sha256": hashlib.sha256(selection_file.read_bytes()).hexdigest(),
        "coordinate_precision_a": 0.00001,
        "fit_definition": "independent proper Kabsch fits of before/after to matched native atoms",
        "native_definition": "only experimental atoms mapped to the model, no synthetic native coordinates",
        "diagnostic_definition": "per-residue errors after global ALL-ATOM fit; NOT local residue fits",
        "display_numbering": "chain A; sequential residue 1..R; atom id equals prediction index + 1",
        "bond_definition": "template covalent connectivity; single display bonds; no distance guessing",
    })
    if make_zip:
        from repack_rank1_pymol_bundle import write_bundle_zip
        archive_path = output_dir.with_suffix(".zip")
        write_bundle_zip(output_dir, archive_path)
    print(f"PYMOL_DIAGNOSTICS_COMPLETE val={expected_counts['val']} test={expected_counts['test']} output={output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--check-data", type=Path, help="check one PT per PDB against archived selection, then exit")
    parser.add_argument("--prediction-root", type=Path, default=Path.home() / "Data_V2")
    parser.add_argument("--native-root", type=Path, default=Path.home() / "pdb_data")
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-val", type=int, default=103)
    parser.add_argument("--expected-test", type=int, default=67)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    expected = {"val": args.expected_val, "test": args.expected_test}
    if args.check_data:
        preflight(args.check_data, args.selection_file, expected, args.prediction_root, args.native_root)
    else:
        if not all((args.val_dir, args.test_dir, args.output_dir)):
            parser.error("packaging requires --val-dir, --test-dir, --output-dir")
        package(args.val_dir, args.test_dir, args.selection_file, args.output_dir, expected, args.zip)


if __name__ == "__main__":
    main()
