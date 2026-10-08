"""Pack the nine paired epoch-49 FoldBench structures for local PyMOL rendering.

Run this on the lab server after compare_foldbench_refinement_v3.py completed.
This script only reads the source/evaluation data; it writes a new bundle and ZIP.
PyMOL itself is not required on the server.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import zipfile


TARGETS = (
    "7sxp-assembly1", "7wia-assembly1", "7wii-assembly1", "7zj4-assembly1",
    "8hb8-assembly1", "8its-assembly1", "8upt-assembly1", "8v1h-assembly1",
    "9g7c-assembly1",
)
ALGORITHMS = ("ProtenixV3Input", "ProtenixV3Refined")


def read_table(path: Path, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def unique_by(rows: list[dict[str, str]], field: str, source: Path) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}={key} in {source}")
        result[key] = row
    return result


def pymol_script(targets: tuple[str, ...]) -> str:
    # All paths are relative to the extracted bundle directory.  The GUI user
    # should first `cd` into that directory, or launch `pymol` from it.
    return f'''python
from pathlib import Path
from pymol import cmd

bundle = Path.cwd()
targets = {targets!r}
for target in targets:
    cmd.reinitialize()
    cmd.load(str(bundle / "structures" / (target + "_input.cif")), "before")
    cmd.load(str(bundle / "structures" / (target + "_refined.cif")), "after")

    # The FoldBench exports use predicted chain A.  Keep the target RNA only.
    for obj in ("before", "after"):
        if cmd.count_atoms(obj + " and chain A"):
            cmd.remove(obj + " and not chain A")
    before_p = cmd.count_atoms("before and name P")
    after_p = cmd.count_atoms("after and name P")
    if before_p < 3 or before_p != after_p:
        raise RuntimeError(f"{{target}}: phosphate atom counts differ: {{before_p}} vs {{after_p}}")

    # Rigidly superpose the refined RNA onto its own input, using phosphate
    # atoms and no outlier rejection.  This is only for display, not a new
    # benchmark RMSD calculation.
    cmd.align("after and name P", "before and name P", cycles=0)
    cmd.hide("everything")
    cmd.set_color("before_orange", [0.85, 0.45, 0.29])
    cmd.set_color("after_teal", [0.09, 0.49, 0.54])
    cmd.color("before_orange", "before")
    cmd.color("after_teal", "after")
    cmd.show("cartoon", "before")
    cmd.show("cartoon", "after")
    cmd.set("cartoon_transparency", 0.45, "before")
    cmd.set("cartoon_transparency", 0.0, "after")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadows", 0)
    cmd.orient("before or after")
    cmd.zoom("before or after", buffer=4)

    (bundle / "images").mkdir(exist_ok=True)
    (bundle / "sessions").mkdir(exist_ok=True)
    cmd.save(str(bundle / "sessions" / (target + ".pse")))
    cmd.png(str(bundle / "images" / (target + "_overlay.png")),
            width=1800, height=1350, dpi=300, ray=1)
    print("RENDERED", target, "P_atoms", before_p)
python end
'''


def build_records(foldbench_dir: Path, rank1_samples_path: Path) -> list[dict[str, str]]:
    selected_path = foldbench_dir / "selected_candidates.tsv"
    selected = unique_by(read_table(selected_path, "\t"), "foldbench_target_id", selected_path)
    if set(selected) != set(TARGETS):
        raise ValueError(f"Expected exactly nine targets; found {sorted(selected)}")

    rank1_rows = read_table(rank1_samples_path, "\t")
    rank1 = unique_by(rank1_rows, "pdb_id", rank1_samples_path)
    scores = {}
    for algorithm in ALGORITHMS:
        path = foldbench_dir / "evaluation" / algorithm / "raw" / "monomer_rna_ost.csv"
        scores[algorithm] = unique_by(read_table(path, ","), "pdb_id", path)
        if set(scores[algorithm]) != set(TARGETS):
            raise ValueError(f"{algorithm}: expected exactly nine scored targets")

    records = []
    for target in TARGETS:
        entry = selected[target]
        pdb_id = entry["pdb_id"].upper()
        sample = rank1.get(pdb_id)
        if sample is None:
            raise ValueError(f"Rank-1 test sample missing for {target}")
        if (sample["protenix_seed"], sample["protenix_sample"]) != (entry["seed"], entry["sample"]):
            raise ValueError(f"Rank-1 and FoldBench selected different candidates for {target}")
        if sample["native_chain_id"] != entry["native_chain_id"]:
            raise ValueError(f"Native chain mismatch for {target}")
        if entry["predicted_chain_id"] != "A":
            raise ValueError(f"Expected predicted chain A for {target}")

        before = Path(entry["input_cif"])
        after = Path(entry["refined_cif"])
        for path in (before, after):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty CIF for {target}: {path}")

        pre_score = scores["ProtenixV3Input"][target]
        post_score = scores["ProtenixV3Refined"][target]
        for algorithm, score in zip(ALGORITHMS, (pre_score, post_score)):
            if (score["seed"], score["sample"]) != (entry["seed"], entry["sample"]):
                raise ValueError(f"{algorithm}: wrong scored candidate for {target}")
            for field in ("rmsd", "lddt"):
                float(score[field])

        records.append({
            "target": target,
            "pdb_id": pdb_id,
            "native_chain_id": sample["native_chain_id"],
            "length_nt": sample["length"],
            "seed": entry["seed"],
            "sample": entry["sample"],
            "project_input_rmsd_a": sample["input_aligned_rmsd"],
            "project_refined_rmsd_a": sample["refined_aligned_rmsd"],
            "foldbench_input_rmsd_a": pre_score["rmsd"],
            "foldbench_refined_rmsd_a": post_score["rmsd"],
            "foldbench_input_lddt": pre_score["lddt"],
            "foldbench_refined_lddt": post_score["lddt"],
            "input_cif_source": str(before.resolve()),
            "refined_cif_source": str(after.resolve()),
        })
    return records


def main() -> None:
    project = Path.home() / "Code_Flow_matching"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foldbench-dir", type=Path,
        default=project / "evaluation" / "v3_20261008" / "foldbench_9targets",
    )
    parser.add_argument(
        "--rank1-test-samples", type=Path,
        default=project / "evaluation" / "v3_rank1_epoch49_20261008" / "test_locked" / "samples.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate the nine pairs without writing")
    args = parser.parse_args()
    foldbench_dir = args.foldbench_dir.resolve()
    output_dir = args.output_dir or foldbench_dir / "pymol_bundle_epoch49"
    records = build_records(foldbench_dir, args.rank1_test_samples.resolve())
    print(f"PAIRS_OK targets={len(records)} output_dir={output_dir}")
    for row in records:
        print("PAIR", row["target"], f"seed={row['seed']}", f"sample={row['sample']}",
              f"length={row['length_nt']}")
    if args.dry_run:
        return
    if output_dir.exists() or output_dir.with_suffix(".zip").exists():
        raise FileExistsError(f"Choose a new output path; bundle or ZIP exists: {output_dir}")

    (output_dir / "structures").mkdir(parents=True)
    for row in records:
        target = row["target"]
        shutil.copy2(row["input_cif_source"], output_dir / "structures" / f"{target}_input.cif")
        shutil.copy2(row["refined_cif_source"], output_dir / "structures" / f"{target}_refined.cif")
    with (output_dir / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "render_all.pml").write_text(pymol_script(TARGETS), encoding="utf-8")
    for target in TARGETS:
        (output_dir / f"view_{target}.pml").write_text(pymol_script((target,)), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "FoldBench 9 目标 PyMOL 叠图：橙色半透明为精修前，蓝绿色不透明为精修后。\n"
        "1. 将整个 ZIP 解压到本地电脑。\n"
        "2. 在 PyMOL 命令行先 cd 到解压后的 pymol_bundle_epoch49 文件夹。\n"
        "3. 输入 @render_all.pml 批量生成 9 张图；或输入 @view_7sxp-assembly1.pml 只看一个目标。\n"
        "4. 图片在 images/*_overlay.png；可编辑会话在 sessions/*.pse。\n"
        "5. metrics.tsv 记录链长、本项目 RMSD 与 FoldBench RMSD/lDDT，方便制作 PPT 图注。\n"
        "图像叠合使用磷原子，不等于本项目或 FoldBench 的 RMSD 计算。\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(output_dir.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir.parent))
    print(f"BUNDLE_READY {output_dir.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
