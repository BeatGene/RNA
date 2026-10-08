"""Export and score paired FoldBench RNA monomers for V3 refinement.

Run from the server's protenix-1.0.5 environment. FoldBench scoring is run
inside a separate Conda environment via ``conda run``. Existing six-target
exports are never reused or overwritten.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile


TARGET_IDS = frozenset(
    f"{pdb.lower()}-assembly1"
    for pdb in ("7SXP", "7WIA", "7WII", "7ZJ4", "8HB8", "8ITS", "8UPT", "8V1H", "9G7C")
)
ALGORITHMS = ("ProtenixV3Input", "ProtenixV3Refined")
MODEL_NAME = "epoch49"
EVAL_DIR_NAME = "v3_20261008"
CHECKPOINT_NAME = "epoch=49-step=157550.ckpt"


def read_delimited(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def validate_export(root: Path) -> dict[str, dict[str, str]]:
    """Require exactly the intended nine targets and paired existing CIFs."""
    selected_path = root / "selected_candidates.tsv"
    check_file(selected_path)
    selected = read_delimited(selected_path, "\t")
    ids = [row["foldbench_target_id"] for row in selected]
    if len(ids) != len(TARGET_IDS) or set(ids) != TARGET_IDS:
        raise ValueError(
            f"{root}: expected exactly 9 FoldBench targets; found {ids}. "
            "Choose a fresh --output-subdir after updating export_foldbench_v3.py."
        )
    indexed = {row["foldbench_target_id"]: row for row in selected}
    for target, row in indexed.items():
        if row.get("predicted_chain_id") != "A":
            raise ValueError(f"{target}: expected corrected exporter with predicted chain A")
        for field in ("input_cif", "refined_cif"):
            check_file(Path(row[field]))
    target_rows = read_delimited(root / "targets" / "monomer_rna.csv", ",")
    if {row["pdb_id"] for row in target_rows} != TARGET_IDS or len(target_rows) != 9:
        raise ValueError(f"{root}: target CSV does not contain exactly the 9 expected targets")
    for algorithm, field in ((ALGORITHMS[0], "input_cif"), (ALGORITHMS[1], "refined_cif")):
        reference = read_delimited(root / "evaluation" / algorithm / "prediction_reference.csv", ",")
        if len(reference) != 9 or {row["pdb_id"] for row in reference} != TARGET_IDS:
            raise ValueError(f"{root}: {algorithm} prediction reference is incomplete")
        for row in reference:
            chosen = indexed[row["pdb_id"]]
            if (row["seed"], row["sample"]) != (chosen["seed"], chosen["sample"]):
                raise ValueError(f"{row['pdb_id']}: input/refined candidate differs")
            if Path(row["prediction_path"]).resolve() != Path(chosen[field]).resolve():
                raise ValueError(f"{row['pdb_id']}: prediction path differs from selected candidate")
    return indexed


def validate_ground_truth(directory: Path) -> None:
    for target in sorted(TARGET_IDS):
        path = directory / f"{target}.cif"
        check_file(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Ground-truth CIF is empty: {path}")


def extract_ground_truth_tar(archive: Path, directory: Path) -> None:
    """Copy only the nine expected regular CIF members, without extracting paths."""
    check_file(archive)
    wanted = {f"{target}.cif" for target in TARGET_IDS}
    with tarfile.open(archive, "r:*") as handle:
        found: dict[str, tarfile.TarInfo] = {}
        for member in handle:
            filename = Path(member.name).name.lower()
            if filename not in wanted:
                continue
            if not member.isfile() or member.size == 0:
                raise ValueError(f"Invalid ground-truth member in {archive}: {member.name}")
            if filename in found:
                raise ValueError(f"Duplicate ground-truth member in {archive}: {filename}")
            found[filename] = member
        missing = wanted - found.keys()
        if missing:
            raise ValueError(f"Reference tar is missing {sorted(missing)}: {archive}")
        directory.mkdir(parents=True, exist_ok=True)
        for filename in sorted(wanted):
            member = found[filename]
            destination = directory / filename
            if destination.exists():
                if destination.is_file() and destination.stat().st_size == member.size:
                    print(f"REFERENCE_EXISTS {destination}", flush=True)
                    continue
                raise ValueError(f"Existing reference differs from archive; inspect it: {destination}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read reference from tar: {member.name}")
            try:
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"Extracted reference size mismatch: {destination}")
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            print(f"REFERENCE_EXTRACTED {destination} bytes={member.size}", flush=True)


def check_foldbench_environment(name: str) -> None:
    """Fail before GPU export if the scoring environment is unavailable."""
    checks = (
        ["python", "-c", "import pandas, tqdm"],
        ["ost", "compare-structures", "--help"],
    )
    for command in checks:
        result = subprocess.run(
            ["conda", "run", "-n", name, *command],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(f"FoldBench Conda environment {name!r} failed {' '.join(command)}: {detail}")
    print(f"FOLDBENCH_ENV_OK name={name} pandas=tqdm ost=available", flush=True)


def load_paired(path: Path) -> dict[str, dict[str, str]]:
    rows = read_delimited(path, "\t")
    ids = [row["foldbench_target_id"] for row in rows]
    if len(ids) != 9 or set(ids) != TARGET_IDS:
        raise ValueError(f"{path}: expected 9 paired lDDT scores; got {ids}")
    for row in rows:
        for field in ("input_lddt", "refined_lddt", "lddt_change"):
            value = float(row[field])
            if not math.isfinite(value) or (field != "lddt_change" and not 0 <= value <= 1):
                raise ValueError(f"{path}: invalid {field} for {row['foldbench_target_id']}")
    return {row["foldbench_target_id"]: row for row in rows}


def write_comparison(path: Path, paired: dict[str, dict[str, dict[str, str]]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for target in sorted(TARGET_IDS):
        for model, data in paired.items():
            row = data[target]
            rows.append({
                "checkpoint": model,
                "foldbench_target_id": target,
                "seed": row["seed"],
                "sample": row["sample"],
                "input_lddt": row["input_lddt"],
                "refined_lddt": row["refined_lddt"],
                "lddt_change": row["lddt_change"],
            })
    fields = list(rows[0])
    with (path / "paired_lddt_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V3 精修前后 FoldBench RNA 单体 lDDT 对比", "",
        "每个目标用原始 Protenix ranking_score 选择一个候选；精修前后使用相同候选。",
        "仅覆盖测试集中的 9/15 个官方 RNA 单体目标，不能当作官方完整榜单分数。", "",
        "| Checkpoint | 目标数 | 精修前平均 lDDT | 精修后平均 lDDT | 平均变化 | 提升目标数 | 降低目标数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, data in paired.items():
        before = [float(row["input_lddt"]) for row in data.values()]
        after = [float(row["refined_lddt"]) for row in data.values()]
        changes = [b - a for a, b in zip(before, after)]
        lines.append(
            f"| {model} | 9 | {statistics.fmean(before):.4f} | {statistics.fmean(after):.4f} | "
            f"{statistics.fmean(changes):+.4f} | {sum(x > 0 for x in changes)} | "
            f"{sum(x < 0 for x in changes)} |"
        )
    lines += ["", "| Checkpoint | PDB | seed | sample | 精修前 lDDT | 精修后 lDDT | 变化 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['foldbench_target_id']} | {row['seed']} | "
            f"{row['sample']} | {float(row['input_lddt']):.4f} | "
            f"{float(row['refined_lddt']):.4f} | {float(row['lddt_change']):+.4f} |"
        )
    (path / "paired_lddt_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foldbench-repo", type=Path, required=True,
                        help="FoldBench repository containing evaluate.py")
    parser.add_argument("--ground-truth-dir", type=Path, required=True,
                        help="Directory with the official <pdb>-assembly1.cif files")
    parser.add_argument("--ground-truth-tar", type=Path,
                        help="Optional official tar archive; extract only the nine RNA references")
    parser.add_argument("--stage", choices=("all", "export", "score"), default="all")
    parser.add_argument("--foldbench-conda-env", default="foldbench")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Data_PT_V2")
    parser.add_argument("--ranking-manifest", type=Path, action="append", default=[],
                        help="Additional PT build manifest.tsv with original ranking_score")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=project / "checkpoints" / "rna_refinement_v3_residual_mobility")
    parser.add_argument("--output-subdir", default="foldbench_9targets")
    parser.add_argument("--comparison-dir", type=Path,
                        default=project / "evaluation" / "foldbench_epoch49_9targets_comparison")
    args = parser.parse_args()
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in (".", ".."):
        parser.error("--output-subdir must be a simple directory name")

    names = (MODEL_NAME,)
    fb_repo = args.foldbench_repo.resolve()
    ground_truth = args.ground_truth_dir.resolve()
    targets = project / "config" / "foldbench_monomer_rna_targets.csv"
    config = project / "config" / "RNA_train_residual_v3.yaml"
    exporter = project / "scripts" / "export_foldbench_v3.py"
    summarizer = project / "scripts" / "summarize_foldbench_v3.py"
    for file in (targets, summarizer):
        check_file(file)
    if args.stage in ("all", "export"):
        for file in (config, exporter):
            check_file(file)
    if args.stage in ("all", "score"):
        for file in (fb_repo / "evaluate.py", fb_repo / "task_score_summary.py"):
            check_file(file)
        if shutil.which("conda") is None:
            raise RuntimeError("conda was not found; activate Conda before running this script")
        check_foldbench_environment(args.foldbench_conda_env)
        if args.ground_truth_tar is not None:
            extract_ground_truth_tar(args.ground_truth_tar.resolve(), ground_truth)
        validate_ground_truth(ground_truth)

    roots: dict[str, Path] = {}
    for name in names:
        eval_root = project / "evaluation" / EVAL_DIR_NAME
        test_dir = eval_root / "test_locked"
        output = eval_root / args.output_subdir
        roots[name] = output
        summary_path = test_dir / "summary.json"
        check_file(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = (args.checkpoint_dir / CHECKPOINT_NAME).resolve()
        if summary.get("split") != "test" or Path(summary["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"{name}: test evaluation checkpoint does not match {checkpoint}")
        if args.stage in ("all", "export") and not (output / "selected_candidates.tsv").exists():
            check_file(checkpoint)
            command = [
                sys.executable, str(exporter), "--test-dir", str(test_dir),
                "--targets", str(targets), "--config", str(config),
                "--checkpoint", str(checkpoint), "--data-dir", str(args.data_dir),
                "--output-dir", str(output),
            ]
            for manifest in args.ranking_manifest:
                command += ["--ranking-manifest", str(manifest.resolve())]
            run(command, cwd=project)
        validate_export(output)
        print(f"EXPORT_OK model={name} targets=9 path={output}", flush=True)
    if args.stage == "export":
        print("EXPORT_STAGE_COMPLETE; rerun with --stage score after ground truths are available")
        return

    fb_python = ["conda", "run", "--no-capture-output", "-n", args.foldbench_conda_env, "python"]
    paired: dict[str, dict[str, dict[str, str]]] = {}
    for name in names:
        output = roots[name]
        for algorithm in ALGORITHMS:
            run(fb_python + [
                str(fb_repo / "evaluate.py"),
                "--targets_dir", str(output / "targets"),
                "--evaluation_dir", str(output / "evaluation"),
                "--algorithm_name", algorithm,
                "--ground_truth_dir", str(ground_truth),
                "--targets", "monomer_rna",
            ], cwd=fb_repo)
        run(fb_python + [
            str(fb_repo / "task_score_summary.py"),
            "--evaluation_dir", str(output / "evaluation"),
            "--target_dir", str(output / "targets"),
            "--output_path", str(output / "summary_table.csv"),
            "--algorithm_names", *ALGORITHMS,
            "--targets", "monomer_rna", "--metric_type", "rank",
        ], cwd=fb_repo)
        run([sys.executable, str(summarizer), "--foldbench-dir", str(output)], cwd=project)
        paired[name] = load_paired(output / "paired_lddt.tsv")
    comparison_dir = args.comparison_dir.resolve()
    write_comparison(comparison_dir, paired)
    (comparison_dir / "run_metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(names), "target_count": 9,
        "foldbench_repo": str(fb_repo), "ground_truth_dir": str(ground_truth),
        "conda_env": args.foldbench_conda_env,
        "export_dirs": {name: str(root) for name, root in roots.items()},
    }, indent=2), encoding="utf-8")
    print(f"PAIRED_FOLDBENCH_COMPLETE models={','.join(names)} targets=9 report={comparison_dir}")


if __name__ == "__main__":
    main()
