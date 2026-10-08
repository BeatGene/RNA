"""Build a val/test dataset with one highest-Protenix-ranking-score .pt per PDB.

Only original Protenix summary-confidence scores are used. The input dataset is
left intact; the output contains symlinks (or copies) of selected .pt files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil


SEED_RE = re.compile(r"seed_(\d+)$")
SAMPLE_RE = re.compile(r"sample_(\d+)\.pt$")


@dataclass(frozen=True)
class Candidate:
    split: str
    pdb_id: str
    seed: int
    sample: int
    pt_path: Path


def discover(data_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, int, int]] = set()
    for split in ("val", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing .pt split directory: {split_dir}")
        for path in sorted(split_dir.rglob("*.pt")):
            relative = path.relative_to(split_dir)
            if len(relative.parts) != 3:
                raise ValueError(f"Unexpected .pt layout (expected PDB/seed_N/sample_M.pt): {path}")
            pdb, seed_dir, sample_name = relative.parts
            seed_match, sample_match = SEED_RE.fullmatch(seed_dir), SAMPLE_RE.fullmatch(sample_name)
            if not pdb or not seed_match or not sample_match:
                raise ValueError(f"Invalid .pt candidate path: {path}")
            candidate = Candidate(split, pdb.upper(), int(seed_match[1]), int(sample_match[1]), path)
            key = (candidate.split, candidate.pdb_id, candidate.seed, candidate.sample)
            if key in seen:
                raise ValueError(f"Duplicate candidate {key}: {path}")
            seen.add(key)
            candidates.append(candidate)
        if not any(candidate.split == split for candidate in candidates):
            raise ValueError(f"No .pt candidates in {split_dir}")
    return candidates


def read_manifest_scores(paths: list[Path]) -> dict[str, tuple[float, str]]:
    scores: dict[str, tuple[float, str]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not row.get("ranking_score") or not row.get("output"):
                    continue
                output = Path(row["output"])
                if not output.is_absolute():
                    raise ValueError(f"Manifest output path is not absolute: {path}: {row['output']}")
                value = float(row["ranking_score"])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite ranking score in {path}: {row}")
                key = str(output.resolve())
                if key in scores and not math.isclose(scores[key][0], value, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Conflicting manifest scores for {key}")
                scores[key] = value, str(path)
    return scores


def ranking_score(candidate: Candidate, prediction_root: Path,
                  manifest_scores: dict[str, tuple[float, str]]) -> tuple[float, str]:
    """Look up exactly this PDB/seed/sample, resolving relocated split links."""
    found: dict[str, Path] = {}
    for split in dict.fromkeys((candidate.split, "test", "val", "train")):
        directory = prediction_root / split / candidate.pdb_id.lower() / f"seed_{candidate.seed}" / "predictions"
        for path in directory.glob(f"*_summary_confidence_sample_{candidate.sample}.json"):
            prefix = path.name.removesuffix(f"_summary_confidence_sample_{candidate.sample}.json")
            if prefix.upper() == candidate.pdb_id:
                found[str(path.resolve())] = path
    if len(found) > 1:
        raise ValueError(f"Ambiguous prediction runs for {candidate.pt_path}: {list(found.values())}")
    if found:
        path = next(iter(found.values()))
        try:
            value = float(json.loads(path.read_text(encoding="utf-8"))["ranking_score"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid Protenix ranking_score in {path}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite Protenix ranking_score in {path}")
        return value, str(path)
    manifest = manifest_scores.get(str(candidate.pt_path.resolve()))
    if manifest is not None:
        return manifest
    raise FileNotFoundError(
        f"No original Protenix ranking_score for {candidate.pt_path}; "
        f"searched {prediction_root}/{{test,val,train}}/{candidate.pdb_id.lower()}/"
        f"seed_{candidate.seed}/predictions"
    )


def select(candidates: list[Candidate], prediction_root: Path,
           manifest_scores: dict[str, tuple[float, str]],
           incomplete_policy: str) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.split, candidate.pdb_id].append(candidate)
    selected, coverage = [], []
    errors = []
    for (split, pdb_id), group in sorted(grouped.items()):
        ranked = []
        for candidate in group:
            try:
                score, source = ranking_score(candidate, prediction_root, manifest_scores)
                ranked.append((score, -candidate.seed, -candidate.sample, candidate, source))
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
        if len(ranked) != len(group):
            continue
        winner = max(ranked)
        score, _, _, candidate, source = winner
        samples_by_seed: dict[int, set[int]] = defaultdict(set)
        for item in group:
            samples_by_seed[item.seed].add(item.sample)
        complete = len(samples_by_seed) == 50 and all(
            samples == {0, 1, 2, 3} for samples in samples_by_seed.values()
        )
        keep = complete or incomplete_policy == "select"
        coverage.append({
            "split": split, "pdb_id": pdb_id, "candidate_count": len(group),
            "seed_count": len(samples_by_seed),
            "complete_50x4": int(complete), "kept": int(keep),
            "best_seed": candidate.seed, "best_sample": candidate.sample,
            "best_ranking_score": score,
        })
        if keep:
            selected.append({
                "split": split, "pdb_id": pdb_id, "seed": candidate.seed,
                "sample": candidate.sample, "ranking_score": score,
                "candidate_count": len(group), "complete_50x4": int(complete),
                "score_source": source, "source_pt": str(candidate.pt_path.resolve()),
            })
    if errors:
        raise ValueError(
            f"Ranking scores missing/invalid for {len(errors)} .pt candidates; "
            "selection stopped before writing anything. First errors:\n" + "\n".join(errors[:12])
        )
    return selected, coverage


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty selection: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Data_PT_V2")
    parser.add_argument("--prediction-root", type=Path, default=Path.home() / "Data_V2")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "Data_PT_V2_rank1")
    parser.add_argument("--incomplete-policy", choices=("select", "skip"), default="select",
                        help="select highest among available .pt, or skip PDBs lacking 50 seeds x 4 samples")
    parser.add_argument("--materialize", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--manifest", type=Path, action="append", default=[],
                        help="Optional PT build manifest.tsv if original summary JSON is missing")
    parser.add_argument("--dry-run", action="store_true", help="Check scores and print counts without writing")
    args = parser.parse_args()
    data_root, prediction_root, output_root = args.data_root.resolve(), args.prediction_root.resolve(), args.output_root.resolve()
    if output_root == data_root or data_root in output_root.parents:
        parser.error("--output-root must be outside --data-root")
    if output_root.exists():
        parser.error(f"output root already exists; use a new path: {output_root}")
    candidates = discover(data_root)
    manifest_paths = sorted((data_root / "logs").glob("**/manifest.tsv")) + args.manifest
    selected, coverage = select(
        candidates, prediction_root, read_manifest_scores(manifest_paths), args.incomplete_policy,
    )
    counts = Counter(row["split"] for row in selected)
    if counts["val"] == 0 or counts["test"] == 0:
        raise ValueError(f"Both splits must retain targets; got {dict(counts)}")
    summary = {
        "selection_rule": "highest original Protenix ranking_score among available .pt candidates per split/PDB",
        "tie_break": "lower seed, then lower sample",
        "incomplete_policy": args.incomplete_policy,
        "materialize": args.materialize,
        "data_root": str(data_root), "prediction_root": str(prediction_root),
        "candidate_count": dict(Counter(c.split for c in candidates)),
        "pdb_count": dict(Counter(row["split"] for row in coverage)),
        "selected_count": dict(counts),
        "incomplete_pdb_count": dict(Counter(row["split"] for row in coverage if not row["complete_50x4"])),
        "output_root": str(output_root),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    output_root.mkdir(parents=True)
    for row in selected:
        destination = output_root / row["split"] / row["pdb_id"].lower() / f"seed_{row['seed']}" / f"sample_{row['sample']}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.materialize == "symlink":
            destination.symlink_to(row["source_pt"])
        else:
            shutil.copy2(row["source_pt"], destination)
        row["selected_pt"] = str(destination)
    write_tsv(output_root / "selection.tsv", selected)
    write_tsv(output_root / "coverage.tsv", coverage)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RANK1_DATASET_COMPLETE val={counts['val']} test={counts['test']} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
