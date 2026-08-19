#!/usr/bin/env python3
"""Run FoldBench/OpenStructure RNA structure evaluation at candidate scale.

The script discovers every primary Protenix sample CIF under directories named
``pred_output_<PDB>_seed_<SEED>`` and evaluates it with the same
``ost compare-structures`` options used by FoldBench.  Outputs are written
atomically and validated, so an interrupted run can safely be restarted.

This runner intentionally depends only on the Python standard library.  The
separate reporting script owns pandas/matplotlib/openpyxl dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:  # Linux server lock; import fallback keeps discovery/tests portable.
    import fcntl
except ImportError:  # pragma: no cover - only exercised on non-POSIX hosts
    fcntl = None


PRED_DIR_RE = re.compile(
    r"^pred_output_(?P<pdb_id>[0-9A-Za-z]{4})_seed_(?P<seed>[0-9]+)$",
    re.IGNORECASE,
)
PRED_CIF_RE = re.compile(
    r"^(?P<pdb_id>[0-9A-Za-z]{4})_sample_(?P<sample>[0-9]+)\.cif$",
    re.IGNORECASE,
)
CONFIDENCE_RE = re.compile(
    r"^(?P<pdb_id>[0-9A-Za-z]{4})_summary_confidence_sample_"
    r"(?P<sample>[0-9]+)\.json$",
    re.IGNORECASE,
)
EXPECTED_SEEDS = (42, 66, 101, 2024, 8888)
EXPECTED_SAMPLES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Candidate:
    pdb_id: str
    seed: int
    sample: int
    prediction_path: str
    reference_path: str
    confidence_path: str
    ranking_score: Optional[float]
    discovery_issue: str = ""

    @property
    def key(self) -> str:
        return f"{self.pdb_id}_seed_{self.seed}_sample_{self.sample}"


@dataclass(frozen=True)
class TaskResult:
    pdb_id: str
    seed: int
    sample: int
    state: str
    elapsed_seconds: float
    output_path: str
    return_code: Optional[int]
    reason: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def index_reference_cifs(reference_root: Path) -> dict[str, Path]:
    if not reference_root.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {reference_root}")
    result: dict[str, Path] = {}
    for path in reference_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".cif":
            continue
        pdb_id = path.stem.upper()
        if not re.fullmatch(r"[0-9A-Z]{4}", pdb_id):
            continue
        if pdb_id in result:
            raise ValueError(
                f"Duplicate case-insensitive reference CIF for {pdb_id}: "
                f"{result[pdb_id]} and {path}"
            )
        # reference_root is already absolute; avoid an NFS stat-heavy resolve().
        result[pdb_id] = path
    if not result:
        raise ValueError(f"No four-character PDB CIFs found in {reference_root}")
    return result


def read_ranking_score(path: Optional[Path]) -> tuple[Optional[float], str]:
    if path is None:
        return None, "matching summary_confidence JSON is missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return None, f"cannot parse confidence JSON: {type(exc).__name__}: {exc}"
    score = finite_float(payload.get("ranking_score")) if isinstance(payload, dict) else None
    if score is None:
        return None, "ranking_score is missing or not finite"
    return score, ""


def discover_candidates(
    prediction_root: Path,
    reference_root: Path,
    *,
    target_ids: Optional[set[str]] = None,
    max_candidates: Optional[int] = None,
) -> tuple[list[Candidate], list[str]]:
    """Discover primary candidate CIFs and their confidence/reference files."""
    if not prediction_root.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {prediction_root}")
    reference_by_id = index_reference_cifs(reference_root)
    candidates: list[Candidate] = []
    issues: list[str] = []
    seen_keys: set[tuple[str, int, int]] = set()

    for seed_root in sorted(prediction_root.iterdir(), key=lambda p: p.name.lower()):
        if not seed_root.is_dir():
            continue
        seed_match = PRED_DIR_RE.fullmatch(seed_root.name)
        if seed_match is None:
            continue
        pdb_id = seed_match.group("pdb_id").upper()
        if target_ids is not None and pdb_id not in target_ids:
            continue
        seed = int(seed_match.group("seed"))
        raw_pdb_id = seed_match.group("pdb_id")
        expected_dir = seed_root / raw_pdb_id / f"seed_{seed}" / "predictions"
        if expected_dir.is_dir():
            prediction_dirs = [expected_dir]
        else:
            # Bounded two-level fallback for case/layout variants.  Do not use
            # rglob here: recursively stat-ing every CIF/JSON is very slow on NFS.
            prediction_dirs = []
            for pdb_dir in seed_root.iterdir():
                if not pdb_dir.is_dir():
                    continue
                for nested_seed_dir in pdb_dir.iterdir():
                    candidate_dir = nested_seed_dir / "predictions"
                    if nested_seed_dir.is_dir() and candidate_dir.is_dir():
                        prediction_dirs.append(candidate_dir)
            prediction_dirs.sort(key=lambda path: str(path).lower())
        if len(prediction_dirs) != 1:
            issues.append(
                f"{pdb_id} seed {seed}: predictions_dir_count="
                f"{len(prediction_dirs)} (expected 1)"
            )
            continue
        prediction_dir = prediction_dirs[0]

        confidence_by_sample: dict[int, Path] = {}
        cif_by_sample: dict[int, Path] = {}
        for path in prediction_dir.iterdir():
            if not path.is_file():
                continue
            cif_match = PRED_CIF_RE.fullmatch(path.name)
            if cif_match and cif_match.group("pdb_id").upper() == pdb_id:
                sample = int(cif_match.group("sample"))
                if sample in cif_by_sample:
                    raise ValueError(
                        f"Duplicate primary CIF for {pdb_id} seed {seed} sample {sample}: "
                        f"{cif_by_sample[sample]} and {path}"
                    )
                cif_by_sample[sample] = path
                continue
            confidence_match = CONFIDENCE_RE.fullmatch(path.name)
            if confidence_match and confidence_match.group("pdb_id").upper() == pdb_id:
                sample = int(confidence_match.group("sample"))
                if sample in confidence_by_sample:
                    raise ValueError(
                        f"Duplicate confidence JSON for {pdb_id} seed {seed} "
                        f"sample {sample}"
                    )
                confidence_by_sample[sample] = path

        if not cif_by_sample:
            issues.append(f"{pdb_id} seed {seed}: no primary sample CIFs found")
            continue
        for sample, cif_path in sorted(cif_by_sample.items()):
            key = (pdb_id, seed, sample)
            if key in seen_keys:
                raise ValueError(f"Duplicate candidate key: {key}")
            seen_keys.add(key)
            confidence_path = confidence_by_sample.get(sample)
            ranking_score, confidence_issue = read_ranking_score(confidence_path)
            reference_path = reference_by_id.get(pdb_id)
            issue_parts = []
            if confidence_issue:
                issue_parts.append(confidence_issue)
            if reference_path is None:
                issue_parts.append("reference CIF is missing")
            candidates.append(
                Candidate(
                    pdb_id=pdb_id,
                    seed=seed,
                    sample=sample,
                    prediction_path=str(cif_path),
                    reference_path=str(reference_path) if reference_path else "",
                    confidence_path=str(confidence_path) if confidence_path else "",
                    ranking_score=ranking_score,
                    discovery_issue="; ".join(issue_parts),
                )
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                candidates.sort(key=lambda c: (c.pdb_id, c.seed, c.sample))
                return candidates[:max_candidates], issues

    candidates.sort(key=lambda c: (c.pdb_id, c.seed, c.sample))
    if not candidates:
        raise ValueError(f"No primary prediction CIFs found in {prediction_root}")
    return candidates, issues


def candidate_output_path(output_root: Path, candidate: Candidate) -> Path:
    return (
        output_root
        / "details"
        / candidate.pdb_id
        / f"seed_{candidate.seed}"
        / f"sample_{candidate.sample}.json"
    )


def candidate_error_path(output_root: Path, candidate: Candidate) -> Path:
    return (
        output_root
        / "errors"
        / candidate.pdb_id
        / f"seed_{candidate.seed}"
        / f"sample_{candidate.sample}.stderr.txt"
    )


def load_valid_ost_output(path: Path) -> tuple[Optional[dict], str]:
    if not path.is_file():
        return None, "output JSON does not exist"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return None, f"invalid JSON: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "output JSON is not an object"
    status = str(payload.get("status", "")).upper()
    if status != "SUCCESS":
        return None, f"OST status is {payload.get('status')!r}, expected 'SUCCESS'"
    rmsd = finite_float(payload.get("rmsd"))
    if rmsd is None or rmsd < 0:
        return None, f"RMSD is not a finite non-negative number: {payload.get('rmsd')!r}"
    return payload, ""


def build_ost_command(ost_executable: str, candidate: Candidate, output_path: Path) -> list[str]:
    return [
        ost_executable,
        "compare-structures",
        "-m",
        candidate.prediction_path,
        "-r",
        candidate.reference_path,
        "-o",
        str(output_path),
        "--fault-tolerant",
        "--min-pep-length",
        "4",
        "--min-nuc-length",
        "4",
        "--lddt",
        "--rigid-scores",
        "--tm-score",
        "--dockq",
    ]


def evaluate_candidate(
    candidate: Candidate,
    *,
    output_root: Path,
    ost_executable: str,
    timeout_seconds: int,
) -> TaskResult:
    started = time.monotonic()
    output_path = candidate_output_path(output_root, candidate)
    valid_existing, _ = load_valid_ost_output(output_path)
    if valid_existing is not None:
        return TaskResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "skipped_valid",
            time.monotonic() - started,
            str(output_path),
            None,
        )
    if not candidate.reference_path:
        return TaskResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output_path),
            None,
            "reference CIF is missing",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)  # OST expects to create the output itself.
    command = build_ost_command(ost_executable, candidate, temp_path)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        payload, validation_error = load_valid_ost_output(temp_path)
        if completed.returncode == 0 and payload is not None:
            os.replace(temp_path, output_path)
            return TaskResult(
                candidate.pdb_id,
                candidate.seed,
                candidate.sample,
                "success",
                time.monotonic() - started,
                str(output_path),
                completed.returncode,
            )
        reason = (
            f"return_code={completed.returncode}; {validation_error}; "
            f"stdout={completed.stdout[-4000:]!r}; stderr={completed.stderr[-12000:]!r}"
        )
        atomic_write_text(candidate_error_path(output_root, candidate), reason + "\n")
        return TaskResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output_path),
            completed.returncode,
            reason,
        )
    except subprocess.TimeoutExpired as exc:
        reason = f"timeout after {timeout_seconds}s; stderr={str(exc.stderr or '')[-12000:]!r}"
        atomic_write_text(candidate_error_path(output_root, candidate), reason + "\n")
        return TaskResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output_path),
            None,
            reason,
        )
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_write_text(candidate_error_path(output_root, candidate), reason)
        return TaskResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output_path),
            None,
            reason,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def write_manifest(candidates: Iterable[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Candidate.__dataclass_fields__)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(asdict(candidate))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def read_manifest(path: Path) -> list[Candidate]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {path}")
    candidates = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(Candidate.__dataclass_fields__)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            score_text = str(row.get("ranking_score", "") or "").strip()
            score = finite_float(score_text) if score_text else None
            candidates.append(
                Candidate(
                    pdb_id=str(row["pdb_id"]).upper(),
                    seed=int(row["seed"]),
                    sample=int(row["sample"]),
                    prediction_path=str(row["prediction_path"]),
                    reference_path=str(row["reference_path"]),
                    confidence_path=str(row["confidence_path"]),
                    ranking_score=score,
                    discovery_issue=str(row.get("discovery_issue", "") or ""),
                )
            )
    duplicate_keys = [
        key
        for key, count in Counter(
            (item.pdb_id, item.seed, item.sample) for item in candidates
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"Duplicate candidate keys in {path}: {duplicate_keys[:10]}")
    return sorted(candidates, key=lambda item: (item.pdb_id, item.seed, item.sample))


def append_jsonl(handle, payload: object) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def write_progress(
    path: Path,
    *,
    total: int,
    completed: int,
    counts: dict[str, int],
    started_monotonic: float,
) -> None:
    elapsed = time.monotonic() - started_monotonic
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    payload = {
        "updated_at_utc": utc_now(),
        "total_candidates": total,
        "completed_this_invocation": completed,
        "remaining_this_invocation": remaining,
        "counts": counts,
        "elapsed_seconds": round(elapsed, 3),
        "rate_candidates_per_second": round(rate, 4),
        "estimated_remaining_seconds": round(remaining / rate, 1) if rate > 0 else None,
    }
    atomic_write_json(path, payload)


def acquire_run_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root / ".run.lock").open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_handle.close()
            raise RuntimeError(
                f"Another evaluation process holds {output_root / '.run.lock'}"
            ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at_utc={utc_now()}\n")
    lock_handle.flush()
    return lock_handle


def run_evaluation(args: argparse.Namespace) -> int:
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ost_executable = args.ost_executable or shutil.which("ost")
    if not ost_executable:
        raise FileNotFoundError("Cannot find 'ost'; activate the foldbench environment")
    ost_executable = str(Path(ost_executable).resolve())

    lock_handle = acquire_run_lock(output_root)
    try:
        manifest_path = output_root / "manifest" / "candidates.csv"
        requested = {value.upper() for value in args.target_id} or None
        scan_started = time.monotonic()
        if manifest_path.is_file() and not args.refresh_manifest:
            print(f"Reusing candidate manifest: {manifest_path}", flush=True)
            candidates = read_manifest(manifest_path)
            scan_issues = []
            if requested is not None:
                candidates = [item for item in candidates if item.pdb_id in requested]
            if args.limit is not None:
                candidates = candidates[: args.limit]
        else:
            scope = "all candidates" if args.limit is None else f"first {args.limit} candidates"
            print(
                f"Scanning prediction layout for {scope}: {prediction_root}",
                flush=True,
            )
            candidates, scan_issues = discover_candidates(
                prediction_root,
                reference_root,
                target_ids=requested,
                max_candidates=args.limit,
            )
        print(
            f"Candidate discovery ready: candidates={len(candidates)}, "
            f"targets={len({item.pdb_id for item in candidates})}, "
            f"elapsed={time.monotonic() - scan_started:.1f}s",
            flush=True,
        )
        if not candidates:
            raise ValueError("Candidate selection is empty")

        if args.limit is None and not args.target_id:
            write_manifest(candidates, manifest_path)
            atomic_write_text(
                output_root / "manifest" / "scan_issues.tsv",
                "issue\n" + "".join(f"{issue}\n" for issue in scan_issues),
            )
        else:
            write_manifest(candidates, output_root / "manifest" / "last_subset.csv")

        metadata = {
            "started_at_utc": utc_now(),
            "pid": os.getpid(),
            "prediction_root": str(prediction_root),
            "reference_root": str(reference_root),
            "output_root": str(output_root),
            "ost_executable": ost_executable,
            "workers": args.workers,
            "timeout_seconds": args.timeout_seconds,
            "candidate_count": len(candidates),
            "target_count": len({candidate.pdb_id for candidate in candidates}),
            "command_options": [
                "--fault-tolerant",
                "--min-pep-length 4",
                "--min-nuc-length 4",
                "--lddt",
                "--rigid-scores",
                "--tm-score",
                "--dockq",
            ],
        }
        atomic_write_json(output_root / "run_metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

        event_path = output_root / "logs" / "task_events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"success": 0, "skipped_valid": 0, "failed": 0}
        started = time.monotonic()
        completed_count = 0
        max_pending = max(args.workers * 2, args.workers)
        candidate_iter = iter(candidates)

        def submit_next(executor, pending) -> bool:
            try:
                candidate = next(candidate_iter)
            except StopIteration:
                return False
            future = executor.submit(
                evaluate_candidate,
                candidate,
                output_root=output_root,
                ost_executable=ost_executable,
                timeout_seconds=args.timeout_seconds,
            )
            pending[future] = candidate
            return True

        with event_path.open("a", encoding="utf-8", buffering=1) as event_handle:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending = {}
                while len(pending) < max_pending and submit_next(executor, pending):
                    pass
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        candidate = pending.pop(future)
                        try:
                            result = future.result()
                        except BaseException as exc:
                            result = TaskResult(
                                candidate.pdb_id,
                                candidate.seed,
                                candidate.sample,
                                "failed",
                                0.0,
                                str(candidate_output_path(output_root, candidate)),
                                None,
                                f"worker exception: {type(exc).__name__}: {exc}",
                            )
                        counts[result.state] = counts.get(result.state, 0) + 1
                        completed_count += 1
                        append_jsonl(
                            event_handle,
                            {"recorded_at_utc": utc_now(), **asdict(result)},
                        )
                        if (
                            completed_count == 1
                            or completed_count % args.progress_every == 0
                            or completed_count == len(candidates)
                        ):
                            write_progress(
                                output_root / "progress.json",
                                total=len(candidates),
                                completed=completed_count,
                                counts=counts,
                                started_monotonic=started,
                            )
                            print(
                                f"progress {completed_count}/{len(candidates)} "
                                f"success={counts['success']} "
                                f"skipped={counts['skipped_valid']} "
                                f"failed={counts['failed']}",
                                flush=True,
                            )
                        submit_next(executor, pending)

        summary = {
            **metadata,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "counts": counts,
            "orchestration_completed": completed_count == len(candidates),
        }
        atomic_write_json(output_root / "run_summary.json", summary)
        print("Final run summary:", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable FoldBench/OpenStructure RNA RMSD evaluation."
    )
    parser.add_argument(
        "--prediction-root",
        default="~/Json_data/Foldbench_predictions",
        help="Directory containing pred_output_<PDB>_seed_<SEED> folders.",
    )
    parser.add_argument(
        "--reference-root",
        default="~/pdb_data",
        help="Directory containing original <PDB>.cif files.",
    )
    parser.add_argument(
        "--output-root",
        default="~/Json_data/Foldbench_evaluation/rmsd",
        help="Stable output root; valid existing results are skipped.",
    )
    parser.add_argument("--ost-executable", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rescan the prediction tree even when a full candidates.csv exists.",
    )
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="Evaluate only this PDB ID; may be supplied more than once.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
