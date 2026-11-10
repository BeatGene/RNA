#!/usr/bin/env python3
"""Rescue rigid RMSD when another OpenStructure 2.8 score action crashes.

The standard FoldBench invocation may fail before rigid-score computation with
``ValueError: need at least one array to concatenate`` in the all-atom lDDT
stage.  A second audited case reaches RMSD but fails later in USalign TM-score
with ``ERROR! No assignable chain``.  This script selects only those two exact
error contracts and evaluates ``--rigid-scores`` in a separate output tree.  It
never replaces or modifies standard FoldBench outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - rescue execution is Linux-only
    fcntl = None

from run_foldbench_rna_rmsd import (
    Candidate,
    atomic_write_json,
    atomic_write_text,
    load_valid_ost_output,
    read_manifest,
    utc_now,
)


LDDT_EMPTY_ARRAY_MARKERS = (
    "Computing all-atom lDDT",
    "ValueError: need at least one array to concatenate",
)
DOWNSTREAM_TM_FAILURE_MARKERS = (
    "Computing RMSD",
    "Computing patch TM-score",
    "ERROR! No assignable chain",
)


@dataclass(frozen=True)
class RescueResult:
    pdb_id: str
    seed: int
    sample: int
    state: str
    elapsed_seconds: float
    output_path: str
    return_code: Optional[int]
    reason: str = ""


def standard_output_path(root: Path, candidate: Candidate) -> Path:
    return root / "details" / candidate.pdb_id / f"seed_{candidate.seed}" / f"sample_{candidate.sample}.json"


def standard_error_path(root: Path, candidate: Candidate) -> Path:
    return root / "errors" / candidate.pdb_id / f"seed_{candidate.seed}" / f"sample_{candidate.sample}.stderr.txt"


def rescue_output_path(root: Path, candidate: Candidate) -> Path:
    return root / "rigid_only_rescue" / "details" / candidate.pdb_id / f"seed_{candidate.seed}" / f"sample_{candidate.sample}.json"


def rescue_error_path(root: Path, candidate: Candidate) -> Path:
    return root / "rigid_only_rescue" / "errors" / candidate.pdb_id / f"seed_{candidate.seed}" / f"sample_{candidate.sample}.stderr.txt"


def has_audited_lddt_failure(root: Path, candidate: Candidate) -> bool:
    standard, _ = load_valid_ost_output(standard_output_path(root, candidate))
    if standard is not None:
        return False
    error_path = standard_error_path(root, candidate)
    if not error_path.is_file():
        return False
    text = error_path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in LDDT_EMPTY_ARRAY_MARKERS)


def audited_rescue_reason(root: Path, candidate: Candidate) -> Optional[str]:
    standard, _ = load_valid_ost_output(standard_output_path(root, candidate))
    if standard is not None:
        return None
    error_path = standard_error_path(root, candidate)
    if not error_path.is_file():
        return None
    text = error_path.read_text(encoding="utf-8", errors="replace")
    if all(marker in text for marker in LDDT_EMPTY_ARRAY_MARKERS):
        return "LDDT_EMPTY_ARRAY_BEFORE_RIGID_SCORE"
    if all(marker in text for marker in DOWNSTREAM_TM_FAILURE_MARKERS):
        return "DOWNSTREAM_TM_NO_ASSIGNABLE_CHAIN_AFTER_RMSD"
    return None


def has_audited_rescuable_failure(root: Path, candidate: Candidate) -> bool:
    return audited_rescue_reason(root, candidate) is not None


def build_rigid_only_command(ost: str, candidate: Candidate, output: Path) -> list[str]:
    return [
        ost,
        "compare-structures",
        "-m",
        candidate.prediction_path,
        "-r",
        candidate.reference_path,
        "-o",
        str(output),
        "--fault-tolerant",
        "--min-pep-length",
        "4",
        "--min-nuc-length",
        "4",
        "--rigid-scores",
    ]


def evaluate_rescue(
    candidate: Candidate,
    *,
    root: Path,
    ost: str,
    timeout_seconds: int,
) -> RescueResult:
    started = time.monotonic()
    output = rescue_output_path(root, candidate)
    valid, _ = load_valid_ost_output(output)
    if valid is not None:
        return RescueResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "skipped_valid",
            time.monotonic() - started,
            str(output),
            None,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    command = build_rigid_only_command(ost, candidate, temp_path)
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
            os.replace(temp_path, output)
            return RescueResult(
                candidate.pdb_id,
                candidate.seed,
                candidate.sample,
                "success",
                time.monotonic() - started,
                str(output),
                completed.returncode,
            )
        reason = (
            f"return_code={completed.returncode}; {validation_error}; "
            f"stdout={completed.stdout[-4000:]!r}; stderr={completed.stderr[-12000:]!r}"
        )
        atomic_write_text(rescue_error_path(root, candidate), reason + "\n")
        return RescueResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output),
            completed.returncode,
            reason,
        )
    except subprocess.TimeoutExpired as exc:
        reason = f"timeout after {timeout_seconds}s; stderr={str(exc.stderr or '')[-12000:]!r}"
        atomic_write_text(rescue_error_path(root, candidate), reason + "\n")
        return RescueResult(
            candidate.pdb_id,
            candidate.seed,
            candidate.sample,
            "failed",
            time.monotonic() - started,
            str(output),
            None,
            reason,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def acquire_lock(root: Path):
    lock_path = root / "rigid_only_rescue" / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"Another rescue process holds {lock_path}") from exc
    return handle


def write_progress(
    root: Path,
    *,
    total: int,
    completed: int,
    counts: Counter,
    started: float,
) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    remaining = total - completed
    atomic_write_json(
        root / "rigid_only_rescue" / "progress.json",
        {
            "updated_at_utc": utc_now(),
            "total_candidates": total,
            "completed": completed,
            "remaining": remaining,
            "counts": dict(counts),
            "elapsed_seconds": round(elapsed, 3),
            "rate_candidates_per_second": round(rate, 4),
            "estimated_remaining_seconds": round(remaining / rate, 1) if rate else None,
        },
    )


def run(args: argparse.Namespace) -> int:
    root = Path(args.output_root).expanduser().resolve()
    manifest = root / "manifest" / "candidates.csv"
    ost = args.ost_executable or shutil.which("ost")
    if not ost:
        raise FileNotFoundError("Cannot find ost; activate the foldbench environment")
    candidates = [
        item
        for item in read_manifest(manifest)
        if has_audited_rescuable_failure(root, item)
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    metadata = {
        "started_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "target_count": len({item.pdb_id for item in candidates}),
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "selection_contracts": {
            "LDDT_EMPTY_ARRAY_BEFORE_RIGID_SCORE": list(LDDT_EMPTY_ARRAY_MARKERS),
            "DOWNSTREAM_TM_NO_ASSIGNABLE_CHAIN_AFTER_RMSD": list(
                DOWNSTREAM_TM_FAILURE_MARKERS
            ),
        },
        "score_protocol": "rigid_only_rescue",
        "command_options": [
            "--fault-tolerant",
            "--min-pep-length 4",
            "--min-nuc-length 4",
            "--rigid-scores",
        ],
    }
    atomic_write_json(root / "rigid_only_rescue" / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if not candidates:
        print("No candidates match the audited lDDT empty-array failure.", flush=True)
        return 0

    lock = acquire_lock(root)
    started = time.monotonic()
    counts: Counter = Counter()
    completed_count = 0
    event_path = root / "rigid_only_rescue" / "task_events.jsonl"
    try:
        with event_path.open("a", encoding="utf-8", buffering=1) as event_handle:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                candidate_iter = iter(candidates)
                pending = {}

                def submit_one() -> bool:
                    try:
                        item = next(candidate_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        evaluate_rescue,
                        item,
                        root=root,
                        ost=str(ost),
                        timeout_seconds=args.timeout_seconds,
                    )
                    pending[future] = item
                    return True

                while len(pending) < args.workers * 2 and submit_one():
                    pass
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = pending.pop(future)
                        try:
                            result = future.result()
                        except BaseException as exc:
                            result = RescueResult(
                                item.pdb_id,
                                item.seed,
                                item.sample,
                                "failed",
                                0.0,
                                str(rescue_output_path(root, item)),
                                None,
                                f"worker exception: {type(exc).__name__}: {exc}",
                            )
                        completed_count += 1
                        counts[result.state] += 1
                        event_handle.write(
                            json.dumps(
                                {"recorded_at_utc": utc_now(), **asdict(result)},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        event_handle.flush()
                        if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(candidates):
                            write_progress(
                                root,
                                total=len(candidates),
                                completed=completed_count,
                                counts=counts,
                                started=started,
                            )
                            print(
                                f"progress {completed_count}/{len(candidates)} "
                                f"success={counts['success']} "
                                f"skipped={counts['skipped_valid']} failed={counts['failed']}",
                                flush=True,
                            )
                        submit_one()
        summary = {
            **metadata,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "counts": dict(counts),
            "orchestration_completed": completed_count == len(candidates),
        }
        atomic_write_json(root / "rigid_only_rescue" / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="~/Json_data/Foldbench_evaluation/rmsd")
    parser.add_argument("--ost-executable", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be at least 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
