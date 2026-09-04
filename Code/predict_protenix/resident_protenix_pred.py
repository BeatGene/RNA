#!/usr/bin/env python3
"""Run one Protenix v1.0.5 model instance over a multi-target JSON.

This uses the official runner API, but keeps the historical per-PDB/per-seed
directory layout required by the downstream RNA refinement pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Heartbeat:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.lock = threading.Lock()
        self.details: dict[str, object] = {}

    def _write(self, status: str) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "status": status,
                        "pid": os.getpid(),
                        "updated_at_utc": utc_now(),
                        **self.details,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)

    def update(self, **details: object) -> None:
        with self.lock:
            self.details.update(details)
        self._write("RUNNING")

    def _loop(self) -> None:
        while not self.stop.wait(30):
            self._write("RUNNING")

    def __enter__(self) -> "Heartbeat":
        self._write("RUNNING")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        self._write("FAILED" if exc_type else "COMPLETE")


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds 必须是不重复的逗号分隔整数")
    return seeds


def cgroup_memory_percent() -> float | None:
    root = Path("/sys/fs/cgroup")
    try:
        current = int((root / "memory.current").read_text().strip())
        maximum_text = (root / "memory.max").read_text().strip()
        if maximum_text == "max":
            return None
        maximum = int(maximum_text)
    except (OSError, ValueError):
        return None
    return 100.0 * current / maximum if maximum > 0 else None


def append_failure(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def run(args: argparse.Namespace, heartbeat: Heartbeat) -> None:
    # Imports intentionally happen after CUDA_VISIBLE_DEVICES is set by the
    # parent launcher and after basic argument validation.
    from runner.batch_inference import get_default_runner
    from runner.dumper import DataDumper
    from runner.inference import infer_predict

    class ConfigurableLayoutDumper(DataDumper):
        def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
            pdb_id = str(sample_name).strip().lower()
            if args.output_layout == "dataset":
                return str(args.output_dir / pdb_id / f"seed_{seed}")
            return str(
                args.output_dir
                / f"pred_output_{pdb_id}_seed_{seed}"
                / pdb_id
                / f"seed_{seed}"
            )

    runner = get_default_runner(
        seeds=args.seeds,
        n_cycle=args.cycles,
        n_step=args.steps,
        n_sample=args.samples,
        dtype=args.dtype,
        model_name=args.model,
        use_msa=True,
        trimul_kernel="cuequivariance",
        triatt_kernel="cuequivariance",
        enable_cache=True,
        enable_fusion=True,
        enable_tf32=True,
        use_template=False,
        use_rna_msa=True,
        use_seeds_in_json=False,
        need_atom_confidence=args.need_atom_confidence,
        kalign_binary_path=None,
    )
    runner.dumper = ConfigurableLayoutDumper(
        base_dir=str(args.output_dir),
        need_atom_confidence=args.need_atom_confidence,
        sorted_by_ranking_score=True,
    )
    tasks = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("input JSON 必须是非空列表")

    work_dir = args.heartbeat.parent / "worker_inputs" / args.heartbeat.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    # Protenix v1.0.5 writes data-error diagnostics to this hard-coded relative
    # directory.  If it is absent, an otherwise recoverable bad target raises a
    # second FileNotFoundError and terminates the whole resident batch.
    protenix_error_dir = Path.cwd() / "output" / "ERR"
    protenix_error_dir.mkdir(parents=True, exist_ok=True)
    failure_path = (
        args.heartbeat.parent / "worker_failures" / f"{args.heartbeat.stem}.jsonl"
    )
    total = len(tasks)
    failed_targets = 0
    consecutive_exceptions = 0
    for index, task in enumerate(tasks, start=1):
        sample_name = str(task.get("name", f"task_{index}"))
        memory_percent = cgroup_memory_percent()
        heartbeat.update(
            current_target=sample_name,
            target_index=index,
            target_total=total,
            cgroup_memory_percent=(
                round(memory_percent, 2) if memory_percent is not None else None
            ),
        )
        if memory_percent is not None and memory_percent >= args.memory_stop_percent:
            raise MemoryError(
                f"cgroup memory {memory_percent:.1f}% >= "
                f"stop threshold {args.memory_stop_percent:.1f}%"
            )

        task_json = work_dir / f"task_{index:05d}_{sample_name}.json"
        temporary = task_json.with_name(f".{task_json.name}.tmp")
        temporary.write_text(
            json.dumps([task], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, task_json)
        runner.configs["input_json_path"] = str(task_json)
        protenix_error = protenix_error_dir / f"{sample_name}.txt"
        error_signature_before = (
            (protenix_error.stat().st_mtime_ns, protenix_error.stat().st_size)
            if protenix_error.is_file()
            else None
        )
        try:
            infer_predict(runner, runner.configs)
        except Exception as error:
            failed_targets += 1
            consecutive_exceptions += 1
            append_failure(
                failure_path,
                {
                    "target": sample_name,
                    "target_index": index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": utc_now(),
                },
            )
            heartbeat.update(
                current_target=sample_name,
                targets_attempted=index,
                failed_targets=failed_targets,
                last_failed_target=sample_name,
                last_error=f"{type(error).__name__}: {error}",
            )
            # One malformed PDB must not discard hundreds of valid targets. If
            # failures are consecutive, assume a systemic worker problem and
            # stop so the parent can retry safely instead of churning the batch.
            if consecutive_exceptions >= 3:
                raise RuntimeError(
                    "three consecutive target exceptions; stopping resident worker"
                ) from error
        else:
            error_signature_after = (
                (protenix_error.stat().st_mtime_ns, protenix_error.stat().st_size)
                if protenix_error.is_file()
                else None
            )
            if (
                error_signature_after is not None
                and error_signature_after != error_signature_before
            ):
                failed_targets += 1
                consecutive_exceptions += 1
                error_text = protenix_error.read_text(
                    encoding="utf-8", errors="replace"
                )
                append_failure(
                    failure_path,
                    {
                        "target": sample_name,
                        "target_index": index,
                        "error_type": "ProtenixTargetError",
                        "error": error_text[-8000:],
                        "failed_at_utc": utc_now(),
                    },
                )
                heartbeat.update(
                    current_target=sample_name,
                    targets_attempted=index,
                    failed_targets=failed_targets,
                    last_failed_target=sample_name,
                    last_error="ProtenixTargetError; see worker failure log",
                )
                if consecutive_exceptions >= 3:
                    raise RuntimeError(
                        "three consecutive Protenix target errors; "
                        "stopping resident worker"
                    )
            else:
                consecutive_exceptions = 0
        finally:
            task_json.unlink(missing_ok=True)
        heartbeat.update(
            current_target=sample_name,
            targets_attempted=index,
            targets_finished=index - failed_targets,
            failed_targets=failed_targets,
            target_total=total,
        )
        time.sleep(0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protenix 常驻 GPU 批量预测 worker")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    parser.add_argument("--model", default="protenix_base_default_v1.0.0")
    parser.add_argument(
        "--need-atom-confidence",
        action="store_true",
        help=(
            "保存每个 sample 的 full_data JSON（atom_plddt、token_pair_pae、"
            "token_pair_pde、contact_probs、atom_to_token_idx）"
        ),
    )
    parser.add_argument(
        "--output-layout",
        choices=("legacy", "dataset"),
        default="legacy",
        help=(
            "legacy: pred_output_<pdb>_seed_<seed>/...；"
            "dataset: <pdb>/seed_<seed>/..."
        ),
    )
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument(
        "--memory-stop-percent",
        type=float,
        default=80.0,
        help="每个 PDB 前检查 cgroup；达到该百分比则安全退出供之后续跑",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("input_json", "output_dir", "heartbeat"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.input_json.is_file():
        raise FileNotFoundError(args.input_json)
    if args.samples < 1 or args.steps < 1 or args.cycles < 1:
        raise ValueError("samples/steps/cycles 必须大于 0")
    if not 1 <= args.memory_stop_percent <= 100:
        raise ValueError("--memory-stop-percent 必须在 1 到 100 之间")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Heartbeat(args.heartbeat) as heartbeat:
        run(args, heartbeat)


if __name__ == "__main__":
    main()
