#!/usr/bin/env python3
"""Sequence-cached RNA prep for the pure-RNA Protenix dataset.

The expensive unit of work is an RNA *sequence*, not a PDB entry.  This
program resolves every unique sequence in the following order:

1. a verified MSA already produced by an existing complete prep;
2. optionally, the official Protenix wwPDB RNA-MSA archive;
3. one new ``protenix prep`` search for the still-unresolved sequence.

It then creates one final-updated JSON per PDB.  Multiple PDBs may safely
reference the same A3M.  Compatibility symlinks are also created under each
``prep_output_<pdb>`` directory when the destination does not already exist.
No existing complete prep is overwritten or deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import stage2_decoy_pipeline as pipeline


@dataclass(frozen=True)
class MsaSource:
    sequence: str
    path: Path
    source_type: str
    source_id: str


class Heartbeat:
    """Persist liveness/progress so a vanished container process is visible."""

    def __init__(self, report_dir: Path) -> None:
        self.path = report_dir / "heartbeat.json"
        self.started_at = pipeline.utc_now()
        self.state: dict[str, Any] = {
            "status": "STARTING",
            "phase": "initializing",
            "pid": os.getpid(),
            "started_at_utc": self.started_at,
            "updated_at_utc": self.started_at,
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _write(self) -> None:
        with self.lock:
            payload = dict(self.state)
            payload["updated_at_utc"] = pipeline.utc_now()
            self.state["updated_at_utc"] = payload["updated_at_utc"]
        pipeline.atomic_write_text(
            self.path, json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _loop(self) -> None:
        while not self.stop_event.wait(30):
            self._write()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self.thread.start()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)
        self._write()

    def finish(self, status: str, **values: Any) -> None:
        self.stop_event.set()
        with self.lock:
            self.state.update(values)
            self.state["status"] = status
            self.state["finished_at_utc"] = pipeline.utc_now()
        self._write()


def normalize_rna(sequence: str) -> str:
    """Normalize RNA spelling for exact sequence-key lookup and validation."""
    return "".join(str(sequence).split()).upper().replace("T", "U")


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(normalize_rna(sequence).encode("ascii")).hexdigest()


def first_a3m_sequence(path: Path) -> str:
    """Read the first A3M record without loading a potentially large file."""
    pieces: list[str] = []
    seen_header = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if seen_header and pieces:
                    break
                seen_header = True
                continue
            if seen_header:
                pieces.append(text)
    # Lowercase A3M insertion characters and alignment gaps are not part of
    # the query spelling.  Query rows normally contain neither, but accepting
    # them makes validation robust to equivalent A3M writers.
    query = "".join(char for char in "".join(pieces) if not char.islower())
    return normalize_rna(query.replace("-", "").replace(".", ""))


def valid_a3m(path: Path | None, sequence: str) -> bool:
    try:
        return bool(
            path
            and path.is_file()
            and path.stat().st_size > 0
            and first_a3m_sequence(path) == normalize_rna(sequence)
        )
    except OSError:
        return False


def build_existing_index(
    targets: list[pipeline.Target],
    simple_json_dir: Path,
    complex_json_dir: Path,
) -> dict[str, MsaSource]:
    """Index verified sequence->MSA pairs from already complete PDB preps."""
    _, updated_index = pipeline.index_json_files(simple_json_dir)
    prep_index, _ = pipeline.index_output_dirs(complex_json_dir)
    result: dict[str, MsaSource] = {}
    for target in targets:
        updated = pipeline.choose_indexed_path(updated_index.get(target.pdb_id, []))
        prep_dir = pipeline.choose_indexed_path(prep_index.get(target.pdb_id, []))
        prep = pipeline.inspect_prep(target.pdb_id, updated, prep_dir)
        if prep.status not in {"COMPLETE", "COMPLETE_REBASABLE"}:
            continue
        for sequence, text in zip(
            prep.updated_json.sequences or [], prep.resolved_msa_paths
        ):
            key = normalize_rna(sequence)
            path = Path(text).expanduser()
            if key not in result and valid_a3m(path, key):
                result[key] = MsaSource(key, path.resolve(), "existing", target.pdb_id)
    return result


def build_cache_index(
    sequences: set[str], cache_dir: Path, database_set_id: str
) -> dict[str, MsaSource]:
    """Load only verified entries from the selected database-specific cache."""
    result: dict[str, MsaSource] = {}
    entries = cache_dir / database_set_id / "entries"
    for sequence in sequences:
        digest = sequence_digest(sequence)
        canonical = entries / digest / "rna_msa.a3m"
        if valid_a3m(canonical, sequence):
            result[sequence] = MsaSource(
                sequence, canonical.resolve(), "cache", digest
            )
    return result


def load_official_mapping(root: Path) -> dict[str, Any]:
    mapping_path = root / "rna_sequence_to_pdb_chains.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"官方 RNA MSA 映射不存在：{mapping_path}；请先下载并解压 rna_msa.tar.gz"
        )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"官方 RNA MSA 映射顶层不是字典：{mapping_path}")
    return {normalize_rna(key): value for key, value in payload.items()}


def official_source(
    root: Path, mapping: dict[str, Any], sequence: str
) -> MsaSource | None:
    key = normalize_rna(sequence)
    values = mapping.get(key, [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return None
    for value in values:
        source_id = str(value).strip()
        if not source_id:
            continue
        candidates = (
            root / "msas" / source_id / f"{source_id}_all.a3m",
            root / "msas" / source_id / "rna_msa.a3m",
        )
        for candidate in candidates:
            if valid_a3m(candidate, key):
                return MsaSource(key, candidate.resolve(), "official", source_id)
    return None


def write_minimal_input(path: Path, sequence: str, name: str) -> None:
    payload = [
        {
            "name": name,
            "sequences": [
                {"rnaSequence": {"sequence": normalize_rna(sequence), "count": 1}}
            ],
        }
    ]
    pipeline.atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


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


def wait_for_memory_below(limit_percent: float) -> None:
    while True:
        percent = cgroup_memory_percent()
        if percent is None or percent < limit_percent:
            return
        print(
            f"[PREP-MEMORY] cgroup={percent:.1f}% >= {limit_percent:.1f}%，"
            "暂停启动下一条序列搜索",
            flush=True,
        )
        time.sleep(30)


def prep_command(raw_json: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        args.protenix_executable,
        "prep",
        "-i",
        str(raw_json),
        "-o",
        str(output_dir),
        "-m",
        "protenix",
        "--nhmmer_n_cpu",
        str(args.nhmmer_cpus),
    ]
    for option, value in (
        ("--seqres_database_path", args.seqres_database),
        ("--ntrna_database_path", args.ntrna_database),
        ("--rfam_database_path", args.rfam_database),
        ("--rna_central_database_path", args.rnacentral_database),
    ):
        if value is not None:
            command.extend([option, str(value)])
    return command


def search_one(sequence: str, args: argparse.Namespace) -> MsaSource | None:
    digest = sequence_digest(sequence)
    entry_dir = args.cache_dir / args.database_set_id / "entries" / digest
    canonical = entry_dir / "rna_msa.a3m"
    if valid_a3m(canonical, sequence):
        return MsaSource(sequence, canonical.resolve(), "cache", digest)

    wait_for_memory_below(args.memory_pause_percent)

    work_dir = args.cache_dir / args.database_set_id / "search_work" / digest
    raw = work_dir / f"seq_{digest[:16]}.json"
    output = work_dir / "prep_output"
    write_minimal_input(raw, sequence, f"seq_{digest[:16]}")
    log = args.report_dir / "logs" / "prep_sequence" / f"{digest}.log"
    code = pipeline.run_logged(
        prep_command(raw, output, args),
        log,
        args.report_dir / "run_events.jsonl",
        {
            "stage": "prep_sequence_search",
            "sequence_sha256": digest,
            "sequence_length": len(sequence),
        },
    )
    found = sorted(
        path
        for path in output.rglob("rna_msa.a3m")
        if valid_a3m(path, sequence)
    )
    if code != 0 or len(found) != 1:
        return None
    entry_dir.mkdir(parents=True, exist_ok=True)
    temporary = canonical.with_name(f".{canonical.name}.tmp")
    shutil.copy2(found[0], temporary)
    os.replace(temporary, canonical)
    return MsaSource(sequence, canonical.resolve(), "searched", digest)


def backup_incomplete_updated(updated: Path, report_dir: Path) -> None:
    if not updated.is_file():
        return
    backup = report_dir / "backups" / "final_updated" / updated.name
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(updated, backup)


def materialize_target(
    target: pipeline.Target,
    raw: Path,
    sources: dict[str, MsaSource],
    args: argparse.Namespace,
) -> tuple[bool, list[dict[str, Any]], str]:
    payload = json.loads(raw.read_text(encoding="utf-8"))
    task_name = str(payload[0].get("name", target.pdb_id.lower())).strip()
    rows: list[dict[str, Any]] = []
    rna_index = 0
    for item in payload[0]["sequences"]:
        if "rnaSequence" not in item:
            continue
        sequence = normalize_rna(item["rnaSequence"].get("sequence", ""))
        source = sources.get(sequence)
        if source is None or not valid_a3m(source.path, sequence):
            return False, rows, f"RNA index={rna_index} 的 MSA 未解析"

        compatibility = (
            args.complex_json_dir
            / f"prep_output_{target.pdb_id.lower()}"
            / task_name
            / "rna_msa"
            / str(rna_index)
            / "rna_msa.a3m"
        )
        selected = source.path
        if not compatibility.exists():
            compatibility.parent.mkdir(parents=True, exist_ok=True)
            try:
                compatibility.symlink_to(source.path.resolve())
            except OSError:
                # Prediction can reference the canonical source directly.  Do
                # not copy a potentially huge A3M merely because symlinks are
                # unavailable on a particular mount.
                pass
        if valid_a3m(compatibility, sequence):
            selected = compatibility.resolve()
        item["rnaSequence"]["unpairedMsaPath"] = str(selected)
        rows.append(
            {
                "PDB_ID": target.pdb_id,
                "RNA_INDEX": rna_index,
                "SEQUENCE_SHA256": sequence_digest(sequence),
                "SEQUENCE_LENGTH": len(sequence),
                "SOURCE_TYPE": source.source_type,
                "SOURCE_ID": source.source_id,
                "SOURCE_PATH": str(source.path),
                "FINAL_MSA_PATH": str(selected),
            }
        )
        rna_index += 1

    updated = raw.with_name(f"{raw.stem}-final-updated.json")
    backup_incomplete_updated(updated, args.report_dir)
    pipeline.atomic_write_text(updated, json.dumps(payload, ensure_ascii=False, indent=4))
    prep_dir = args.complex_json_dir / f"prep_output_{target.pdb_id.lower()}"
    status = pipeline.inspect_prep(target.pdb_id, updated, prep_dir).status
    ok = status in {"COMPLETE", "COMPLETE_REBASABLE"}
    return ok, rows, status


def write_reports(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    source_by_sequence: dict[str, MsaSource],
    needed_sequences: set[str],
    failures: list[str],
) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    detail = args.report_dir / "rna_msa_cache_usage.csv"
    fields = [
        "PDB_ID",
        "RNA_INDEX",
        "SEQUENCE_SHA256",
        "SEQUENCE_LENGTH",
        "SOURCE_TYPE",
        "SOURCE_ID",
        "SOURCE_PATH",
        "FINAL_MSA_PATH",
    ]
    with detail.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for sequence in needed_sequences:
        source = source_by_sequence.get(sequence)
        key = source.source_type if source else "unresolved"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "database_set_id": args.database_set_id,
        "official_msa_enabled": args.allow_official_msa,
        "database_files": database_provenance(args),
        "unique_sequences_needed": len(needed_sequences),
        "source_counts": counts,
        "materialized_rna_entries": len(rows),
        "failures": failures,
        "all_complete": not failures,
    }
    pipeline.atomic_write_text(
        args.report_dir / "rna_msa_cache_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def database_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Record inexpensive database identity metadata for reproducibility."""
    result: dict[str, Any] = {}
    for name in (
        "seqres_database",
        "rfam_database",
        "rnacentral_database",
        "ntrna_database",
    ):
        path = getattr(args, name)
        if path is None:
            result[name] = None
            continue
        try:
            stat = path.stat()
            result[name] = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError as exc:
            result[name] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return result


def run(args: argparse.Namespace) -> None:
    heartbeat = Heartbeat(args.report_dir)
    heartbeat.start()
    try:
        _run(args, heartbeat)
    except BaseException as exc:
        heartbeat.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        heartbeat.finish("COMPLETE", phase="done")


def _run(args: argparse.Namespace, heartbeat: Heartbeat) -> None:
    args.protenix_executable = pipeline.locate_executable(args.protenix)
    targets = pipeline.load_targets(args.manifest)
    raw_index, updated_index = pipeline.index_json_files(args.simple_json_dir)
    prep_index, _ = pipeline.index_output_dirs(args.complex_json_dir)

    needed: list[tuple[pipeline.Target, Path, list[str]]] = []
    for target in targets:
        raw = pipeline.choose_indexed_path(raw_index.get(target.pdb_id, []))
        updated = pipeline.choose_indexed_path(updated_index.get(target.pdb_id, []))
        prep_dir = pipeline.choose_indexed_path(prep_index.get(target.pdb_id, []))
        status = pipeline.inspect_prep(target.pdb_id, updated, prep_dir).status
        if status in {"COMPLETE", "COMPLETE_REBASABLE"}:
            continue
        info = pipeline.read_json_info(raw)
        if raw is None or not info.valid:
            print(f"[PREP-CACHE] {target.pdb_id}: 跳过，原始 JSON 缺失或损坏")
            continue
        needed.append((target, raw, [normalize_rna(s) for s in info.sequences or []]))

    needed_sequences = {sequence for _, _, sequences in needed for sequence in sequences}
    print(f"需要补 prep：{len(needed)}/{len(targets)} PDB")
    print(f"涉及唯一 RNA 序列：{len(needed_sequences)}")
    heartbeat.update(
        status="RUNNING",
        phase="index_existing",
        pdb_needed=len(needed),
        unique_sequences_needed=len(needed_sequences),
        sequences_resolved=0,
        searches_finished=0,
        pdb_materialized=0,
    )

    sources = build_existing_index(targets, args.simple_json_dir, args.complex_json_dir)
    resolved: dict[str, MsaSource] = {
        sequence: sources[sequence]
        for sequence in needed_sequences
        if sequence in sources
    }
    print(f"命中既有有效 MSA：{len(resolved)} 条唯一序列")
    heartbeat.update(sequences_resolved=len(resolved))

    cache_hits = build_cache_index(
        needed_sequences - set(resolved), args.cache_dir, args.database_set_id
    )
    resolved.update(cache_hits)
    print(f"命中当前数据库集合的序列缓存：{len(cache_hits)} 条唯一序列")
    heartbeat.update(sequences_resolved=len(resolved))

    official_hits = 0
    if args.allow_official_msa:
        heartbeat.update(phase="index_official")
        official_mapping = load_official_mapping(args.official_rna_msa_root)
        for sequence in sorted(needed_sequences - set(resolved)):
            source = official_source(args.official_rna_msa_root, official_mapping, sequence)
            if source is not None:
                resolved[sequence] = source
                official_hits += 1
    print(f"命中官方预计算 MSA：{official_hits} 条唯一序列")
    heartbeat.update(sequences_resolved=len(resolved))

    unresolved = sorted(needed_sequences - set(resolved), key=lambda s: (len(s), s))
    print(f"仍需新搜索：{len(unresolved)} 条唯一序列")
    lock = threading.Lock()

    usage_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    materialized_count = 0
    materialized_success = 0
    pending: dict[str, tuple[pipeline.Target, Path, list[str]]] = {
        target.pdb_id: (target, raw, sequences)
        for target, raw, sequences in needed
    }

    def materialize_ready() -> None:
        nonlocal materialized_count, materialized_success
        ready_ids = [
            pdb_id
            for pdb_id, (_, _, sequences) in pending.items()
            if all(sequence in resolved for sequence in sequences)
        ]
        for pdb_id in ready_ids:
            target, raw, sequences = pending.pop(pdb_id)
            try:
                ok, rows, status = materialize_target(target, raw, resolved, args)
            except Exception as exc:
                ok, rows, status = False, [], f"{type(exc).__name__}: {exc}"
            usage_rows.extend(rows)
            materialized_count += 1
            print(f"[PREP-CACHE] {target.pdb_id}: {'OK' if ok else 'FAILED'} ({status})")
            if not ok:
                failures.append(f"{target.pdb_id}: {status}")
            else:
                materialized_success += 1
            heartbeat.update(
                phase="materialize_incremental",
                pdb_materialized=materialized_count,
                pdb_remaining=len(pending),
                pdb_failures=len(failures),
                current_pdb=target.pdb_id,
            )

    # Commit every PDB already unlocked by an existing or cached MSA before
    # launching any new expensive search.
    materialize_ready()
    if unresolved:
        heartbeat.update(phase="search_unique_sequences", searches_total=len(unresolved))
        searches_finished = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(search_one, sequence, args): sequence for sequence in unresolved}
            for future in as_completed(futures):
                sequence = futures[future]
                try:
                    source = future.result()
                except Exception as exc:
                    source = None
                    message = f"{type(exc).__name__}: {exc}"
                else:
                    message = source.source_type if source else "FAILED"
                with lock:
                    searches_finished += 1
                    if source is not None:
                        resolved[sequence] = source
                    print(
                        f"[SEQUENCE len={len(sequence)} sha={sequence_digest(sequence)[:12]}] "
                        f"{message}"
                    )
                    heartbeat.update(
                        phase="search_unique_sequences",
                        searches_finished=searches_finished,
                        sequences_resolved=len(resolved),
                    )
                    materialize_ready()

    # Record unresolved PDBs only after every scheduled search returned.
    for target, _, sequences in pending.values():
        missing = sum(sequence not in resolved for sequence in sequences)
        failures.append(f"{target.pdb_id}: 仍缺少 {missing} 条 RNA MSA")

    heartbeat.update(
        phase="write_reports",
        pdb_materialized=materialized_count,
        pdb_remaining=len(pending),
        pdb_failures=len(failures),
    )

    write_reports(args, usage_rows, resolved, needed_sequences, failures)
    print(f"本次成功物化 PDB：{materialized_success}/{len(needed)}")
    print(f"来源报告：{args.report_dir / 'rna_msa_cache_summary.json'}")
    if failures:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(description="按唯一 RNA 序列缓存的 Protenix prep")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--simple-json-dir", type=Path, required=True)
    parser.add_argument("--complex-json-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--official-rna-msa-root", type=Path)
    parser.add_argument(
        "--allow-official-msa",
        action="store_true",
        help="显式允许使用官方预计算 MSA；默认关闭以保持自定义数据库口径",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=home / "protenix_data/rna_msa_sequence_cache"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nhmmer-cpus", type=int, default=8)
    parser.add_argument(
        "--memory-pause-percent",
        type=float,
        default=75.0,
        help="每条新搜索前检查 cgroup；达到该百分比则等待内存回落",
    )
    parser.add_argument(
        "--database-set-id",
        default="legacy_rna_databases_20260807",
        help="数据库集合稳定标识；用于隔离不同数据库生成的序列缓存",
    )
    parser.add_argument("--protenix", default="protenix")
    parser.add_argument("--seqres-database", type=Path)
    parser.add_argument("--ntrna-database", type=Path)
    parser.add_argument("--rfam-database", type=Path)
    parser.add_argument("--rnacentral-database", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.workers < 1 or args.nhmmer_cpus < 1:
        raise ValueError("--workers 和 --nhmmer-cpus 必须大于 0")
    if not 1 <= args.memory_pause_percent <= 100:
        raise ValueError("--memory-pause-percent 必须在 1 到 100 之间")
    if args.allow_official_msa and args.official_rna_msa_root is None:
        raise ValueError("使用 --allow-official-msa 时必须提供 --official-rna-msa-root")
    run(args)


if __name__ == "__main__":
    main()
