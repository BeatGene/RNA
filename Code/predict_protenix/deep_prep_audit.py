#!/usr/bin/env python3
"""对 Protenix RNA prep 结果进行序列级深度审计。

本程序只读取已有文件，不修改原始 JSON、final JSON、A3M 或 prep 目录。

默认检查当前 2241 个目标中属于旧集合的 1974 个 PDB，并验证：

1. 原始 JSON 与 *-final-updated.json 的 RNA sequence/count 完全一致；
2. JSON 展开的 RNA 链序列多重集合与第一阶段链清单一致；
3. final JSON 的每个 RNA 条目都能唯一定位到 prep 中的 rna_msa.a3m；
4. A3M 可解析，且第一条 query 序列与对应 JSON RNA 序列完全一致；
5. 旧绝对路径可以依据标准 RNA 索引安全重定位。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PDB_ID_PATTERN = re.compile(r"^[0-9A-Z]{4}$")
PREP_FOLDER_RE = re.compile(r"^prep_output_(.+)$", re.IGNORECASE)
PRED_FOLDER_RE = re.compile(r"^pred_output_.+_seed_\d+$", re.IGNORECASE)

DEFAULT_PIPELINE_HOME = Path(
    os.environ.get("RNA_PIPELINE_HOME", "/storage9920/home/tinghao.xia")
)


@dataclass(frozen=True)
class Target:
    pdb_id: str
    current_target: bool
    legacy_1979: bool


@dataclass
class RnaEntry:
    sequence: str
    count: int
    msa_path: str


@dataclass
class JsonAudit:
    path: str = ""
    valid: bool = False
    error: str = ""
    task_count: int = 0
    task_name: str = ""
    entries: list[RnaEntry] = field(default_factory=list)
    payload: Any = None


@dataclass
class A3mAudit:
    path: str = ""
    valid: bool = False
    error: str = ""
    file_size: int = 0
    sha256: str = ""
    depth: int = 0
    query_header: str = ""
    query_raw: str = ""
    query_normalized: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_pdb_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not PDB_ID_PATTERN.fullmatch(text):
        raise ValueError(f"非法 PDB ID：{value!r}")
    return text


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "ok"}


def normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def load_targets(path: Path, scope: str) -> list[Target]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到第一阶段清单：{path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"PDB_ID", "CURRENT_TARGET", "LEGACY_1979"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"第一阶段清单缺少列：{sorted(missing)}")

    targets: list[Target] = []
    seen: set[str] = set()
    for row in frame.to_dict("records"):
        pdb_id = normalize_pdb_id(row["PDB_ID"])
        current = is_true(row["CURRENT_TARGET"])
        legacy = is_true(row["LEGACY_1979"])
        selected = {
            "legacy-current": current and legacy,
            "all-current": current,
            "legacy-all": legacy,
        }[scope]
        if selected and pdb_id not in seen:
            targets.append(Target(pdb_id, current, legacy))
            seen.add(pdb_id)
    return sorted(targets, key=lambda item: item.pdb_id)


def load_chain_sequences(path: Path) -> dict[str, Counter[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 RNA 链清单：{path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"PDB_ID", "CHAIN_ID", "SEQUENCE_CANONICAL"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RNA 链清单缺少列：{sorted(missing)}")

    result: dict[str, Counter[str]] = {}
    for row in frame.to_dict("records"):
        pdb_id = normalize_pdb_id(row["PDB_ID"])
        sequence = normalize_sequence(row["SEQUENCE_CANONICAL"])
        if sequence:
            result.setdefault(pdb_id, Counter())[sequence] += 1
    return result


def index_json_files(
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    raw: dict[str, list[Path]] = {}
    updated: dict[str, list[Path]] = {}
    if not root.is_dir():
        return raw, updated

    for path in root.rglob("*.json"):
        stem = path.stem
        lower = stem.lower()
        try:
            if lower.endswith("-final-updated"):
                pdb_id = normalize_pdb_id(stem[: -len("-final-updated")])
                updated.setdefault(pdb_id, []).append(path.resolve())
            elif "-update-msa" not in lower and "-runtime" not in lower:
                pdb_id = normalize_pdb_id(stem)
                raw.setdefault(pdb_id, []).append(path.resolve())
        except ValueError:
            continue
    return raw, updated


def index_prep_dirs(root: Path) -> dict[str, list[Path]]:
    prep: dict[str, list[Path]] = {}
    if not root.is_dir():
        return prep

    for current, dirnames, _ in os.walk(root):
        kept: list[str] = []
        for dirname in dirnames:
            path = (Path(current) / dirname).resolve()
            prep_match = PREP_FOLDER_RE.fullmatch(dirname)
            if prep_match:
                try:
                    pdb_id = normalize_pdb_id(prep_match.group(1))
                except ValueError:
                    continue
                prep.setdefault(pdb_id, []).append(path)
            elif not PRED_FOLDER_RE.fullmatch(dirname):
                kept.append(dirname)
        dirnames[:] = kept
    return prep


def choose_path(paths: Iterable[Path]) -> tuple[Path | None, int]:
    choices = sorted(
        set(paths),
        key=lambda path: (len(path.parts), len(str(path)), str(path)),
    )
    return (choices[0] if choices else None), len(choices)


def read_json(path: Path | None) -> JsonAudit:
    if path is None:
        return JsonAudit(error="文件不存在")

    result = JsonAudit(path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("顶层必须是非空 JSON 列表")
        if len(payload) != 1:
            raise ValueError(f"预期 1 个任务，实际 {len(payload)} 个")
        task = payload[0]
        if not isinstance(task, dict):
            raise ValueError("任务必须是 JSON 对象")
        if not isinstance(task.get("sequences"), list):
            raise ValueError("缺少 sequences 列表")

        entries: list[RnaEntry] = []
        for item in task["sequences"]:
            if not isinstance(item, dict) or "rnaSequence" not in item:
                continue
            rna = item["rnaSequence"]
            if not isinstance(rna, dict):
                raise ValueError("rnaSequence 必须是 JSON 对象")
            sequence = normalize_sequence(rna.get("sequence", ""))
            if not sequence:
                raise ValueError("RNA sequence 为空")
            count = int(rna.get("count", 1))
            if count < 1:
                raise ValueError("RNA count 必须大于 0")
            entries.append(
                RnaEntry(
                    sequence=sequence,
                    count=count,
                    msa_path=str(rna.get("unpairedMsaPath", "")).strip(),
                )
            )
        if not entries:
            raise ValueError("JSON 中没有 RNA 条目")

        result.valid = True
        result.task_count = len(payload)
        result.task_name = str(task.get("name", "")).strip()
        result.entries = entries
        result.payload = payload
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def json_without_runtime_paths(payload: Any) -> Any:
    cleaned = copy.deepcopy(payload)
    if not isinstance(cleaned, list):
        return cleaned
    for task in cleaned:
        if not isinstance(task, dict):
            continue
        for item in task.get("sequences", []):
            if not isinstance(item, dict):
                continue
            for sequence_type in (
                "rnaSequence",
                "proteinSequence",
                "dnaSequence",
            ):
                sequence = item.get(sequence_type)
                if isinstance(sequence, dict):
                    sequence.pop("unpairedMsaPath", None)
                    sequence.pop("pairedMsaPath", None)
                    sequence.pop("templatesPath", None)
    return cleaned


def expanded_counter(entries: list[RnaEntry]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        counter[entry.sequence] += entry.count
    return counter


def a3m_files_by_index(
    prep_dir: Path | None,
) -> tuple[dict[int, list[Path]], list[Path]]:
    by_index: dict[int, list[Path]] = {}
    all_files: list[Path] = []
    if prep_dir is None or not prep_dir.is_dir():
        return by_index, all_files

    for path in sorted(prep_dir.rglob("rna_msa.a3m"), key=str):
        if not path.is_file():
            continue
        resolved = path.resolve()
        all_files.append(resolved)
        try:
            index = int(path.parent.name)
        except ValueError:
            continue
        by_index.setdefault(index, []).append(resolved)
    return by_index, all_files


def parse_a3m(path: Path | None) -> A3mAudit:
    if path is None:
        return A3mAudit(error="A3M 文件不存在")

    result = A3mAudit(path=str(path))
    try:
        stat = path.stat()
        result.file_size = stat.st_size
        if stat.st_size == 0:
            raise ValueError("A3M 文件为空")

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result.sha256 = digest.hexdigest()

        headers: list[str] = []
        sequences: list[str] = []
        current: list[str] | None = None
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                if text.startswith(">"):
                    headers.append(text[1:].strip())
                    current = []
                    sequences.append("")
                else:
                    if current is None:
                        raise ValueError(
                            f"第 {line_number} 行在第一个 FASTA header 之前出现序列"
                        )
                    current.append("".join(text.split()))
                    sequences[-1] = "".join(current)

        if not sequences:
            raise ValueError("A3M 中没有 FASTA 记录")
        if any(not sequence for sequence in sequences):
            raise ValueError("A3M 中存在空序列记录")

        query = sequences[0]
        # A3M 小写字符表示相对 query 的插入；'-' 和 '.' 表示 gap。
        query_without_insertions = "".join(
            char for char in query if not char.islower()
        )
        normalized = (
            query_without_insertions.replace("-", "").replace(".", "").upper()
        )
        result.valid = True
        result.depth = len(sequences)
        result.query_header = headers[0] if headers else ""
        result.query_raw = query
        result.query_normalized = normalized
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def path_index(path_text: str) -> int | None:
    if not path_text:
        return None
    try:
        return int(Path(path_text).parent.name)
    except ValueError:
        return None


def path_is_nonempty_file(path_text: str) -> bool:
    if not path_text:
        return False
    try:
        path = Path(path_text).expanduser()
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def resolve_a3m(
    index: int,
    final_path: str,
    by_index: dict[int, list[Path]],
) -> tuple[Path | None, str, str]:
    if path_is_nonempty_file(final_path):
        return Path(final_path).expanduser().resolve(), "DIRECT", ""

    candidates = by_index.get(index, [])
    if len(candidates) == 1:
        return candidates[0], "REBASED_INDEX", ""
    if not candidates:
        return None, "MISSING", f"RNA索引 {index} 没有 rna_msa.a3m"
    return (
        None,
        "AMBIGUOUS",
        f"RNA索引 {index} 找到 {len(candidates)} 个 rna_msa.a3m",
    )


def audit_target(
    target: Target,
    raw_index: dict[str, list[Path]],
    final_index: dict[str, list[Path]],
    prep_index: dict[str, list[Path]],
    manifest_sequences: dict[str, Counter[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path, raw_duplicates = choose_path(raw_index.get(target.pdb_id, []))
    final_path, final_duplicates = choose_path(final_index.get(target.pdb_id, []))
    prep_dir, prep_duplicates = choose_path(prep_index.get(target.pdb_id, []))
    raw = read_json(raw_path)
    final = read_json(final_path)
    by_index, all_a3m_files = a3m_files_by_index(prep_dir)

    reasons: list[str] = []
    if raw_duplicates != 1:
        reasons.append(f"原始JSON数量={raw_duplicates}")
    if final_duplicates != 1:
        reasons.append(f"final JSON数量={final_duplicates}")
    if prep_duplicates != 1:
        reasons.append(f"prep目录数量={prep_duplicates}")
    if not raw.valid:
        reasons.append(f"原始JSON无效：{raw.error}")
    if not final.valid:
        reasons.append(f"final JSON无效：{final.error}")
    if prep_dir is None or not prep_dir.is_dir():
        reasons.append("prep目录不存在")

    raw_final_entries_match = False
    json_equal_ignoring_paths = False
    raw_manifest_match = False
    task_name_match = False
    expected_entries = 0

    if raw.valid:
        expected_entries = len(raw.entries)
        raw_manifest_match = (
            expanded_counter(raw.entries)
            == manifest_sequences.get(target.pdb_id, Counter())
        )
        task_name_match = raw.task_name.strip().upper() == target.pdb_id
        if not raw_manifest_match:
            reasons.append("原始JSON展开序列与RNA链清单不一致")
        if not task_name_match:
            reasons.append(
                f"原始JSON任务名={raw.task_name!r}，预期={target.pdb_id!r}"
            )

    if raw.valid and final.valid:
        raw_signature = [
            (entry.sequence, entry.count) for entry in raw.entries
        ]
        final_signature = [
            (entry.sequence, entry.count) for entry in final.entries
        ]
        raw_final_entries_match = raw_signature == final_signature
        json_equal_ignoring_paths = (
            json_without_runtime_paths(raw.payload)
            == json_without_runtime_paths(final.payload)
        )
        if not raw_final_entries_match:
            reasons.append("原始JSON与final JSON的RNA sequence/count不一致")
        if not json_equal_ignoring_paths:
            reasons.append("原始JSON与final JSON除运行路径外仍存在差异")

    rna_rows: list[dict[str, Any]] = []
    query_match_count = 0
    resolved_count = 0
    parsed_count = 0
    final_entries = final.entries if final.valid else []

    for index in range(max(len(raw.entries), len(final_entries))):
        raw_entry = raw.entries[index] if index < len(raw.entries) else None
        final_entry = (
            final_entries[index] if index < len(final_entries) else None
        )
        expected_sequence = (
            raw_entry.sequence
            if raw_entry is not None
            else (final_entry.sequence if final_entry is not None else "")
        )
        final_msa_path = final_entry.msa_path if final_entry else ""
        final_path_index = path_index(final_msa_path)
        final_index_match = final_path_index == index
        resolved, resolution, resolution_error = resolve_a3m(
            index, final_msa_path, by_index
        )
        if resolved is not None:
            resolved_count += 1
        a3m = parse_a3m(resolved)
        if a3m.valid:
            parsed_count += 1
        query_match = a3m.valid and a3m.query_normalized == expected_sequence
        query_tu_match = (
            a3m.valid
            and a3m.query_normalized.replace("T", "U")
            == expected_sequence.replace("T", "U")
        )
        if query_match:
            query_match_count += 1

        entry_reasons: list[str] = []
        if raw_entry is None or final_entry is None:
            entry_reasons.append("原始JSON/final JSON缺少对应RNA条目")
        elif (
            raw_entry.sequence != final_entry.sequence
            or raw_entry.count != final_entry.count
        ):
            entry_reasons.append("原始JSON与final JSON条目不一致")
        if not final_msa_path:
            entry_reasons.append("final JSON缺少unpairedMsaPath")
        elif not final_index_match:
            entry_reasons.append(
                f"final JSON路径索引={final_path_index}，预期={index}"
            )
        if resolution_error:
            entry_reasons.append(resolution_error)
        if not a3m.valid:
            entry_reasons.append(f"A3M无效：{a3m.error}")
        elif not query_match:
            entry_reasons.append(
                "A3M第一条query与JSON序列不一致"
                + ("（仅T/U归一化后相同）" if query_tu_match else "")
            )

        entry_status = "PASS" if not entry_reasons else "FAIL"
        rna_rows.append(
            {
                "PDB_ID": target.pdb_id,
                "RNA_INDEX": index,
                "STATUS": entry_status,
                "RAW_SEQUENCE": raw_entry.sequence if raw_entry else "",
                "RAW_COUNT": raw_entry.count if raw_entry else "",
                "FINAL_SEQUENCE": final_entry.sequence if final_entry else "",
                "FINAL_COUNT": final_entry.count if final_entry else "",
                "JSON_ENTRY_MATCH": bool(
                    raw_entry
                    and final_entry
                    and raw_entry.sequence == final_entry.sequence
                    and raw_entry.count == final_entry.count
                ),
                "FINAL_MSA_PATH": final_msa_path,
                "FINAL_PATH_DIRECT_VALID": path_is_nonempty_file(
                    final_msa_path
                ),
                "FINAL_PATH_INDEX": (
                    final_path_index if final_path_index is not None else ""
                ),
                "FINAL_PATH_INDEX_MATCH": final_index_match,
                "MSA_RESOLUTION": resolution,
                "RESOLVED_MSA_PATH": str(resolved or ""),
                "MSA_FILE_SIZE": a3m.file_size,
                "MSA_SHA256": a3m.sha256,
                "A3M_VALID": a3m.valid,
                "A3M_DEPTH": a3m.depth,
                "A3M_QUERY_HEADER": a3m.query_header,
                "A3M_QUERY_NORMALIZED": a3m.query_normalized,
                "A3M_QUERY_MATCH": query_match,
                "A3M_QUERY_TU_NORMALIZED_MATCH": query_tu_match,
                "MESSAGE": "; ".join(entry_reasons) or "OK",
            }
        )

    expected_index_set = set(range(len(final_entries)))
    actual_index_set = set(by_index)
    msa_index_set_match = actual_index_set == expected_index_set
    duplicate_msa_indices = sorted(
        index for index, paths in by_index.items() if len(paths) != 1
    )
    if duplicate_msa_indices:
        reasons.append(f"重复MSA索引={duplicate_msa_indices}")
    if not msa_index_set_match:
        reasons.append(
            f"MSA索引集合={sorted(actual_index_set)}，"
            f"预期={sorted(expected_index_set)}"
        )
    if len(all_a3m_files) != len(final_entries):
        reasons.append(
            f"A3M文件数={len(all_a3m_files)}，"
            f"final RNA条目数={len(final_entries)}"
        )
    if any(row["STATUS"] != "PASS" for row in rna_rows):
        reasons.append("至少一个RNA条目深度校验失败")

    status = "PASS" if not reasons else "FAIL"
    pdb_row = {
        "PDB_ID": target.pdb_id,
        "STATUS": status,
        "CURRENT_TARGET": target.current_target,
        "LEGACY_1979": target.legacy_1979,
        "RAW_JSON_PATH": str(raw_path or ""),
        "RAW_JSON_COUNT": raw_duplicates,
        "RAW_JSON_VALID": raw.valid,
        "FINAL_JSON_PATH": str(final_path or ""),
        "FINAL_JSON_COUNT": final_duplicates,
        "FINAL_JSON_VALID": final.valid,
        "PREP_DIR": str(prep_dir or ""),
        "PREP_DIR_COUNT": prep_duplicates,
        "TASK_NAME_MATCH": task_name_match,
        "RAW_FINAL_RNA_ENTRIES_MATCH": raw_final_entries_match,
        "JSON_EQUAL_IGNORING_RUNTIME_PATHS": json_equal_ignoring_paths,
        "RAW_JSON_MANIFEST_SEQUENCE_MATCH": raw_manifest_match,
        "RNA_ENTRY_COUNT": expected_entries,
        "A3M_FILE_COUNT": len(all_a3m_files),
        "MSA_INDEX_SET_MATCH": msa_index_set_match,
        "MSA_RESOLVED_COUNT": resolved_count,
        "A3M_PARSED_COUNT": parsed_count,
        "A3M_QUERY_MATCH_COUNT": query_match_count,
        "MESSAGE": "; ".join(reasons) or "OK",
    }
    return pdb_row, rna_rows


def write_reports(
    report_dir: Path,
    pdb_rows: list[dict[str, Any]],
    rna_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pdb_frame = pd.DataFrame(pdb_rows)
    rna_frame = pd.DataFrame(rna_rows)
    summary_frame = pd.DataFrame(
        [{"METRIC": key, "VALUE": value} for key, value in summary.items()]
    )

    pdb_frame.to_csv(
        report_dir / "deep_prep_pdb_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rna_frame.to_csv(
        report_dir / "deep_prep_rna_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (report_dir / "deep_prep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    workbook = report_dir / "deep_prep_audit.xlsx"
    temporary = workbook.with_name(f".{workbook.stem}.tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="摘要", index=False)
        pdb_frame.to_excel(writer, sheet_name="逐PDB", index=False)
        rna_frame.to_excel(writer, sheet_name="逐RNA条目", index=False)
    temporary.replace(workbook)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protenix RNA prep 序列级深度审计（只读）"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PIPELINE_HOME
        / "Code/pipeline_reports/PDB_RAW/pdb_cif_manifest.csv",
    )
    parser.add_argument(
        "--chain-manifest",
        type=Path,
        default=DEFAULT_PIPELINE_HOME
        / "Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv",
    )
    parser.add_argument(
        "--simple-json-dir",
        type=Path,
        default=DEFAULT_PIPELINE_HOME / "Json_data/Simple_json",
    )
    parser.add_argument(
        "--complex-json-dir",
        type=Path,
        default=DEFAULT_PIPELINE_HOME / "Json_data/Complex_json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_PIPELINE_HOME
        / "Code/pipeline_reports/DECOYS/deep_prep_audit",
    )
    parser.add_argument(
        "--scope",
        choices=("legacy-current", "all-current", "legacy-all"),
        default="legacy-current",
        help=(
            "legacy-current=当前目标中的旧1974个（默认）；"
            "all-current=全部当前目标；legacy-all=旧1979个"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in (
        "manifest",
        "chain_manifest",
        "simple_json_dir",
        "complex_json_dir",
        "report_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.workers < 1:
        raise ValueError("--workers 必须大于 0")

    started = utc_now()
    targets = load_targets(args.manifest, args.scope)
    chain_sequences = load_chain_sequences(args.chain_manifest)
    raw_index, final_index = index_json_files(args.simple_json_dir)
    prep_index = index_prep_dirs(args.complex_json_dir)
    print(f"深度审计目标：{len(targets)} 个 PDB；scope={args.scope}")

    pdb_rows: list[dict[str, Any]] = []
    rna_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                audit_target,
                target,
                raw_index,
                final_index,
                prep_index,
                chain_sequences,
            ): target.pdb_id
            for target in targets
        }
        completed = 0
        for future in as_completed(futures):
            pdb_row, rows = future.result()
            pdb_rows.append(pdb_row)
            rna_rows.extend(rows)
            completed += 1
            if completed % 100 == 0 or completed == len(targets):
                failed = sum(row["STATUS"] != "PASS" for row in pdb_rows)
                print(
                    f"进度 {completed}/{len(targets)}；"
                    f"当前失败 PDB={failed}",
                    flush=True,
                )

    pdb_rows.sort(key=lambda row: row["PDB_ID"])
    rna_rows.sort(key=lambda row: (row["PDB_ID"], int(row["RNA_INDEX"])))
    finished = utc_now()
    summary = {
        "started_at_utc": started,
        "finished_at_utc": finished,
        "scope": args.scope,
        "target_count": len(pdb_rows),
        "pass_pdb_count": sum(row["STATUS"] == "PASS" for row in pdb_rows),
        "fail_pdb_count": sum(row["STATUS"] != "PASS" for row in pdb_rows),
        "raw_json_valid_count": sum(
            bool(row["RAW_JSON_VALID"]) for row in pdb_rows
        ),
        "final_json_valid_count": sum(
            bool(row["FINAL_JSON_VALID"]) for row in pdb_rows
        ),
        "raw_final_rna_entries_match_count": sum(
            bool(row["RAW_FINAL_RNA_ENTRIES_MATCH"]) for row in pdb_rows
        ),
        "json_equal_ignoring_runtime_paths_count": sum(
            bool(row["JSON_EQUAL_IGNORING_RUNTIME_PATHS"])
            for row in pdb_rows
        ),
        "raw_json_manifest_sequence_match_count": sum(
            bool(row["RAW_JSON_MANIFEST_SEQUENCE_MATCH"])
            for row in pdb_rows
        ),
        "rna_entry_count": len(rna_rows),
        "pass_rna_entry_count": sum(
            row["STATUS"] == "PASS" for row in rna_rows
        ),
        "fail_rna_entry_count": sum(
            row["STATUS"] != "PASS" for row in rna_rows
        ),
        "direct_msa_path_count": sum(
            row["MSA_RESOLUTION"] == "DIRECT" for row in rna_rows
        ),
        "rebased_msa_path_count": sum(
            row["MSA_RESOLUTION"] == "REBASED_INDEX" for row in rna_rows
        ),
        "a3m_valid_count": sum(bool(row["A3M_VALID"]) for row in rna_rows),
        "a3m_query_match_count": sum(
            bool(row["A3M_QUERY_MATCH"]) for row in rna_rows
        ),
        "all_pass": bool(pdb_rows)
        and all(row["STATUS"] == "PASS" for row in pdb_rows),
    }
    write_reports(args.report_dir, pdb_rows, rna_rows, summary)

    print("\n深度审计完成")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\n报告目录：{args.report_dir}")
    print(f"易读报告：{args.report_dir / 'deep_prep_audit.xlsx'}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        raise SystemExit(130)
