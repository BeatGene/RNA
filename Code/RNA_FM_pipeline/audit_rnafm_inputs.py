#!/usr/bin/env python3
"""Audit RNA-FM inputs before generating residue embeddings.

The audit is deliberately read-only with respect to JSON, mmCIF metadata, and
dataset split directories.  It writes a timestamped report directory containing
machine-readable JSON plus human-readable TSV/log files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PIPELINE_VERSION = "1.1-rnafm-input-audit"
DEFAULT_HOME = Path("/storage9920/home/tinghao.xia")
DEFAULT_JSON_DIR = DEFAULT_HOME / "Json_data/Simple_json"
DEFAULT_CHAIN_MANIFEST = (
    DEFAULT_HOME / "Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv"
)
DEFAULT_SPLIT_ROOT = DEFAULT_HOME / "Data"
DEFAULT_REPORT_ROOT = DEFAULT_HOME / "Code/pipeline_reports"
DEFAULT_EXCLUDED = ("3OK2", "3OK4", "5EME", "176D", "5EMF")
VALID_BASES = frozenset("ACGU")
RAW_JSON_RE = re.compile(r"^[A-Za-z0-9]{4}\.json$", re.IGNORECASE)
FINAL_JSON_RE = re.compile(
    r"^[A-Za-z0-9]{4}-final-updated\.json$", re.IGNORECASE
)
PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")


PDB_COLUMNS = [
    "PDB_ID",
    "RAW_JSON",
    "TASK_NAME",
    "SPLIT",
    "SPLIT_STATUS",
    "JSON_RNA_CHAIN_COUNT",
    "MANIFEST_RNA_CHAIN_COUNT",
    "MAX_CHAIN_LENGTH",
    "SEQUENCE_MULTISET_MATCH",
    "HAS_NON_ACGU",
    "HAS_OVERLENGTH_CHAIN",
    "STATUS",
    "ISSUE_CODES",
]

CHAIN_COLUMNS = [
    "PDB_ID",
    "SPLIT",
    "SPLIT_STATUS",
    "JSON_ENTRY_INDEX",
    "COPY_INDEX",
    "EXPECTED_PROTENIX_CHAIN_ID",
    "ORIGINAL_CHAIN_ID",
    "ENTITY_ID",
    "SEQUENCE_LENGTH",
    "SEQUENCE_SHA256",
    "SEQUENCE",
    "UNKNOWN_SYMBOLS",
    "OVER_MAX_LENGTH",
    "MAPPING_STATUS",
]

ISSUE_COLUMNS = ["SEVERITY", "CODE", "PDB_ID", "DETAIL"]


@dataclass(frozen=True)
class ManifestChain:
    pdb_id: str
    chain_id: str
    entity_id: str
    sequence: str


@dataclass(frozen=True)
class JsonChain:
    pdb_id: str
    json_entry_index: int
    copy_index: int
    sequence: str


@dataclass(frozen=True)
class SplitManifestRecord:
    pdb_id: str
    initial_split: str
    final_split: str
    final_status: str


@dataclass
class AuditResult:
    report_dir: Path
    summary: dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_text(value: datetime | None = None) -> str:
    value = value or utc_now()
    return value.strftime("%Y%m%dT%H%M%SZ")


def normalize_pdb_id(value: Any) -> str:
    pdb_id = str(value).strip().upper()
    if not PDB_ID_RE.fullmatch(pdb_id):
        raise ValueError(f"非法 PDB ID：{value!r}")
    return pdb_id


def normalize_sequence(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii", errors="strict")).hexdigest()


def int_to_letters(value: int) -> str:
    """Return 1 -> A, 26 -> Z, 27 -> AA."""
    if value < 1:
        raise ValueError("value 必须 >= 1")
    letters: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def create_report_dir(root: Path, requested_name: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_name = requested_name or f"RNA_FM_AUDIT_{timestamp_text()}"
    candidate = root / base_name
    counter = 2
    while candidate.exists():
        candidate = root / f"{base_name}_{counter}"
        counter += 1
    candidate.mkdir(parents=False)
    return candidate


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def write_tsv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def index_json_files(
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"JSON 目录不存在：{root}")

    raw: dict[str, list[Path]] = defaultdict(list)
    final: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*.json"):
        if RAW_JSON_RE.fullmatch(path.name):
            raw[normalize_pdb_id(path.stem)].append(path.resolve())
        elif FINAL_JSON_RE.fullmatch(path.name):
            pdb_id = path.name[:4]
            final[normalize_pdb_id(pdb_id)].append(path.resolve())
    return dict(raw), dict(final)


def load_manifest(path: Path) -> dict[str, list[ManifestChain]]:
    if not path.is_file():
        raise FileNotFoundError(f"RNA 链清单不存在：{path}")

    result: dict[str, list[ManifestChain]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"PDB_ID", "CHAIN_ID", "SEQUENCE_CANONICAL"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"RNA 链清单缺少列 {sorted(missing)}；实际列为 {reader.fieldnames}"
            )
        for row in reader:
            pdb_id = normalize_pdb_id(row["PDB_ID"])
            sequence = normalize_sequence(row["SEQUENCE_CANONICAL"])
            if not sequence:
                raise ValueError(f"{pdb_id} 的 SEQUENCE_CANONICAL 为空")
            result[pdb_id].append(
                ManifestChain(
                    pdb_id=pdb_id,
                    chain_id=str(row["CHAIN_ID"]).strip(),
                    entity_id=str(row.get("ENTITY_ID", "")).strip(),
                    sequence=sequence,
                )
            )
    return dict(result)


def index_splits(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"split 根目录不存在：{root}")
    result: dict[str, list[str]] = defaultdict(list)
    invalid_names: list[str] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split 目录不存在：{split_dir}")
        for path in split_dir.iterdir():
            if not path.is_dir():
                continue
            if not PDB_ID_RE.fullmatch(path.name):
                invalid_names.append(str(path.resolve()))
                continue
            result[normalize_pdb_id(path.name)].append(split)
    return dict(result), invalid_names


def discover_split_manifest(report_root: Path) -> Path | None:
    candidates = sorted(
        report_root.glob(
            "DATA_SPLIT_2241_CHAINMASK_*_EXECUTE/final_manifest.tsv"
        ),
        key=lambda path: path.parent.name,
    )
    return candidates[-1].resolve() if candidates else None


def load_split_manifest(path: Path) -> dict[str, SplitManifestRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest 不存在：{path}")

    result: dict[str, SplitManifestRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"PDB_ID", "INITIAL_SPLIT", "FINAL_SPLIT", "FINAL_STATUS"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"split manifest 缺少列 {sorted(missing)}；"
                f"实际列为 {reader.fieldnames}"
            )
        for row in reader:
            pdb_id = normalize_pdb_id(row["PDB_ID"])
            if pdb_id in result:
                raise ValueError(f"split manifest 中 PDB ID 重复：{pdb_id}")
            result[pdb_id] = SplitManifestRecord(
                pdb_id=pdb_id,
                initial_split=str(row["INITIAL_SPLIT"]).strip(),
                final_split=str(row["FINAL_SPLIT"]).strip(),
                final_status=str(row["FINAL_STATUS"]).strip(),
            )
    return result


def parse_raw_json(path: Path, expected_pdb_id: str) -> tuple[str, list[JsonChain]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("顶层必须是长度为 1 的 JSON 列表")
    task = payload[0]
    if not isinstance(task, dict):
        raise ValueError("JSON task 必须是对象")
    sequences = task.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("JSON 缺少 sequences 列表")

    task_name = str(task.get("name", "")).strip().upper()
    expanded: list[JsonChain] = []
    for entry_index, item in enumerate(sequences):
        if not isinstance(item, dict) or "rnaSequence" not in item:
            continue
        rna = item["rnaSequence"]
        if not isinstance(rna, dict):
            raise ValueError(f"rnaSequence[{entry_index}] 必须是对象")
        sequence = normalize_sequence(rna.get("sequence", ""))
        if not sequence:
            raise ValueError(f"rnaSequence[{entry_index}].sequence 为空")
        count = int(rna.get("count", 1))
        if count < 1:
            raise ValueError(f"rnaSequence[{entry_index}].count={count}")
        for copy_index in range(count):
            expanded.append(
                JsonChain(
                    pdb_id=expected_pdb_id,
                    json_entry_index=entry_index,
                    copy_index=copy_index,
                    sequence=sequence,
                )
            )
    if not expanded:
        raise ValueError("JSON 中没有 RNA sequence")
    return task_name, expanded


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    pdb_id: str,
    detail: str,
) -> None:
    issues.append(
        {
            "SEVERITY": severity,
            "CODE": code,
            "PDB_ID": pdb_id,
            "DETAIL": detail,
        }
    )


def map_json_chains(
    pdb_id: str,
    split: str,
    split_status: str,
    json_chains: Sequence[JsonChain],
    manifest_chains: Sequence[ManifestChain],
    max_residues: int,
) -> list[dict[str, Any]]:
    by_sequence: dict[str, list[ManifestChain]] = defaultdict(list)
    for chain in manifest_chains:
        by_sequence[chain.sequence].append(chain)
    used_chain_ids: set[str] = set()

    json_sequence_counts = Counter(item.sequence for item in json_chains)
    rows: list[dict[str, Any]] = []
    for chain_index, item in enumerate(json_chains):
        candidates = [
            candidate
            for candidate in by_sequence.get(item.sequence, [])
            if candidate.chain_id not in used_chain_ids
        ]
        if candidates:
            matched = candidates[0]
            used_chain_ids.add(matched.chain_id)
            manifest_same_count = len(by_sequence[item.sequence])
            mapping_status = (
                "EXACT_UNIQUE_SEQUENCE"
                if json_sequence_counts[item.sequence] == 1
                and manifest_same_count == 1
                else "IDENTICAL_SEQUENCE_ORDER_ASSUMED"
            )
            original_chain_id = matched.chain_id
            entity_id = matched.entity_id
        else:
            mapping_status = "NO_SEQUENCE_MATCH"
            original_chain_id = ""
            entity_id = ""

        unknown = "".join(sorted(set(item.sequence) - VALID_BASES))
        rows.append(
            {
                "PDB_ID": pdb_id,
                "SPLIT": split,
                "SPLIT_STATUS": split_status,
                "JSON_ENTRY_INDEX": item.json_entry_index,
                "COPY_INDEX": item.copy_index + 1,
                "EXPECTED_PROTENIX_CHAIN_ID": int_to_letters(chain_index + 1),
                "ORIGINAL_CHAIN_ID": original_chain_id,
                "ENTITY_ID": entity_id,
                "SEQUENCE_LENGTH": len(item.sequence),
                "SEQUENCE_SHA256": sequence_sha256(item.sequence),
                "SEQUENCE": item.sequence,
                "UNKNOWN_SYMBOLS": unknown,
                "OVER_MAX_LENGTH": len(item.sequence) > max_residues,
                "MAPPING_STATUS": mapping_status,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在生成 RNA-FM embedding 前，审计 2241 个原始 Protenix JSON、"
            "CIF chain 清单和 train/val/test 归属。"
        )
    )
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument(
        "--chain-manifest", type=Path, default=DEFAULT_CHAIN_MANIFEST
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "数据划分 final_manifest.tsv；不指定时会在 report-root 下自动选择"
            "最新的 DATA_SPLIT_2241_CHAINMASK_*_EXECUTE/final_manifest.tsv。"
        ),
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--report-name",
        help="可选的固定报告目录名；默认使用 UTC 时间戳。",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDED),
        help="需要排除的 PDB ID。",
    )
    parser.add_argument("--expected-source-count", type=int, default=2246)
    parser.add_argument("--expected-target-count", type=int, default=2241)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="RNA-FM 可接受的最大残基数（默认 1024 token 减 BOS/EOS）。",
    )
    parser.add_argument(
        "--non-acgu-severity",
        choices=("WARNING", "ERROR"),
        default="WARNING",
        help="含 N/X/I/T 等非 ACGU 字符时的严重级别。默认保留并警告。",
    )
    parser.add_argument(
        "--overlength-severity",
        choices=("WARNING", "ERROR"),
        default="WARNING",
        help="链长超过 max-residues 时的严重级别。默认按后续滑窗策略警告。",
    )
    return parser


def run_audit(args: argparse.Namespace) -> AuditResult:
    started = utc_now()
    report_dir = create_report_dir(args.report_root.resolve(), args.report_name)
    logger = AuditLogger(report_dir / "audit.log")
    issues: list[dict[str, str]] = []
    pdb_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []

    logger.log(f"RNA-FM input audit {PIPELINE_VERSION}")
    logger.log(f"Started UTC: {started.isoformat()}")
    logger.log(f"JSON dir: {args.json_dir.resolve()}")
    logger.log(f"Chain manifest: {args.chain_manifest.resolve()}")
    logger.log(f"Split root: {args.split_root.resolve()}")
    split_manifest_path = (
        args.split_manifest.resolve()
        if args.split_manifest is not None
        else discover_split_manifest(args.report_root.resolve())
    )
    logger.log(
        "Split manifest: "
        + (str(split_manifest_path) if split_manifest_path else "NOT FOUND")
    )
    logger.log(f"Report dir: {report_dir}")

    excluded = {normalize_pdb_id(item) for item in args.exclude}
    raw_index, final_index = index_json_files(args.json_dir.resolve())
    manifest = load_manifest(args.chain_manifest.resolve())
    split_index, invalid_split_names = index_splits(args.split_root.resolve())
    split_manifest = (
        load_split_manifest(split_manifest_path) if split_manifest_path else None
    )

    duplicate_raw_ids = {
        pdb_id: paths for pdb_id, paths in raw_index.items() if len(paths) != 1
    }
    for pdb_id, paths in sorted(duplicate_raw_ids.items()):
        add_issue(
            issues,
            "ERROR",
            "DUPLICATE_RAW_JSON",
            pdb_id,
            " | ".join(map(str, paths)),
        )

    raw_ids = set(raw_index)
    target_ids = sorted(raw_ids - excluded)
    missing_exclusions = sorted(excluded - raw_ids)
    if len(raw_ids) != args.expected_source_count:
        add_issue(
            issues,
            "ERROR",
            "SOURCE_COUNT_MISMATCH",
            "",
            f"actual={len(raw_ids)}, expected={args.expected_source_count}",
        )
    if len(target_ids) != args.expected_target_count:
        add_issue(
            issues,
            "ERROR",
            "TARGET_COUNT_MISMATCH",
            "",
            f"actual={len(target_ids)}, expected={args.expected_target_count}",
        )
    if missing_exclusions:
        add_issue(
            issues,
            "ERROR",
            "EXCLUDED_IDS_NOT_FOUND",
            "",
            ",".join(missing_exclusions),
        )
    for path in invalid_split_names:
        add_issue(issues, "WARNING", "IGNORED_SPLIT_DIRECTORY", "", path)

    extra_split_ids = sorted(set(split_index) - set(target_ids))
    for pdb_id in extra_split_ids:
        add_issue(
            issues,
            "ERROR",
            "SPLIT_ID_NOT_TARGET",
            pdb_id,
            ",".join(split_index[pdb_id]),
        )

    if split_manifest is not None:
        for pdb_id in sorted(set(split_manifest) - set(target_ids)):
            add_issue(
                issues,
                "ERROR",
                "SPLIT_MANIFEST_ID_NOT_TARGET",
                pdb_id,
                f"manifest={split_manifest_path}",
            )

    parsed_pdb_count = 0
    mapped_chain_count = 0
    long_chain_count = 0
    non_acgu_chain_count = 0
    identical_order_count = 0
    sequence_multiset_mismatch_count = 0
    split_counts: Counter[str] = Counter()
    expected_dropped_count = 0

    for pdb_id in target_ids:
        row_error_codes: list[str] = []
        row_warning_codes: list[str] = []
        raw_paths = raw_index[pdb_id]
        raw_path = sorted(raw_paths, key=str)[0]
        splits = split_index.get(pdb_id, [])
        split = splits[0] if len(splits) == 1 else ""
        split_status = ""
        if split_manifest is None:
            if len(splits) != 1:
                code = "MISSING_SPLIT" if not splits else "DUPLICATE_SPLIT"
                row_error_codes.append(code)
                add_issue(issues, "ERROR", code, pdb_id, f"splits={splits}")
            else:
                split_counts[split] += 1
        else:
            split_record = split_manifest.get(pdb_id)
            if split_record is None:
                code = "MISSING_SPLIT_MANIFEST_RECORD"
                row_error_codes.append(code)
                add_issue(
                    issues,
                    "ERROR",
                    code,
                    pdb_id,
                    f"manifest={split_manifest_path}",
                )
            else:
                split_status = split_record.final_status
                if split_record.final_split:
                    split = split_record.final_split
                    if splits != [split_record.final_split]:
                        code = "SPLIT_MANIFEST_DIRECTORY_MISMATCH"
                        row_error_codes.append(code)
                        add_issue(
                            issues,
                            "ERROR",
                            code,
                            pdb_id,
                            f"manifest_final_split={split_record.final_split}; "
                            f"directories={splits}",
                        )
                    else:
                        split_counts[split] += 1
                elif split_record.final_status == "DROP_NO_EVALUABLE_CHAINS":
                    expected_dropped_count += 1
                    split = ""
                    if splits:
                        code = "DROPPED_PDB_HAS_SPLIT_DIRECTORY"
                        row_error_codes.append(code)
                        add_issue(
                            issues,
                            "ERROR",
                            code,
                            pdb_id,
                            f"directories={splits}",
                        )
                else:
                    code = "SPLIT_MANIFEST_NO_FINAL_SPLIT"
                    row_error_codes.append(code)
                    add_issue(
                        issues,
                        "ERROR",
                        code,
                        pdb_id,
                        f"final_status={split_record.final_status!r}",
                    )

        task_name = ""
        json_chains: list[JsonChain] = []
        try:
            task_name, json_chains = parse_raw_json(raw_path, pdb_id)
            parsed_pdb_count += 1
        except Exception as exc:
            row_error_codes.append("JSON_PARSE_ERROR")
            add_issue(
                issues,
                "ERROR",
                "JSON_PARSE_ERROR",
                pdb_id,
                f"{type(exc).__name__}: {exc}; path={raw_path}",
            )

        if task_name and task_name != pdb_id:
            row_error_codes.append("TASK_NAME_MISMATCH")
            add_issue(
                issues,
                "ERROR",
                "TASK_NAME_MISMATCH",
                pdb_id,
                f"task_name={task_name!r}",
            )

        manifest_chains = manifest.get(pdb_id, [])
        json_counter = Counter(item.sequence for item in json_chains)
        manifest_counter = Counter(item.sequence for item in manifest_chains)
        sequence_match = bool(json_chains) and json_counter == manifest_counter
        if not sequence_match:
            sequence_multiset_mismatch_count += 1
            row_error_codes.append("SEQUENCE_MULTISET_MISMATCH")
            add_issue(
                issues,
                "ERROR",
                "SEQUENCE_MULTISET_MISMATCH",
                pdb_id,
                "JSON="
                + json.dumps(dict(json_counter), ensure_ascii=False, sort_keys=True)
                + "; MANIFEST="
                + json.dumps(
                    dict(manifest_counter), ensure_ascii=False, sort_keys=True
                ),
            )

        current_chain_rows = map_json_chains(
            pdb_id,
            split,
            split_status,
            json_chains,
            manifest_chains,
            args.max_residues,
        )
        chain_rows.extend(current_chain_rows)

        for chain_row in current_chain_rows:
            mapping_status = chain_row["MAPPING_STATUS"]
            if mapping_status == "NO_SEQUENCE_MATCH":
                if "CHAIN_MAPPING_FAILED" not in row_error_codes:
                    row_error_codes.append("CHAIN_MAPPING_FAILED")
                add_issue(
                    issues,
                    "ERROR",
                    "CHAIN_MAPPING_FAILED",
                    pdb_id,
                    f"json_entry={chain_row['JSON_ENTRY_INDEX']}, "
                    f"copy={chain_row['COPY_INDEX']}",
                )
            else:
                mapped_chain_count += 1
            if mapping_status == "IDENTICAL_SEQUENCE_ORDER_ASSUMED":
                identical_order_count += 1
            if chain_row["UNKNOWN_SYMBOLS"]:
                non_acgu_chain_count += 1
                destination = (
                    row_error_codes
                    if args.non_acgu_severity == "ERROR"
                    else row_warning_codes
                )
                if "NON_ACGU_SEQUENCE" not in destination:
                    destination.append("NON_ACGU_SEQUENCE")
                add_issue(
                    issues,
                    args.non_acgu_severity,
                    "NON_ACGU_SEQUENCE",
                    pdb_id,
                    f"chain={chain_row['ORIGINAL_CHAIN_ID'] or '?'}; "
                    f"symbols={chain_row['UNKNOWN_SYMBOLS']}",
                )
            if chain_row["OVER_MAX_LENGTH"]:
                long_chain_count += 1
                destination = (
                    row_error_codes
                    if args.overlength_severity == "ERROR"
                    else row_warning_codes
                )
                if "OVER_MAX_LENGTH" not in destination:
                    destination.append("OVER_MAX_LENGTH")
                add_issue(
                    issues,
                    args.overlength_severity,
                    "OVER_MAX_LENGTH",
                    pdb_id,
                    f"chain={chain_row['ORIGINAL_CHAIN_ID'] or '?'}; "
                    f"length={chain_row['SEQUENCE_LENGTH']}; "
                    f"max={args.max_residues}",
                )

        max_length = max(
            (len(item.sequence) for item in json_chains), default=0
        )
        pdb_rows.append(
            {
                "PDB_ID": pdb_id,
                "RAW_JSON": str(raw_path),
                "TASK_NAME": task_name,
                "SPLIT": split,
                "SPLIT_STATUS": split_status,
                "JSON_RNA_CHAIN_COUNT": len(json_chains),
                "MANIFEST_RNA_CHAIN_COUNT": len(manifest_chains),
                "MAX_CHAIN_LENGTH": max_length,
                "SEQUENCE_MULTISET_MATCH": sequence_match,
                "HAS_NON_ACGU": any(
                    bool(item["UNKNOWN_SYMBOLS"])
                    for item in current_chain_rows
                ),
                "HAS_OVERLENGTH_CHAIN": any(
                    bool(item["OVER_MAX_LENGTH"])
                    for item in current_chain_rows
                ),
                "STATUS": (
                    "FAIL"
                    if row_error_codes
                    else "WARN"
                    if row_warning_codes
                    else "PASS"
                ),
                "ISSUE_CODES": ",".join(row_error_codes + row_warning_codes),
            }
        )

    error_count = sum(item["SEVERITY"] == "ERROR" for item in issues)
    warning_count = sum(item["SEVERITY"] == "WARNING" for item in issues)
    finished = utc_now()
    status = "PASS" if error_count == 0 else "NEEDS_REVIEW"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "status": status,
        "inputs": {
            "json_dir": str(args.json_dir.resolve()),
            "chain_manifest": str(args.chain_manifest.resolve()),
            "split_root": str(args.split_root.resolve()),
            "split_manifest": (
                str(split_manifest_path) if split_manifest_path else None
            ),
            "excluded_pdb_ids": sorted(excluded),
            "max_residues": args.max_residues,
        },
        "counts": {
            "raw_json_pdb_ids": len(raw_ids),
            "raw_json_files": sum(map(len, raw_index.values())),
            "final_updated_pdb_ids_seen": len(final_index),
            "final_updated_files_seen": sum(map(len, final_index.values())),
            "excluded_ids_found": len(excluded & raw_ids),
            "target_pdb_ids": len(target_ids),
            "parsed_target_pdb_ids": parsed_pdb_count,
            "pdb_rows": len(pdb_rows),
            "expanded_json_rna_chains": len(chain_rows),
            "mapped_original_chains": mapped_chain_count,
            "identical_sequence_order_assumed_chains": identical_order_count,
            "manifest_target_chains": sum(
                len(manifest.get(pdb_id, [])) for pdb_id in target_ids
            ),
            "sequence_multiset_mismatch_pdbs": sequence_multiset_mismatch_count,
            "non_acgu_chains": non_acgu_chain_count,
            "overlength_chains": long_chain_count,
            "split_counts": dict(sorted(split_counts.items())),
            "expected_dropped_pdbs": expected_dropped_count,
            "error_issues": error_count,
            "warning_issues": warning_count,
        },
        "outputs": {
            "pdb_audit_tsv": str(report_dir / "pdb_audit.tsv"),
            "chain_audit_tsv": str(report_dir / "chain_audit.tsv"),
            "issues_tsv": str(report_dir / "issues.tsv"),
            "log": str(report_dir / "audit.log"),
        },
    }

    write_tsv(report_dir / "pdb_audit.tsv", PDB_COLUMNS, pdb_rows)
    write_tsv(report_dir / "chain_audit.tsv", CHAIN_COLUMNS, chain_rows)
    write_tsv(report_dir / "issues.tsv", ISSUE_COLUMNS, issues)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.log("")
    logger.log("=== SUMMARY ===")
    logger.log(f"Raw JSON PDB IDs: {len(raw_ids)}")
    logger.log(f"Final target PDB IDs: {len(target_ids)}")
    logger.log(f"Parsed target PDB IDs: {parsed_pdb_count}")
    logger.log(f"Expanded RNA chains: {len(chain_rows)}")
    logger.log(f"Manifest target chains: {summary['counts']['manifest_target_chains']}")
    logger.log(f"Mapped original chains: {mapped_chain_count}")
    logger.log(f"Split counts: {dict(sorted(split_counts.items()))}")
    logger.log(f"Expected dropped PDBs: {expected_dropped_count}")
    logger.log(f"Sequence multiset mismatch PDBs: {sequence_multiset_mismatch_count}")
    logger.log(f"Non-ACGU chains: {non_acgu_chain_count}")
    logger.log(f"Overlength chains: {long_chain_count}")
    logger.log(f"Errors: {error_count}; warnings: {warning_count}")
    logger.log(f"FINAL STATUS: {status}")
    logger.log(f"Report: {report_dir}")
    return AuditResult(report_dir=report_dir, summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_audit(args)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0 if result.summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
