#!/usr/bin/env python3
"""Generate per-residue RNA-FM embeddings from a passed input audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


GENERATOR_VERSION = "1.0-rnafm-residue-embedding"
SCHEMA_VERSION = 1
MODEL_NAME = "rna_fm_t12"
DEFAULT_HOME = Path("/storage9920/home/tinghao.xia")
DEFAULT_REPORT_ROOT = DEFAULT_HOME / "Code/pipeline_reports"
DEFAULT_OUTPUT_ROOT = DEFAULT_HOME / "Data_FM/RNA_FM_embeddings"
OUTPUT_FILENAME = "rnafm_t12_residue_embeddings.pt"


@dataclass(frozen=True)
class ChainRecord:
    pdb_id: str
    split: str
    split_status: str
    json_entry_index: int
    copy_index: int
    expected_protenix_chain_id: str
    original_chain_id: str
    entity_id: str
    sequence: str
    sequence_sha256: str
    mapping_status: str


@dataclass
class SequenceEmbedding:
    embedding: Any
    token_ids: Any
    method: str
    window_starts: list[int]
    coverage_min: int
    coverage_max: int


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_text(value: datetime | None = None) -> str:
    return (value or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def normalize_pdb_id(value: str) -> str:
    value = value.strip().upper()
    if len(value) != 4 or not value.isalnum():
        raise ValueError(f"非法 PDB ID：{value!r}")
    return value


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii", errors="strict")).hexdigest()


def create_report_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"RNA_FM_EMBED_{timestamp_text()}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = Path(f"{base}_{counter}")
        counter += 1
    candidate.mkdir()
    return candidate


def write_tsv(
    path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def discover_passed_audit(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("RNA_FM_AUDIT_*/summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") == "PASS" and (
            summary_path.parent / "chain_audit.tsv"
        ).is_file():
            return summary_path.parent.resolve()
    raise FileNotFoundError(
        f"在 {report_root} 下没有找到包含 chain_audit.tsv 的 PASS 审计报告"
    )


def resolve_audit_report(
    requested: Path | None, report_root: Path
) -> tuple[Path, dict[str, Any]]:
    audit_dir = (
        requested.resolve()
        if requested is not None
        else discover_passed_audit(report_root)
    )
    if audit_dir.is_file():
        audit_dir = audit_dir.parent
    summary_path = audit_dir / "summary.json"
    chain_path = audit_dir / "chain_audit.tsv"
    if not summary_path.is_file() or not chain_path.is_file():
        raise FileNotFoundError(
            f"审计目录必须包含 summary.json 和 chain_audit.tsv：{audit_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError(
            f"只允许使用 PASS 审计报告；实际状态={summary.get('status')!r}"
        )
    return audit_dir.resolve(), summary


def load_chain_audit(path: Path) -> list[ChainRecord]:
    required = {
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
        "MAPPING_STATUS",
    }
    records: list[ChainRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"chain_audit.tsv 缺少列：{sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            sequence = str(row["SEQUENCE"]).strip().upper()
            if not sequence:
                raise ValueError(f"chain_audit.tsv 第 {row_number} 行序列为空")
            expected_length = int(row["SEQUENCE_LENGTH"])
            if len(sequence) != expected_length:
                raise ValueError(
                    f"chain_audit.tsv 第 {row_number} 行长度不一致："
                    f"sequence={len(sequence)}, column={expected_length}"
                )
            digest = sequence_sha256(sequence)
            if digest != row["SEQUENCE_SHA256"]:
                raise ValueError(
                    f"chain_audit.tsv 第 {row_number} 行序列 SHA256 不一致"
                )
            mapping_status = str(row["MAPPING_STATUS"]).strip()
            if mapping_status == "NO_SEQUENCE_MATCH":
                raise ValueError(
                    f"chain_audit.tsv 第 {row_number} 行链映射失败"
                )
            records.append(
                ChainRecord(
                    pdb_id=normalize_pdb_id(row["PDB_ID"]),
                    split=str(row["SPLIT"]).strip(),
                    split_status=str(row["SPLIT_STATUS"]).strip(),
                    json_entry_index=int(row["JSON_ENTRY_INDEX"]),
                    copy_index=int(row["COPY_INDEX"]),
                    expected_protenix_chain_id=str(
                        row["EXPECTED_PROTENIX_CHAIN_ID"]
                    ).strip(),
                    original_chain_id=str(row["ORIGINAL_CHAIN_ID"]).strip(),
                    entity_id=str(row["ENTITY_ID"]).strip(),
                    sequence=sequence,
                    sequence_sha256=digest,
                    mapping_status=mapping_status,
                )
            )
    if not records:
        raise ValueError("chain_audit.tsv 没有链记录")
    return records


def get_model_max_tokens(model: Any) -> int:
    args = getattr(model, "args", None)
    value = getattr(args, "max_positions", None)
    if isinstance(value, int):
        return value
    value = getattr(model, "max_positions", None)
    if callable(value):
        value = value()
    if isinstance(value, int):
        return value
    raise RuntimeError("无法从 RNA-FM 模型读取 max_positions")


def sliding_window_starts(
    sequence_length: int, window_size: int, overlap: int
) -> list[int]:
    if sequence_length <= window_size:
        return [0]
    if not 0 <= overlap < window_size:
        raise ValueError(
            f"window overlap 必须满足 0 <= overlap < {window_size}"
        )
    stride = window_size - overlap
    starts = list(range(0, sequence_length - window_size + 1, stride))
    final_start = sequence_length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_batches(
    items: Sequence[tuple[str, str]],
    special_tokens: int,
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[tuple[str, str]]]:
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_max = 0
    for item in items:
        length = len(item[1]) + special_tokens
        projected_max = max(current_max, length)
        projected_tokens = projected_max * (len(current) + 1)
        if current and (
            projected_tokens > max_batch_tokens
            or len(current) >= max_batch_size
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def validate_existing_output(
    path: Path,
    chains: Sequence[ChainRecord],
    torch_module: Any,
    repr_layer: int,
    embedding_dim: int,
    output_dtype: str,
    window_overlap: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        payload = torch_module.load(
            path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        return False, f"无法读取：{type(exc).__name__}: {exc}", None
    expected_hashes = [chain.sequence_sha256 for chain in chains]
    expected_offsets = [0]
    for chain in chains:
        expected_offsets.append(expected_offsets[-1] + len(chain.sequence))
    expected_residues = expected_offsets[-1]
    checks = {
        "schema_version": payload.get("schema_version") == SCHEMA_VERSION,
        "pdb_id": payload.get("pdb_id") == chains[0].pdb_id,
        "model_name": payload.get("model_name") == MODEL_NAME,
        "repr_layer": payload.get("repr_layer") == repr_layer,
        "embedding_dim": payload.get("embedding_dim") == embedding_dim,
        "embedding_dtype": payload.get("embedding_dtype") == output_dtype,
        "window_overlap": payload.get("window_overlap") == window_overlap,
        "sequence_hashes": payload.get("sequence_sha256") == expected_hashes,
        "original_chain_ids": payload.get("original_chain_ids")
        == [chain.original_chain_id for chain in chains],
        "expected_protenix_chain_ids": payload.get(
            "expected_protenix_chain_ids"
        )
        == [chain.expected_protenix_chain_id for chain in chains],
    }
    embedding = payload.get("residue_embedding")
    checks["embedding_shape"] = (
        hasattr(embedding, "shape")
        and tuple(embedding.shape) == (expected_residues, embedding_dim)
    )
    offsets = payload.get("chain_offsets")
    checks["chain_offsets"] = (
        hasattr(offsets, "tolist") and offsets.tolist() == expected_offsets
    )
    for key in (
        "residue_token_id",
        "residue_chain_index",
        "residue_index_in_chain",
        "residue_non_acgu",
        "residue_is_unk",
    ):
        tensor = payload.get(key)
        checks[f"{key}_shape"] = (
            hasattr(tensor, "shape")
            and tuple(tensor.shape) == (expected_residues,)
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        return False, "校验失败：" + ",".join(failed), payload
    return True, "VALID", payload


def atomic_torch_save(
    payload: dict[str, Any], path: Path, torch_module: Any
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        torch_module.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "根据 PASS RNA-FM 输入审计，为每条 RNA 链生成逐残基 embedding。"
        )
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="PASS 的 RNA_FM_AUDIT_* 目录；默认自动选择最新 PASS 报告。",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repr-layer", type=int, default=12)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument(
        "--pdb-id",
        nargs="*",
        help="只处理指定 PDB ID；适合 smoke test。默认处理审计中的全部 PDB。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="按 PDB ID 排序后最多处理多少个；仅用于测试。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新计算并原子替换已有输出。默认校验后跳过。",
    )
    parser.add_argument(
        "--no-sequence-cache",
        action="store_true",
        help="每个 PDB 保存后清空序列 embedding 内存缓存。",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    import torch
    import fm

    if args.max_batch_tokens < 1 or args.max_batch_size < 1:
        raise ValueError("batch 参数必须为正整数")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须为正整数")

    started = utc_now()
    report_dir = create_report_dir(args.report_root.resolve())
    logger = Logger(report_dir / "generation.log")
    pdb_manifest_rows: list[dict[str, Any]] = []
    chain_manifest_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    status = "RUNNING"
    completed = 0
    skipped = 0
    failed = 0

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "started_utc": started.isoformat(),
        "status": status,
        "report_dir": str(report_dir),
    }

    try:
        audit_dir, audit_summary = resolve_audit_report(
            args.audit_report, args.report_root.resolve()
        )
        records = load_chain_audit(audit_dir / "chain_audit.tsv")
        audited_chain_count = audit_summary.get("counts", {}).get(
            "expanded_json_rna_chains"
        )
        if audited_chain_count != len(records):
            raise ValueError(
                "summary.json 与 chain_audit.tsv 链数不一致："
                f"summary={audited_chain_count}, rows={len(records)}"
            )
        grouped: dict[str, list[ChainRecord]] = defaultdict(list)
        for record in records:
            grouped[record.pdb_id].append(record)

        selected_ids = sorted(grouped)
        if args.pdb_id is not None:
            requested_ids = {normalize_pdb_id(item) for item in args.pdb_id}
            missing_ids = sorted(requested_ids - set(grouped))
            if missing_ids:
                raise ValueError(
                    "请求的 PDB ID 不在审计报告中：" + ",".join(missing_ids)
                )
            selected_ids = [item for item in selected_ids if item in requested_ids]
        if args.limit is not None:
            selected_ids = selected_ids[: args.limit]
        if not selected_ids:
            raise ValueError("没有选中需要处理的 PDB")

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False")

        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        logger.log(f"RNA-FM embedding generator {GENERATOR_VERSION}")
        logger.log(f"Audit report: {audit_dir}")
        logger.log(f"Output root: {output_root}")
        logger.log(f"Selected PDBs: {len(selected_ids)}")
        logger.log(
            "Selected chains: "
            f"{sum(len(grouped[pdb_id]) for pdb_id in selected_ids)}"
        )
        logger.log(f"Device: {device}")
        if device.type == "cuda":
            logger.log(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.log("Loading rna_fm_t12 ...")

        model, alphabet = fm.pretrained.rna_fm_t12()
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        model_max_tokens = get_model_max_tokens(model)
        special_tokens = int(bool(alphabet.prepend_bos)) + int(
            bool(alphabet.append_eos)
        )
        max_residues = model_max_tokens - special_tokens
        if max_residues < 1:
            raise RuntimeError(
                f"非法最大长度：tokens={model_max_tokens}, special={special_tokens}"
            )
        if not 0 <= args.window_overlap < max_residues:
            raise ValueError(
                f"--window-overlap 必须位于 [0, {max_residues - 1}]"
            )

        embedding_dim = int(getattr(model.args, "embed_dim"))
        if embedding_dim != 640:
            raise RuntimeError(
                f"RNA-FM embedding_dim 应为 640，实际为 {embedding_dim}"
            )
        output_torch_dtype = (
            torch.float32 if args.output_dtype == "float32" else torch.float16
        )
        logger.log(f"Model max tokens: {model_max_tokens}")
        logger.log(f"Max residues per window: {max_residues}")
        logger.log(f"Window overlap: {args.window_overlap}")
        logger.log(f"Embedding dim: {embedding_dim}")
        logger.log(f"Output dtype: {args.output_dtype}")
        logger.log(f"unk_idx: {alphabet.unk_idx}")

        observed_symbols = sorted(
            {
                symbol
                for pdb_id in selected_ids
                for chain in grouped[pdb_id]
                for symbol in chain.sequence
            }
        )
        token_policy = {
            symbol: {
                "token_id": int(alphabet.get_idx(symbol)),
                "token_text": alphabet.get_tok(alphabet.get_idx(symbol)),
                "is_unk": alphabet.get_idx(symbol) == alphabet.unk_idx,
            }
            for symbol in observed_symbols
        }
        logger.log(
            "Observed token policy: "
            + json.dumps(token_policy, ensure_ascii=False, sort_keys=True)
        )

        sequence_cache: dict[str, SequenceEmbedding] = {}

        def infer_raw_sequences(
            items: Sequence[tuple[str, str]]
        ) -> dict[str, tuple[Any, Any]]:
            outputs: dict[str, tuple[Any, Any]] = {}
            batches = make_batches(
                items,
                special_tokens,
                args.max_batch_tokens,
                args.max_batch_size,
            )
            offset = int(bool(alphabet.prepend_bos))
            for batch in batches:
                labels, _, tokens = batch_converter(batch)
                if tokens.shape[1] > model_max_tokens:
                    raise RuntimeError(
                        f"batch token 长度 {tokens.shape[1]} 超过 {model_max_tokens}"
                    )
                tokens = tokens.to(device)
                with torch.inference_mode():
                    representations = model(
                        tokens, repr_layers=[args.repr_layer]
                    )["representations"][args.repr_layer]
                for index, ((key, sequence), label) in enumerate(
                    zip(batch, labels)
                ):
                    if key != label:
                        raise RuntimeError("batch label 顺序发生变化")
                    length = len(sequence)
                    embedding = representations[
                        index, offset : offset + length
                    ].detach().to(device="cpu", dtype=torch.float32).contiguous()
                    token_ids = tokens[
                        index, offset : offset + length
                    ].detach().to(device="cpu").contiguous()
                    if tuple(embedding.shape) != (length, embedding_dim):
                        raise RuntimeError(
                            f"embedding 形状异常：key={key}, "
                            f"shape={tuple(embedding.shape)}"
                        )
                    outputs[key] = (embedding, token_ids)
                del representations, tokens
            return outputs

        def compute_sequence(sequence: str) -> SequenceEmbedding:
            digest = sequence_sha256(sequence)
            if len(sequence) <= max_residues:
                raw = infer_raw_sequences([(digest, sequence)])[digest]
                return SequenceEmbedding(
                    embedding=raw[0],
                    token_ids=raw[1],
                    method="full_sequence",
                    window_starts=[0],
                    coverage_min=1,
                    coverage_max=1,
                )

            starts = sliding_window_starts(
                len(sequence), max_residues, args.window_overlap
            )
            window_items = [
                (f"{digest}:{start}", sequence[start : start + max_residues])
                for start in starts
            ]
            raw_windows = infer_raw_sequences(window_items)
            accumulator = torch.zeros(
                (len(sequence), embedding_dim), dtype=torch.float32
            )
            coverage = torch.zeros(len(sequence), dtype=torch.int32)
            for start in starts:
                key = f"{digest}:{start}"
                window_embedding, _ = raw_windows[key]
                end = start + window_embedding.shape[0]
                accumulator[start:end] += window_embedding
                coverage[start:end] += 1
            if int(coverage.min()) < 1:
                raise RuntimeError(f"滑窗未覆盖完整序列：{digest}")
            accumulator /= coverage.to(torch.float32).unsqueeze(1)
            token_ids = torch.tensor(
                [alphabet.get_idx(symbol) for symbol in sequence],
                dtype=torch.int64,
            )
            return SequenceEmbedding(
                embedding=accumulator.contiguous(),
                token_ids=token_ids,
                method="sliding_window_uniform_mean",
                window_starts=starts,
                coverage_min=int(coverage.min()),
                coverage_max=int(coverage.max()),
            )

        total_selected = len(selected_ids)
        for pdb_number, pdb_id in enumerate(selected_ids, start=1):
            chains = grouped[pdb_id]
            output_path = output_root / pdb_id / OUTPUT_FILENAME
            if output_path.exists() and not args.overwrite:
                valid, detail, existing = validate_existing_output(
                    output_path,
                    chains,
                    torch,
                    args.repr_layer,
                    embedding_dim,
                    args.output_dtype,
                    args.window_overlap,
                )
                if not valid:
                    raise RuntimeError(
                        f"已有输出无效且未指定 --overwrite：{output_path}; {detail}"
                    )
                skipped += 1
                methods = existing.get("chain_embedding_method", [])
                pdb_manifest_rows.append(
                    {
                        "PDB_ID": pdb_id,
                        "STATUS": "SKIPPED_VALID",
                        "SPLIT": chains[0].split,
                        "SPLIT_STATUS": chains[0].split_status,
                        "CHAIN_COUNT": len(chains),
                        "RESIDUE_COUNT": sum(len(item.sequence) for item in chains),
                        "OUTPUT_PATH": str(output_path),
                        "METHODS": ",".join(sorted(set(methods))),
                        "DETAIL": detail,
                    }
                )
                existing_offsets = existing["chain_offsets"].tolist()
                existing_token_ids = existing["residue_token_id"]
                existing_methods = existing.get("chain_embedding_method", [])
                existing_starts = existing.get("chain_window_starts", [])
                for chain_index, chain in enumerate(chains):
                    start = existing_offsets[chain_index]
                    end = existing_offsets[chain_index + 1]
                    chain_token_ids = existing_token_ids[start:end]
                    window_starts = existing_starts[chain_index]
                    chain_manifest_rows.append(
                        {
                            "PDB_ID": pdb_id,
                            "CHAIN_INDEX": chain_index,
                            "ORIGINAL_CHAIN_ID": chain.original_chain_id,
                            "EXPECTED_PROTENIX_CHAIN_ID": (
                                chain.expected_protenix_chain_id
                            ),
                            "ENTITY_ID": chain.entity_id,
                            "SEQUENCE_LENGTH": len(chain.sequence),
                            "SEQUENCE_SHA256": chain.sequence_sha256,
                            "NON_ACGU_COUNT": sum(
                                symbol not in "ACGU"
                                for symbol in chain.sequence
                            ),
                            "UNK_TOKEN_COUNT": int(
                                chain_token_ids.eq(
                                    int(alphabet.unk_idx)
                                ).sum()
                            ),
                            "EMBEDDING_METHOD": existing_methods[chain_index],
                            "WINDOW_COUNT": len(window_starts),
                            "WINDOW_STARTS": ",".join(
                                map(str, window_starts)
                            ),
                            "OUTPUT_PATH": str(output_path),
                        }
                    )
                logger.log(
                    f"[{pdb_number}/{total_selected}] {pdb_id}: SKIP valid"
                )
                continue

            unique_sequences: dict[str, str] = {}
            for chain in chains:
                unique_sequences.setdefault(chain.sequence_sha256, chain.sequence)
            for digest, sequence in unique_sequences.items():
                if digest not in sequence_cache:
                    sequence_cache[digest] = compute_sequence(sequence)

            chain_results = [
                sequence_cache[chain.sequence_sha256] for chain in chains
            ]
            lengths = [len(chain.sequence) for chain in chains]
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            total_residues = offsets[-1]
            residue_embedding = torch.cat(
                [item.embedding for item in chain_results], dim=0
            ).to(dtype=output_torch_dtype).contiguous()
            residue_token_id = torch.cat(
                [item.token_ids for item in chain_results], dim=0
            ).to(dtype=torch.int64).contiguous()
            residue_chain_index = torch.cat(
                [
                    torch.full((length,), index, dtype=torch.int64)
                    for index, length in enumerate(lengths)
                ]
            )
            residue_index_in_chain = torch.cat(
                [torch.arange(length, dtype=torch.int64) for length in lengths]
            )
            residue_non_acgu = torch.tensor(
                [
                    symbol not in "ACGU"
                    for chain in chains
                    for symbol in chain.sequence
                ],
                dtype=torch.bool,
            )
            residue_is_unk = residue_token_id.eq(int(alphabet.unk_idx))
            if tuple(residue_embedding.shape) != (
                total_residues,
                embedding_dim,
            ):
                raise RuntimeError(f"{pdb_id} 拼接后的 embedding 形状异常")

            splits = {chain.split for chain in chains}
            split_statuses = {chain.split_status for chain in chains}
            if len(splits) != 1 or len(split_statuses) != 1:
                raise RuntimeError(f"{pdb_id} 的链级 split 元数据不一致")

            payload: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "created_utc": utc_now().isoformat(),
                "pdb_id": pdb_id,
                "split": chains[0].split,
                "split_status": chains[0].split_status,
                "source_audit_report": str(audit_dir),
                "model_name": MODEL_NAME,
                "repr_layer": args.repr_layer,
                "embedding_dim": embedding_dim,
                "embedding_dtype": args.output_dtype,
                "model_max_tokens": model_max_tokens,
                "max_residues_per_window": max_residues,
                "window_overlap": args.window_overlap,
                "window_merge": "uniform_mean",
                "prepend_bos": bool(alphabet.prepend_bos),
                "append_eos": bool(alphabet.append_eos),
                "unk_token_id": int(alphabet.unk_idx),
                "token_policy": token_policy,
                "chain_count": len(chains),
                "residue_count": total_residues,
                "original_chain_ids": [
                    chain.original_chain_id for chain in chains
                ],
                "expected_protenix_chain_ids": [
                    chain.expected_protenix_chain_id for chain in chains
                ],
                "entity_ids": [chain.entity_id for chain in chains],
                "json_entry_indices": [
                    chain.json_entry_index for chain in chains
                ],
                "copy_indices": [chain.copy_index for chain in chains],
                "mapping_status": [chain.mapping_status for chain in chains],
                "sequences": [chain.sequence for chain in chains],
                "sequence_sha256": [
                    chain.sequence_sha256 for chain in chains
                ],
                "chain_embedding_method": [
                    item.method for item in chain_results
                ],
                "chain_window_starts": [
                    item.window_starts for item in chain_results
                ],
                "chain_window_coverage_min": [
                    item.coverage_min for item in chain_results
                ],
                "chain_window_coverage_max": [
                    item.coverage_max for item in chain_results
                ],
                "chain_offsets": torch.tensor(offsets, dtype=torch.int64),
                "residue_embedding": residue_embedding,
                "residue_token_id": residue_token_id,
                "residue_chain_index": residue_chain_index,
                "residue_index_in_chain": residue_index_in_chain,
                "residue_non_acgu": residue_non_acgu,
                "residue_is_unk": residue_is_unk,
            }
            atomic_torch_save(payload, output_path, torch)
            valid, detail, _ = validate_existing_output(
                output_path,
                chains,
                torch,
                args.repr_layer,
                embedding_dim,
                args.output_dtype,
                args.window_overlap,
            )
            if not valid:
                raise RuntimeError(
                    f"写入后校验失败：{output_path}; {detail}"
                )

            completed += 1
            methods = sorted({item.method for item in chain_results})
            pdb_manifest_rows.append(
                {
                    "PDB_ID": pdb_id,
                    "STATUS": "GENERATED",
                    "SPLIT": chains[0].split,
                    "SPLIT_STATUS": chains[0].split_status,
                    "CHAIN_COUNT": len(chains),
                    "RESIDUE_COUNT": total_residues,
                    "OUTPUT_PATH": str(output_path),
                    "METHODS": ",".join(methods),
                    "DETAIL": "VALID",
                }
            )
            for chain_index, (chain, chain_result) in enumerate(
                zip(chains, chain_results)
            ):
                chain_manifest_rows.append(
                    {
                        "PDB_ID": pdb_id,
                        "CHAIN_INDEX": chain_index,
                        "ORIGINAL_CHAIN_ID": chain.original_chain_id,
                        "EXPECTED_PROTENIX_CHAIN_ID": (
                            chain.expected_protenix_chain_id
                        ),
                        "ENTITY_ID": chain.entity_id,
                        "SEQUENCE_LENGTH": len(chain.sequence),
                        "SEQUENCE_SHA256": chain.sequence_sha256,
                        "NON_ACGU_COUNT": sum(
                            symbol not in "ACGU" for symbol in chain.sequence
                        ),
                        "UNK_TOKEN_COUNT": int(
                            chain_result.token_ids.eq(
                                int(alphabet.unk_idx)
                            ).sum()
                        ),
                        "EMBEDDING_METHOD": chain_result.method,
                        "WINDOW_COUNT": len(chain_result.window_starts),
                        "WINDOW_STARTS": ",".join(
                            map(str, chain_result.window_starts)
                        ),
                        "OUTPUT_PATH": str(output_path),
                    }
                )
            logger.log(
                f"[{pdb_number}/{total_selected}] {pdb_id}: GENERATED "
                f"chains={len(chains)} residues={total_residues} "
                f"methods={','.join(methods)}"
            )
            if args.no_sequence_cache:
                sequence_cache.clear()

        status = "PASS"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        issues.append(
            {
                "SEVERITY": "ERROR",
                "CODE": "INTERRUPTED",
                "PDB_ID": "",
                "DETAIL": "KeyboardInterrupt",
            }
        )
        logger.log("Run interrupted by user; completed files remain resumable.")
    except Exception as exc:
        status = "FAIL"
        failed += 1
        issues.append(
            {
                "SEVERITY": "ERROR",
                "CODE": type(exc).__name__,
                "PDB_ID": "",
                "DETAIL": str(exc),
            }
        )
        logger.log(f"FATAL: {type(exc).__name__}: {exc}")

    finished = utc_now()
    summary.update(
        {
            "status": status,
            "finished_utc": finished.isoformat(),
            "counts": {
                "generated_pdbs": completed,
                "skipped_valid_pdbs": skipped,
                "failed_events": failed,
                "pdb_manifest_rows": len(pdb_manifest_rows),
                "chain_manifest_rows": len(chain_manifest_rows),
            },
            "outputs": {
                "output_root": str(args.output_root.resolve()),
                "pdb_manifest_tsv": str(report_dir / "pdb_manifest.tsv"),
                "chain_manifest_tsv": str(report_dir / "chain_manifest.tsv"),
                "issues_tsv": str(report_dir / "issues.tsv"),
                "log": str(report_dir / "generation.log"),
            },
        }
    )
    write_tsv(
        report_dir / "pdb_manifest.tsv",
        [
            "PDB_ID",
            "STATUS",
            "SPLIT",
            "SPLIT_STATUS",
            "CHAIN_COUNT",
            "RESIDUE_COUNT",
            "OUTPUT_PATH",
            "METHODS",
            "DETAIL",
        ],
        pdb_manifest_rows,
    )
    write_tsv(
        report_dir / "chain_manifest.tsv",
        [
            "PDB_ID",
            "CHAIN_INDEX",
            "ORIGINAL_CHAIN_ID",
            "EXPECTED_PROTENIX_CHAIN_ID",
            "ENTITY_ID",
            "SEQUENCE_LENGTH",
            "SEQUENCE_SHA256",
            "NON_ACGU_COUNT",
            "UNK_TOKEN_COUNT",
            "EMBEDDING_METHOD",
            "WINDOW_COUNT",
            "WINDOW_STARTS",
            "OUTPUT_PATH",
        ],
        chain_manifest_rows,
    )
    write_tsv(
        report_dir / "issues.tsv",
        ["SEVERITY", "CODE", "PDB_ID", "DETAIL"],
        issues,
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.log(f"FINAL STATUS: {status}")
    logger.log(f"Report: {report_dir}")
    if status == "PASS":
        return 0
    if status == "INTERRUPTED":
        return 130
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
