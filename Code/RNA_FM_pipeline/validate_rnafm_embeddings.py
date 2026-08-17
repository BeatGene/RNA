#!/usr/bin/env python3
"""Validate generated RNA-FM per-residue embedding files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

import generate_rnafm_embeddings as generator


VALIDATOR_VERSION = "1.0-rnafm-embedding-validator"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_text() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def create_report_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"RNA_FM_VALIDATE_{timestamp_text()}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = Path(f"{base}_{counter}")
        counter += 1
    candidate.mkdir()
    return candidate


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "PDB_ID",
        "STATUS",
        "CHAIN_COUNT",
        "RESIDUE_COUNT",
        "NON_ACGU_COUNT",
        "UNK_COUNT",
        "MIN_ABS_VALUE",
        "MAX_ABS_VALUE",
        "MIN_ROW_NORM",
        "MAX_ROW_NORM",
        "METHODS",
        "OUTPUT_PATH",
        "DETAIL",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计已生成的 RNA-FM residue embedding .pt 文件。"
    )
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument(
        "--report-root", type=Path, default=generator.DEFAULT_REPORT_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=generator.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--repr-layer", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=640)
    parser.add_argument(
        "--output-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--pdb-id", nargs="*")
    return parser


def validate_payload(
    payload: dict[str, Any],
    chains: Sequence[generator.ChainRecord],
) -> dict[str, Any]:
    errors: list[str] = []
    embedding = payload["residue_embedding"]
    token_ids = payload["residue_token_id"]
    chain_indices = payload["residue_chain_index"]
    indices_in_chain = payload["residue_index_in_chain"]
    non_acgu = payload["residue_non_acgu"]
    is_unk = payload["residue_is_unk"]
    offsets = payload["chain_offsets"].tolist()
    sequences = [chain.sequence for chain in chains]
    total_residues = sum(map(len, sequences))

    if not torch.isfinite(embedding).all().item():
        errors.append("NON_FINITE_EMBEDDING")
    row_norms = torch.linalg.vector_norm(embedding.to(torch.float32), dim=1)
    if not torch.isfinite(row_norms).all().item():
        errors.append("NON_FINITE_ROW_NORM")
    if (row_norms <= 0).any().item():
        errors.append("ZERO_ROW_NORM")

    expected_chain_indices = torch.cat(
        [
            torch.full((len(sequence),), index, dtype=torch.int64)
            for index, sequence in enumerate(sequences)
        ]
    )
    expected_indices_in_chain = torch.cat(
        [torch.arange(len(sequence), dtype=torch.int64) for sequence in sequences]
    )
    expected_non_acgu = torch.tensor(
        [symbol not in "ACGU" for sequence in sequences for symbol in sequence],
        dtype=torch.bool,
    )
    token_policy = payload["token_policy"]
    try:
        expected_token_ids = torch.tensor(
            [
                int(token_policy[symbol]["token_id"])
                for sequence in sequences
                for symbol in sequence
            ],
            dtype=torch.int64,
        )
    except KeyError as exc:
        errors.append(f"TOKEN_POLICY_MISSING_{exc.args[0]}")
        expected_token_ids = token_ids.clone()
    expected_is_unk = expected_token_ids.eq(int(payload["unk_token_id"]))

    tensor_checks = {
        "TOKEN_ID_MISMATCH": torch.equal(token_ids, expected_token_ids),
        "CHAIN_INDEX_MISMATCH": torch.equal(
            chain_indices, expected_chain_indices
        ),
        "INDEX_IN_CHAIN_MISMATCH": torch.equal(
            indices_in_chain, expected_indices_in_chain
        ),
        "NON_ACGU_MASK_MISMATCH": torch.equal(non_acgu, expected_non_acgu),
        "UNK_MASK_MISMATCH": torch.equal(is_unk, expected_is_unk),
    }
    errors.extend(code for code, passed in tensor_checks.items() if not passed)

    methods = payload["chain_embedding_method"]
    window_starts = payload["chain_window_starts"]
    max_residues = int(payload["max_residues_per_window"])
    overlap = int(payload["window_overlap"])
    for chain_index, (sequence, method, starts) in enumerate(
        zip(sequences, methods, window_starts)
    ):
        expected_starts = generator.sliding_window_starts(
            len(sequence), max_residues, overlap
        )
        if starts != expected_starts:
            errors.append(f"WINDOW_START_MISMATCH_CHAIN_{chain_index}")
        expected_method = (
            "full_sequence"
            if len(sequence) <= max_residues
            else "sliding_window_uniform_mean"
        )
        if method != expected_method:
            errors.append(f"METHOD_MISMATCH_CHAIN_{chain_index}")

    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, chain in enumerate(chains):
        by_hash[chain.sequence_sha256].append(index)
    for chain_group in by_hash.values():
        if len(chain_group) < 2:
            continue
        reference_index = chain_group[0]
        reference = embedding[
            offsets[reference_index] : offsets[reference_index + 1]
        ]
        for other_index in chain_group[1:]:
            other = embedding[offsets[other_index] : offsets[other_index + 1]]
            if not torch.equal(reference, other):
                errors.append(
                    f"IDENTICAL_SEQUENCE_EMBEDDING_MISMATCH_"
                    f"CHAINS_{reference_index}_{other_index}"
                )

    return {
        "errors": errors,
        "chain_count": len(chains),
        "residue_count": total_residues,
        "non_acgu_count": int(non_acgu.sum()),
        "unk_count": int(is_unk.sum()),
        "min_abs_value": float(embedding.abs().min()),
        "max_abs_value": float(embedding.abs().max()),
        "min_row_norm": float(row_norms.min()),
        "max_row_norm": float(row_norms.max()),
        "methods": sorted(set(methods)),
    }


def run(args: argparse.Namespace) -> int:
    report_dir = create_report_dir(args.report_root.resolve())
    logger = Logger(report_dir / "validation.log")
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    started = utc_now()

    try:
        audit_dir, audit_summary = generator.resolve_audit_report(
            args.audit_report, args.report_root.resolve()
        )
        records = generator.load_chain_audit(audit_dir / "chain_audit.tsv")
        expected_count = audit_summary.get("counts", {}).get(
            "expanded_json_rna_chains"
        )
        if expected_count != len(records):
            raise ValueError("审计 summary 与 chain_audit.tsv 链数不一致")
        grouped: dict[str, list[generator.ChainRecord]] = defaultdict(list)
        for record in records:
            grouped[record.pdb_id].append(record)

        selected_ids = sorted(grouped)
        if args.pdb_id is not None:
            requested = {
                generator.normalize_pdb_id(item) for item in args.pdb_id
            }
            missing = sorted(requested - set(grouped))
            if missing:
                raise ValueError("PDB ID 不在审计报告中：" + ",".join(missing))
            selected_ids = [item for item in selected_ids if item in requested]
        if not selected_ids:
            raise ValueError("没有选中待验证 PDB")

        logger.log(f"RNA-FM embedding validator {VALIDATOR_VERSION}")
        logger.log(f"Audit report: {audit_dir}")
        logger.log(f"Output root: {args.output_root.resolve()}")
        logger.log(f"Selected PDBs: {len(selected_ids)}")

        for number, pdb_id in enumerate(selected_ids, start=1):
            chains = grouped[pdb_id]
            output_path = (
                args.output_root.resolve()
                / pdb_id
                / generator.OUTPUT_FILENAME
            )
            if not output_path.is_file():
                detail = "MISSING_OUTPUT"
                issues.append(
                    {
                        "SEVERITY": "ERROR",
                        "CODE": detail,
                        "PDB_ID": pdb_id,
                        "DETAIL": str(output_path),
                    }
                )
                rows.append(
                    {
                        "PDB_ID": pdb_id,
                        "STATUS": "FAIL",
                        "OUTPUT_PATH": str(output_path),
                        "DETAIL": detail,
                    }
                )
                logger.log(f"[{number}/{len(selected_ids)}] {pdb_id}: FAIL missing")
                continue

            valid, detail, payload = generator.validate_existing_output(
                output_path,
                chains,
                torch,
                args.repr_layer,
                args.embedding_dim,
                args.output_dtype,
                args.window_overlap,
            )
            errors: list[str] = [] if valid else [detail]
            stats: dict[str, Any] = {
                "chain_count": len(chains),
                "residue_count": sum(len(chain.sequence) for chain in chains),
                "non_acgu_count": "",
                "unk_count": "",
                "min_abs_value": "",
                "max_abs_value": "",
                "min_row_norm": "",
                "max_row_norm": "",
                "methods": [],
            }
            if payload is not None and valid:
                stats = validate_payload(payload, chains)
                errors.extend(stats["errors"])
            status = "PASS" if not errors else "FAIL"
            for error in errors:
                issues.append(
                    {
                        "SEVERITY": "ERROR",
                        "CODE": "VALIDATION_ERROR",
                        "PDB_ID": pdb_id,
                        "DETAIL": error,
                    }
                )
            rows.append(
                {
                    "PDB_ID": pdb_id,
                    "STATUS": status,
                    "CHAIN_COUNT": stats["chain_count"],
                    "RESIDUE_COUNT": stats["residue_count"],
                    "NON_ACGU_COUNT": stats["non_acgu_count"],
                    "UNK_COUNT": stats["unk_count"],
                    "MIN_ABS_VALUE": stats["min_abs_value"],
                    "MAX_ABS_VALUE": stats["max_abs_value"],
                    "MIN_ROW_NORM": stats["min_row_norm"],
                    "MAX_ROW_NORM": stats["max_row_norm"],
                    "METHODS": ",".join(stats["methods"]),
                    "OUTPUT_PATH": str(output_path),
                    "DETAIL": ";".join(errors) if errors else "VALID",
                }
            )
            logger.log(
                f"[{number}/{len(selected_ids)}] {pdb_id}: {status} "
                f"chains={stats['chain_count']} residues={stats['residue_count']} "
                f"unk={stats['unk_count']} methods={','.join(stats['methods'])}"
            )

        error_count = len(issues)
        final_status = "PASS" if error_count == 0 else "FAIL"
    except Exception as exc:
        error_count = 1
        final_status = "FAIL"
        issues.append(
            {
                "SEVERITY": "ERROR",
                "CODE": type(exc).__name__,
                "PDB_ID": "",
                "DETAIL": str(exc),
            }
        )
        logger.log(f"FATAL: {type(exc).__name__}: {exc}")

    write_tsv(report_dir / "validation.tsv", rows)
    with (report_dir / "issues.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["SEVERITY", "CODE", "PDB_ID", "DETAIL"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(issues)
    summary = {
        "schema_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "started_utc": started.isoformat(),
        "finished_utc": utc_now().isoformat(),
        "status": final_status,
        "counts": {
            "validated_pdbs": sum(row.get("STATUS") == "PASS" for row in rows),
            "failed_pdbs": sum(row.get("STATUS") == "FAIL" for row in rows),
            "error_issues": error_count,
        },
        "outputs": {
            "validation_tsv": str(report_dir / "validation.tsv"),
            "issues_tsv": str(report_dir / "issues.tsv"),
            "log": str(report_dir / "validation.log"),
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.log(f"Errors: {error_count}")
    logger.log(f"FINAL STATUS: {final_status}")
    logger.log(f"Report: {report_dir}")
    return 0 if final_status == "PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
