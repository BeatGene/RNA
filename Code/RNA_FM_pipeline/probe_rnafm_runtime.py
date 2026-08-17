#!/usr/bin/env python3
"""Probe the installed RNA-FM runtime before bulk embedding generation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import fm


PROBE_VERSION = "1.0-rnafm-runtime-probe"
DEFAULT_REPORT_ROOT = Path(
    "/storage9920/home/tinghao.xia/Code/pipeline_reports"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_text(value: datetime | None = None) -> str:
    return (value or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def create_report_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"RNA_FM_RUNTIME_PROBE_{timestamp_text()}"
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


def get_model_max_tokens(model: torch.nn.Module) -> int | None:
    args = getattr(model, "args", None)
    value = getattr(args, "max_positions", None)
    if isinstance(value, int):
        return value

    value = getattr(model, "max_positions", None)
    if callable(value):
        value = value()
    if isinstance(value, int):
        return value
    return None


def token_mapping(
    alphabet: Any,
    batch_converter: Any,
    sequence: str,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    _, _, tokens = batch_converter([("token_probe", sequence)])
    offset = int(bool(alphabet.prepend_bos))
    residue_tokens = tokens[0, offset : offset + len(sequence)]
    if residue_tokens.numel() != len(sequence):
        raise RuntimeError(
            "token 数与序列长度不一致："
            f"tokens={residue_tokens.numel()}, residues={len(sequence)}"
        )

    rows: list[dict[str, Any]] = []
    standard_tokens = set(alphabet.standard_toks)
    for index, (symbol, token_id) in enumerate(
        zip(sequence, residue_tokens.tolist())
    ):
        rows.append(
            {
                "residue_index": index,
                "input_symbol": symbol,
                "token_id": token_id,
                "token_text": alphabet.get_tok(token_id),
                "is_standard_token": symbol in standard_tokens,
                "is_unk": token_id == alphabet.unk_idx,
            }
        )
    return rows, tuple(tokens.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 RNA-FM tokenizer、最大长度和 GPU 前向计算。"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repr-layer", type=int, default=12)
    parser.add_argument("--probe-sequence", default="ACGUNXIT")
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    parser.add_argument(
        "--skip-boundary-forward",
        action="store_true",
        help="只探测 tokenizer，不用最大长度序列做前向计算。",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    started = utc_now()
    report_dir = create_report_dir(args.report_root.resolve())
    logger = Logger(report_dir / "probe.log")
    result: dict[str, Any] = {
        "schema_version": 1,
        "probe_version": PROBE_VERSION,
        "started_utc": started.isoformat(),
        "status": "RUNNING",
        "report_dir": str(report_dir),
    }

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False")

        logger.log(f"RNA-FM runtime probe {PROBE_VERSION}")
        logger.log(f"PyTorch: {torch.__version__}")
        logger.log(f"Torch CUDA: {torch.version.cuda}")
        logger.log(f"Device: {device}")
        if device.type == "cuda":
            logger.log(f"GPU: {torch.cuda.get_device_name(device)}")

        logger.log("Loading rna_fm_t12 ...")
        model, alphabet = fm.pretrained.rna_fm_t12()
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()

        model_max_tokens = get_model_max_tokens(model)
        special_token_count = int(bool(alphabet.prepend_bos)) + int(
            bool(alphabet.append_eos)
        )
        max_residues = (
            model_max_tokens - special_token_count
            if model_max_tokens is not None
            else None
        )

        mapping, token_shape = token_mapping(
            alphabet, batch_converter, args.probe_sequence
        )
        logger.log(f"prepend_bos: {alphabet.prepend_bos}")
        logger.log(f"append_eos: {alphabet.append_eos}")
        logger.log(f"model_max_tokens: {model_max_tokens}")
        logger.log(f"inferred_max_residues: {max_residues}")
        logger.log(f"RNA vocabulary: {alphabet.standard_toks}")
        logger.log(f"unk_idx: {alphabet.unk_idx}")
        logger.log(f"Probe token tensor shape: {token_shape}")
        logger.log("Token mapping:")
        for row in mapping:
            logger.log(
                f"  index={row['residue_index']} "
                f"symbol={row['input_symbol']} "
                f"token_id={row['token_id']} "
                f"token={row['token_text']} "
                f"is_unk={row['is_unk']}"
            )

        _, _, small_tokens = batch_converter(
            [("small_forward", args.probe_sequence)]
        )
        small_tokens = small_tokens.to(device)
        with torch.inference_mode():
            small_output = model(
                small_tokens, repr_layers=[args.repr_layer]
            )["representations"][args.repr_layer]
        expected_small_shape = (
            1,
            small_tokens.shape[1],
            int(small_output.shape[-1]),
        )
        if tuple(small_output.shape) != expected_small_shape:
            raise RuntimeError(
                f"小序列输出形状异常：{tuple(small_output.shape)}"
            )
        logger.log(f"Small forward output: {tuple(small_output.shape)}")

        boundary_shape: tuple[int, ...] | None = None
        if not args.skip_boundary_forward:
            if max_residues is None or max_residues < 1:
                raise RuntimeError(
                    "无法从模型读取合法 max_positions；"
                    "可临时使用 --skip-boundary-forward。"
                )
            boundary_sequence = "A" * max_residues
            _, _, boundary_tokens = batch_converter(
                [("boundary_forward", boundary_sequence)]
            )
            if boundary_tokens.shape[1] != model_max_tokens:
                raise RuntimeError(
                    "边界 token 长度异常："
                    f"actual={boundary_tokens.shape[1]}, "
                    f"expected={model_max_tokens}"
                )
            boundary_tokens = boundary_tokens.to(device)
            logger.log(
                "Running boundary forward: "
                f"residues={max_residues}, tokens={boundary_tokens.shape[1]}"
            )
            with torch.inference_mode():
                boundary_output = model(
                    boundary_tokens, repr_layers=[args.repr_layer]
                )["representations"][args.repr_layer]
            boundary_shape = tuple(boundary_output.shape)
            logger.log(f"Boundary forward output: {boundary_shape}")

        result.update(
            {
                "status": "PASS",
                "finished_utc": utc_now().isoformat(),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "model_name": "rna_fm_t12",
                "repr_layer": args.repr_layer,
                "embedding_dim": int(small_output.shape[-1]),
                "prepend_bos": bool(alphabet.prepend_bos),
                "append_eos": bool(alphabet.append_eos),
                "model_max_tokens": model_max_tokens,
                "inferred_max_residues": max_residues,
                "standard_tokens": list(alphabet.standard_toks),
                "unk_idx": int(alphabet.unk_idx),
                "probe_sequence": args.probe_sequence,
                "token_mapping": mapping,
                "small_forward_shape": list(small_output.shape),
                "boundary_forward_shape": (
                    list(boundary_shape) if boundary_shape else None
                ),
            }
        )
        logger.log("FINAL STATUS: PASS")
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "finished_utc": utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        logger.log(f"FINAL STATUS: FAIL: {type(exc).__name__}: {exc}")

    (report_dir / "runtime_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.log(f"Report: {report_dir}")
    return 0 if result["status"] == "PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
