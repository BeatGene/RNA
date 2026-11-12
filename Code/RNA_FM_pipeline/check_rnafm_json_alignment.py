#!/usr/bin/env python3
"""Check current Simple JSON against RNA-FM .pt files; never change inputs.

Standalone: Python >=3.9 and PyTorch with weights_only=True loading support.
No RNA-FM installation, GPU, old audit report or model weights are needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.1-json-embedding-alignment-resume"
HOME_ROOT = Path("/storage9920/home/tinghao.xia")
FILENAME = "rnafm_t12_residue_embeddings.pt"
EXCLUDED = ("176D", "3OK2", "3OK4", "5EME", "5EMF")
ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")
RAW_RE = re.compile(r"^([A-Za-z0-9]{4})\.json$", re.I)
TOKEN_IDS = {symbol: index + 4 for index, symbol in enumerate("ACGURYKMSWBDHVN-")}
PDB_FIELDS = [
    "PDB_ID", "STATUS", "JSON_PATH", "PT_PATH", "JSON_SHA256",
    "JSON_CHAIN_COUNT", "JSON_RESIDUE_COUNT", "PT_CHAIN_COUNT", "PT_RESIDUE_COUNT",
    "EMBEDDING_SHAPE", "SEQUENCES_MATCH", "OFFSETS_MATCH", "RESIDUE_INDICES_MATCH",
    "TOKENS_MATCH", "ALL_ROWS_FINITE_NONZERO", "ISSUES", "WARNINGS", "CHECK_ACTION",
]
CHAIN_FIELDS = [
    "PDB_ID", "CHAIN_INDEX", "JSON_ENTRY_INDEX", "COPY_INDEX", "PROTENIX_CHAIN_ID",
    "JSON_LENGTH", "PT_SEQUENCE_LENGTH", "EXPECTED_START", "EXPECTED_END",
    "PT_START", "PT_END", "EMBEDDING_ROWS", "SEQUENCE_SHA256", "SEQUENCE_MATCH",
    "ORIGINAL_CHAIN_ID", "STATUS", "ISSUES",
]
ISSUE_FIELDS = ["SEVERITY", "PDB_ID", "CODE", "DETAIL"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def chain_id(index):
    result = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def parse_json(path, pdb_id):
    """Mirror generation's whitespace/case normalization and count expansion."""
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8-sig"), object_pairs_hook=no_duplicates,
                         parse_constant=invalid_constant)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("expected exactly one task in a JSON list")
    task = payload[0]
    if not isinstance(task.get("name"), str) or task["name"].strip().upper() != pdb_id:
        raise ValueError("task name does not match PDB ID")
    if not isinstance(task.get("sequences"), list) or not task["sequences"]:
        raise ValueError("missing/empty sequences list")
    chains, warnings = [], []
    for entry_index, entry in enumerate(task["sequences"]):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"invalid sequence entry {entry_index}")
        if "rnaSequence" not in entry:
            if not set(entry) <= {"ligand", "proteinChain", "dnaSequence"}:
                raise ValueError(f"unknown entity type at entry {entry_index}")
            warnings.append(f"NON_RNA_ENTRY_NOT_EMBEDDED:{entry_index}")
            continue
        body = entry["rnaSequence"]
        if not isinstance(body, dict) or not isinstance(body.get("sequence"), str):
            raise ValueError(f"invalid RNA sequence at entry {entry_index}")
        raw = body["sequence"]
        sequence = re.sub(r"\s+", "", raw).upper()
        if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
            raise ValueError(f"invalid RNA characters at entry {entry_index}")
        if raw != sequence:
            warnings.append(f"NORMALIZED_CASE_OR_WHITESPACE:{entry_index}")
        count = body.get("count", 1)
        if type(count) is not int or count < 1:
            raise ValueError(f"count must be a positive integer at entry {entry_index}")
        for copy in range(1, count + 1):
            chains.append({"sequence": sequence, "entry": entry_index, "copy": copy})
    if not chains:
        raise ValueError("no RNA chains")
    return chains, hashlib.sha256(content).hexdigest(), warnings


def inventory(json_root, embedding_root, excluded):
    raw, outputs = defaultdict(list), defaultdict(list)
    for path in sorted(json_root.iterdir()):
        match = RAW_RE.fullmatch(path.name)
        if match and (path.is_file() or path.is_symlink()):
            raw[match[1].upper()].append(path)
    for path in sorted(embedding_root.iterdir()):
        if ID_RE.fullmatch(path.name) and (path.is_dir() or path.is_symlink()):
            outputs[path.name.upper()].append(path)
    targets = set(raw) - set(excluded)
    return raw, outputs, targets


def signature(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def check_target(pdb_id, json_paths, output_dirs, is_target, torch):
    row = dict.fromkeys(PDB_FIELDS, "")
    row["PDB_ID"] = pdb_id
    issues, chain_rows, warnings = [], [], []

    def issue(code, detail=""):
        issues.append({"SEVERITY": "ERROR", "PDB_ID": pdb_id, "CODE": code,
                       "DETAIL": str(detail)})

    if not is_target:
        issue("EXTRA_EMBEDDING_TARGET", "No selected raw JSON (or ID is explicitly excluded)")
    if len(json_paths) != 1:
        issue("RAW_JSON_COUNT", f"expected=1, found={len(json_paths)}")
    if len(output_dirs) != 1:
        issue("EMBEDDING_DIRECTORY_COUNT", f"expected=1, found={len(output_dirs)}")
    if json_paths:
        row["JSON_PATH"] = str(json_paths[0])
    if output_dirs:
        row["PT_PATH"] = str(output_dirs[0] / FILENAME)

    chains = None
    before_json = None
    if len(json_paths) == 1:
        try:
            before_json = signature(json_paths[0])
            chains, row["JSON_SHA256"], warnings = parse_json(json_paths[0], pdb_id)
            row["JSON_CHAIN_COUNT"] = len(chains)
            row["JSON_RESIDUE_COUNT"] = sum(len(c["sequence"]) for c in chains)
        except Exception as error:
            issue("INVALID_RAW_JSON", f"{type(error).__name__}: {error}")

    payload = None
    before_pt = None
    if len(output_dirs) == 1:
        path = output_dirs[0] / FILENAME
        try:
            # One file at a time, on CPU; accept only tensor/basic-type payloads.
            before_pt = signature(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("PT payload must be a dictionary")
        except Exception as error:
            issue("PT_LOAD_FAILED", f"{type(error).__name__}: {error}")
            payload = None
        extras = sorted(str(p) for p in output_dirs[0].rglob("*.pt") if p != path)
        if extras:
            issue("EXTRA_PT_FILES", ";".join(extras))

    if chains is not None and payload is not None:
        try:
            sequences = [c["sequence"] for c in chains]
            lengths = list(map(len, sequences))
            total, count = sum(lengths), len(chains)
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            row["PT_CHAIN_COUNT"] = payload.get("chain_count", "")
            row["PT_RESIDUE_COUNT"] = payload.get("residue_count", "")
            for key, expected in {
                "schema_version": 1, "pdb_id": pdb_id, "model_name": "rna_fm_t12",
                "repr_layer": 12, "embedding_dim": 640, "chain_count": count,
                "residue_count": total, "sequence_sha256": [digest(s) for s in sequences],
                "json_entry_indices": [c["entry"] for c in chains],
                "copy_indices": [c["copy"] for c in chains],
                "expected_protenix_chain_ids": [chain_id(i) for i in range(count)],
                "unk_token_id": 3,
            }.items():
                if payload.get(key) != expected:
                    issue("METADATA_MISMATCH", key)

            actual_sequences = payload.get("sequences")
            row["SEQUENCES_MATCH"] = actual_sequences == sequences
            if not row["SEQUENCES_MATCH"]:
                issue("SEQUENCE_OR_CHAIN_ORDER_MISMATCH")
            actual_offsets = payload.get("chain_offsets")
            offsets_ok = (torch.is_tensor(actual_offsets)
                          and actual_offsets.dtype == torch.int64
                          and tuple(actual_offsets.shape) == (count + 1,)
                          and actual_offsets.tolist() == offsets)
            row["OFFSETS_MATCH"] = offsets_ok
            if not offsets_ok:
                issue("CHAIN_OFFSETS_MISMATCH")
            offset_values = (actual_offsets.tolist() if torch.is_tensor(actual_offsets)
                             and actual_offsets.ndim == 1 else [])

            embedding = payload.get("residue_embedding")
            embedding_ok = (torch.is_tensor(embedding) and embedding.ndim == 2
                            and tuple(embedding.shape) == (total, 640)
                            and embedding.dtype in {torch.float32, torch.float16})
            row["EMBEDDING_SHAPE"] = (str(list(embedding.shape))
                                       if torch.is_tensor(embedding) else "NOT_A_TENSOR")
            if not embedding_ok:
                issue("EMBEDDING_SHAPE_OR_DTYPE_MISMATCH", row["EMBEDDING_SHAPE"])
            elif payload.get("embedding_dtype") != str(embedding.dtype).split(".")[-1]:
                issue("EMBEDDING_DTYPE_METADATA_MISMATCH")

            expected_chain = torch.repeat_interleave(torch.arange(count), torch.tensor(lengths))
            expected_index = torch.cat([torch.arange(n) for n in lengths])
            flat_sequence = "".join(sequences)
            expected_tokens = torch.tensor([TOKEN_IDS.get(s, 3) for s in flat_sequence], dtype=torch.int64)
            expected_non_acgu = torch.tensor([s not in "ACGU" for s in flat_sequence], dtype=torch.bool)
            expected_tensors = {
                "residue_chain_index": expected_chain, "residue_index_in_chain": expected_index,
                "residue_token_id": expected_tokens, "residue_non_acgu": expected_non_acgu,
                "residue_is_unk": expected_tokens.eq(3),
            }
            tensor_ok = {}
            for key, expected in expected_tensors.items():
                actual = payload.get(key)
                passed = (torch.is_tensor(actual) and actual.dtype == expected.dtype
                          and tuple(actual.shape) == (total,) and torch.equal(actual, expected))
                tensor_ok[key] = passed
                if not passed:
                    issue("RESIDUE_ARRAY_MISMATCH", key)
            row["RESIDUE_INDICES_MATCH"] = (tensor_ok["residue_chain_index"]
                                            and tensor_ok["residue_index_in_chain"])
            row["TOKENS_MATCH"] = all(tensor_ok[k] for k in
                                      ("residue_token_id", "residue_non_acgu", "residue_is_unk"))
            policy = payload.get("token_policy", {})
            if not isinstance(policy, dict):
                policy = {}
            for symbol in sorted(set(flat_sequence)):
                expected = TOKEN_IDS.get(symbol, 3)
                entry = policy.get(symbol)
                if (not isinstance(entry, dict) or entry.get("token_id") != expected
                        or entry.get("is_unk") != (expected == 3)
                        or entry.get("token_text") != ("<unk>" if expected == 3 else symbol)):
                    issue("TOKEN_POLICY_MISMATCH", symbol)

            bad_rows = set()
            if embedding_ok:
                for start in range(0, total, 4096):
                    block = embedding[start:start + 4096]
                    valid = torch.isfinite(block).all(dim=1) & block.ne(0).any(dim=1)
                    bad_rows.update(start + i for i in torch.nonzero(~valid).flatten().tolist())
                if bad_rows:
                    issue("NONFINITE_OR_ZERO_EMBEDDING_ROWS", str(sorted(bad_rows)[:30]))
            row["ALL_ROWS_FINITE_NONZERO"] = embedding_ok and not bad_rows

            original_ids = payload.get("original_chain_ids", [])
            for index, chain in enumerate(chains):
                start, end = offsets[index:index + 2]
                actual = (actual_sequences[index] if isinstance(actual_sequences, list)
                          and index < len(actual_sequences) else None)
                sequence_match = actual == chain["sequence"]
                actual_start = offset_values[index] if index < len(offset_values) else ""
                actual_end = offset_values[index + 1] if index + 1 < len(offset_values) else ""
                chain_issues = []
                if not sequence_match:
                    chain_issues.append("SEQUENCE_MISMATCH")
                if actual_start != start or actual_end != end:
                    chain_issues.append("OFFSET_MISMATCH")
                if not embedding_ok or (embedding_ok and embedding[start:end].shape[0] != end - start):
                    chain_issues.append("EMBEDDING_ROW_COUNT_MISMATCH")
                if any(start <= value < end for value in bad_rows):
                    chain_issues.append("NONFINITE_OR_ZERO_ROWS")
                if not row["RESIDUE_INDICES_MATCH"] or not row["TOKENS_MATCH"]:
                    chain_issues.append("PDB_RESIDUE_ARRAY_MISMATCH")
                chain_rows.append({
                    "PDB_ID": pdb_id, "CHAIN_INDEX": index, "JSON_ENTRY_INDEX": chain["entry"],
                    "COPY_INDEX": chain["copy"], "PROTENIX_CHAIN_ID": chain_id(index),
                    "JSON_LENGTH": end - start,
                    "PT_SEQUENCE_LENGTH": len(actual) if isinstance(actual, str) else "",
                    "EXPECTED_START": start, "EXPECTED_END": end,
                    "PT_START": actual_start, "PT_END": actual_end,
                    "EMBEDDING_ROWS": (embedding[start:end].shape[0] if embedding_ok else ""),
                    "SEQUENCE_SHA256": digest(chain["sequence"]), "SEQUENCE_MATCH": sequence_match,
                    "ORIGINAL_CHAIN_ID": (original_ids[index] if isinstance(original_ids, list)
                                          and index < len(original_ids) else ""),
                    "STATUS": "FAIL" if chain_issues else "PASS", "ISSUES": ";".join(chain_issues),
                })
        except Exception as error:
            issue("PAYLOAD_CHECK_FAILED", f"{type(error).__name__}: {error}")

    for path, before in [(json_paths[0] if json_paths else None, before_json),
                         (Path(row["PT_PATH"]) if row["PT_PATH"] else None, before_pt)]:
        if before is not None:
            try:
                if signature(path) != before:
                    issue("INPUT_CHANGED_DURING_CHECK", path)
            except OSError as error:
                issue("INPUT_CHANGED_DURING_CHECK", error)
    row.update(STATUS="FAIL" if issues else "PASS",
               ISSUES=";".join(sorted({i["CODE"] for i in issues})), WARNINGS=";".join(warnings))
    return row, chain_rows, issues


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def report_lock(report):
    """Prevent concurrent updated checkers from replacing the same reports."""
    import fcntl
    with (report / ".run.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def input_fingerprint(json_paths, output_dirs):
    def describe(path):
        return {"path": str(path), "resolved": str(path.resolve(strict=True)),
                "stat": list(signature(path))}
    return {
        "json": [describe(p) for p in json_paths],
        "output": [{"directory": describe(d),
                    "pt_files": [describe(p) for p in sorted(d.rglob("*.pt"))]}
                   for d in output_dirs],
    }


def load_checkpoint(path, fingerprint, validation_signature):
    """Only reuse a complete PASS produced by the same checker and environment."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["validation_signature"] == validation_signature
                and data["fingerprint"] == fingerprint
                and data["row"]["STATUS"] == "PASS" and not data["issues"]
                and len(data["chains"]) == int(data["row"]["JSON_CHAIN_COUNT"])
                and all(row["STATUS"] == "PASS" for row in data["chains"])
                and data["row"]["PDB_ID"] == path.stem):
            return data["row"], data["chains"], data["issues"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare_resume(report, args, torch):
    config = {"json_dir": str(args.json_dir), "embedding_root": str(args.embedding_root),
              "expected_count": args.expected_count, "excluded_ids": sorted(set(x.upper() for x in args.exclude))}
    config_path = report / "config.json"
    backup = ""
    if args.resume:
        if config_path.is_file():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if old != config:
                raise ValueError("resume input paths, expected count or exclusion list differ; use the original arguments")
        else:
            # Version 1.0 reports have no per-file fingerprints. Never infer a
            # reusable PT check from an old PASS TSV row alone.
            old = json.loads((report / "summary.json").read_text(encoding="utf-8"))
            if (old.get("script_version") != "1.0-json-embedding-alignment"
                    or old.get("json_dir") != config["json_dir"]
                    or old.get("embedding_root") != config["embedding_root"]
                    or old.get("expected_target_count") != config["expected_count"]
                    or sorted(old.get("excluded_ids", [])) != config["excluded_ids"]):
                raise ValueError("not a compatible version 1.0 report; choose a new report directory")
        folder = report / "previous_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder.mkdir(parents=True)
        for name in ("summary.json", "per_pdb.tsv", "per_chain.tsv", "issues.tsv"):
            if (report / name).is_file():
                shutil.copy2(report / name, folder / name)
        backup = str(folder)
    atomic_json(config_path, config)
    (report / "checkpoints").mkdir(exist_ok=True)
    validation_signature = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "torch_version": str(torch.__version__)}
    return backup, validation_signature


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-dir", type=Path, default=HOME_ROOT / "Json_data/Simple_json")
    parser.add_argument("--embedding-root", type=Path, default=HOME_ROOT / "Data_FM/RNA_FM_embeddings")
    parser.add_argument("--expected-count", type=int, default=2241)
    parser.add_argument("--exclude", nargs="*", default=list(EXCLUDED), metavar="PDB_ID",
                        help="Excluded raw IDs; default: five legacy-only IDs. --exclude alone clears the list.")
    parser.add_argument("--report-dir", type=Path,
                        help="Report outside both input roots; new unless --resume. Default: timestamped directory under HOME.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume --report-dir; reuse unchanged PASS checkpoints, recheck failures and changes.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("json_dir", "embedding_root"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_dir():
            parser.error(f"input directory does not exist: {value}")
        setattr(args, name, value)
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    if any(not ID_RE.fullmatch(x) for x in args.exclude):
        parser.error("--exclude accepts four-character PDB IDs only")
    excluded = {x.upper() for x in args.exclude}
    report = (args.report_dir or HOME_ROOT / ("RNA_FM_JSON_CHECK_" +
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).expanduser().resolve()
    if any(source == report or source in report.parents for source in (args.json_dir, args.embedding_root)):
        parser.error("report directory must be outside the JSON and embedding roots")
    if args.resume and (args.report_dir is None or not report.is_dir()):
        parser.error("--resume requires --report-dir pointing to an existing report")
    if report.exists() and not args.resume:
        parser.error("report directory already exists; choose a new --report-dir")
    try:
        import torch
    except ImportError:
        parser.error("PyTorch is required; run this script in the existing rna_fm environment")
    report.mkdir(parents=True, exist_ok=True)
    try:
        with report_lock(report):
            backup, validation_signature = prepare_resume(report, args, torch)
            return run_check(args, report, torch, backup, validation_signature)
    except BlockingIOError:
        parser.error("another checker is already using this report directory")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


def run_check(args, report, torch, backup, validation_signature):
    torch.set_num_threads(1)
    excluded = {x.upper() for x in args.exclude}
    raw, outputs, targets = inventory(args.json_dir, args.embedding_root, excluded)
    all_ids = sorted(targets | set(outputs))
    logger = logging.getLogger(VERSION)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(report / "check.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    logger.info("REPORT_DIR=%s", report)
    if args.resume:
        logger.info("RESUME: unchanged PASS checkpoints will be reused; old reports backed up to %s", backup)
        logger.info("Version 1.0 results without fingerprints must be checked again once")
    logger.info("raw=%d selected=%d embedding_IDs=%d expected=%d", len(raw), len(targets), len(outputs), args.expected_count)
    summary = {
        "script_version": VERSION, "status": "RUNNING", "started_utc": now(),
        "json_dir": str(args.json_dir), "embedding_root": str(args.embedding_root),
        "report_dir": str(report), "torch_version": torch.__version__,
        "expected_target_count": args.expected_count, "excluded_ids": sorted(excluded),
        "excluded_raw_ids_found": sorted(set(raw) & excluded), "raw_json_id_count": len(raw),
        "target_json_id_count": len(targets), "embedding_id_count": len(outputs),
        "missing_embedding_ids": sorted(targets - set(outputs)),
        "extra_embedding_ids": sorted(set(outputs) - targets),
        "checked_pdbs": 0, "pass_pdbs": 0, "failed_pdbs": 0,
        "json_chain_count": 0, "json_residue_count": 0, "issue_count": 0,
        "issue_counts": {}, "all_targets_passed": False,
        "resume": args.resume, "previous_report_backup": backup,
        "reused_pdbs": 0, "rechecked_pdbs": 0,
        "scope": "Current raw RNA sequence/count/order versus PT metadata, offsets, residue arrays and finite nonzero vectors. "
                 "Does not rerun RNA-FM or independently recompute embedding values; does not certify coordinate/modified-chemistry mapping.",
    }
    counts = Counter()
    fatal, interrupted = "", False
    with (report / "per_pdb.tsv").open("w", encoding="utf-8-sig", newline="") as pf, \
         (report / "per_chain.tsv").open("w", encoding="utf-8-sig", newline="") as cf, \
         (report / "issues.tsv").open("w", encoding="utf-8-sig", newline="") as ef:
        writers = [csv.DictWriter(f, fieldnames=fields, delimiter="\t") for f, fields in
                   ((pf, PDB_FIELDS), (cf, CHAIN_FIELDS), (ef, ISSUE_FIELDS))]
        for writer in writers:
            writer.writeheader()
        if len(targets) != args.expected_count:
            writers[2].writerow({"SEVERITY": "ERROR", "PDB_ID": "", "CODE": "TARGET_COUNT_MISMATCH",
                                "DETAIL": f"expected={args.expected_count}, found={len(targets)}"})
            counts["TARGET_COUNT_MISMATCH"] += 1
        atomic_json(report / "summary.json", summary)
        try:
            for number, pdb_id in enumerate(all_ids, 1):
                json_paths, output_dirs = raw.get(pdb_id, []), outputs.get(pdb_id, [])
                checkpoint = report / "checkpoints" / (pdb_id + ".json")
                try:
                    before = input_fingerprint(json_paths, output_dirs)
                except OSError:
                    before = None
                cached = (load_checkpoint(checkpoint, before, validation_signature)
                          if args.resume and before is not None and pdb_id in targets
                          and len(json_paths) == len(output_dirs) == 1 else None)
                if cached is not None:
                    # Recheck metadata after reading the checkpoint as well.
                    try:
                        if input_fingerprint(json_paths, output_dirs) != before:
                            cached = None
                    except OSError:
                        cached = None
                if cached is not None:
                    row, chain_rows, issues = cached
                    row["CHECK_ACTION"] = "REUSED_PASS"
                    summary["reused_pdbs"] += 1
                else:
                    row, chain_rows, issues = check_target(pdb_id, json_paths, output_dirs,
                                                          pdb_id in targets, torch)
                    row["CHECK_ACTION"] = "CHECKED"
                    summary["rechecked_pdbs"] += 1
                    try:
                        after = input_fingerprint(json_paths, output_dirs)
                    except OSError:
                        after = None
                    if before != after and row["STATUS"] == "PASS":
                        issues.append({"SEVERITY": "ERROR", "PDB_ID": pdb_id,
                                       "CODE": "INPUT_CHANGED_DURING_CHECK", "DETAIL": "checkpoint fingerprint changed"})
                        row.update(STATUS="FAIL", ISSUES="INPUT_CHANGED_DURING_CHECK")
                    atomic_json(checkpoint, {"validation_signature": validation_signature,
                                            "fingerprint": after, "row": row,
                                            "chains": chain_rows, "issues": issues})
                writers[0].writerow(row)
                writers[1].writerows(chain_rows)
                writers[2].writerows(issues)
                for handle in (pf, cf, ef):
                    handle.flush()
                counts.update(i["CODE"] for i in issues)
                summary["checked_pdbs"] += 1
                summary["pass_pdbs" if row["STATUS"] == "PASS" else "failed_pdbs"] += 1
                summary["json_chain_count"] += int(row["JSON_CHAIN_COUNT"] or 0)
                summary["json_residue_count"] += int(row["JSON_RESIDUE_COUNT"] or 0)
                summary.update(issue_count=sum(counts.values()), issue_counts=dict(counts), updated_utc=now())
                atomic_json(report / "summary.json", summary)
                logger.info("[%d/%d] %s %s %s chains=%s residues=%s %s", number, len(all_ids), pdb_id,
                            row["STATUS"], row["CHECK_ACTION"], row["JSON_CHAIN_COUNT"], row["JSON_RESIDUE_COUNT"], row["ISSUES"])
        except KeyboardInterrupt:
            interrupted = True
        except Exception as error:
            fatal = f"{type(error).__name__}: {error}"
            logger.exception("Check stopped")
    passed = (not fatal and not interrupted and len(targets) == args.expected_count
              and summary["checked_pdbs"] == len(all_ids) and summary["pass_pdbs"] == len(targets)
              and not counts and not summary["extra_embedding_ids"])
    summary.update(status="INTERRUPTED" if interrupted else "FAILED" if fatal else "PASS" if passed else "FAIL",
                   all_targets_passed=passed, finished_utc=now(), fatal_error=fatal,
                   issue_count=sum(counts.values()), issue_counts=dict(counts))
    atomic_json(report / "summary.json", summary)
    logger.info("FINAL STATUS=%s all_targets_passed=%s checked=%d pass=%d failed=%d issues=%d",
                summary["status"], passed, summary["checked_pdbs"], summary["pass_pdbs"],
                summary["failed_pdbs"], summary["issue_count"])
    logger.info("REPORT_DIR=%s", report)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    return 130 if interrupted else 2 if fatal else 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
