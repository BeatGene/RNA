"""Plan and apply a local-authoritative sync of Code and Code_Flow_matching.

The archives contain only inventory-listed files. Server-only files are
downloaded first; local files that are new or differ are then sent to the
server. Existing server files are backed up before they are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path


MANIFEST_MEMBER = "SYNC_MANIFEST.jsonl"
PREFIXES = ("code/", "flow/", "reports/")


def check_key(key: str) -> str:
    if (not isinstance(key, str) or not key.startswith(PREFIXES)
            or "\\" in key or "\x00" in key or key.startswith("/")
            or any(part in ("", ".", "..") for part in key.split("/"))):
        raise ValueError(f"Unsafe logical key: {key!r}")
    return key


def read_manifest(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            key = check_key(item["key"])
            if key in result or item["size"] < 0 or len(item["sha256"]) != 64:
                raise ValueError(f"Invalid or duplicate manifest entry: {key}")
            result[key] = item
    return result


def read_keys(path: Path) -> list[str]:
    keys = [check_key(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate key in {path}")
    return keys


def path_for(root: Path, side: str, key: str) -> Path:
    check_key(key)
    prefix, remainder = key.split("/", 1)
    if prefix == "code":
        base = root / "Code"
    elif prefix == "flow":
        base = root / "Code_Flow_matching"
    else:
        base = root / "Code" / ("New_Data_pipeline_reports" if side == "local" else "pipeline_reports")
    return base.joinpath(*remainder.split("/"))


def io_path(path: Path) -> str:
    resolved = os.path.abspath(path)
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(io_path(path), "rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plan(args):
    local = read_manifest(args.local_manifest)
    server = read_manifest(args.server_manifest)
    local_keys, server_keys = set(local), set(server)
    server_only = sorted(server_keys - local_keys)
    local_only = sorted(local_keys - server_keys)
    different = sorted(key for key in local_keys & server_keys
                       if local[key]["sha256"] != server[key]["sha256"])
    same = len(local_keys & server_keys) - len(different)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, keys in (("server_only", server_only), ("local_only", local_only),
                       ("different_local_wins", different),
                       ("local_push", sorted(local_only + different))):
        (args.output_dir / f"{name}.txt").write_text(
            "".join(key + "\n" for key in keys), encoding="utf-8")
    summary = {
        "local_files": len(local), "server_files": len(server),
        "same": same, "server_only": len(server_only),
        "local_only": len(local_only), "different_local_wins": len(different),
        "server_only_bytes": sum(server[key]["size"] for key in server_only),
        "local_push_bytes": sum(local[key]["size"] for key in local_only + different),
        "report_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        return data


def pack(args):
    manifest = read_manifest(args.manifest)
    keys = read_keys(args.keys)
    selected = [manifest[key] for key in keys]
    payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in selected).encode()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".part")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        with tarfile.open(temp, "w:gz", compresslevel=1) as archive:
            info = tarfile.TarInfo(MANIFEST_MEMBER)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for index, item in enumerate(selected, 1):
                path = path_for(args.root, args.side, item["key"])
                if os.path.islink(io_path(path)) or not os.path.isfile(io_path(path)):
                    raise ValueError(f"Source changed or is a symlink: {path}")
                info = tarfile.TarInfo(item["key"])
                info.size = item["size"]
                with open(io_path(path), "rb") as source:
                    hashed = HashingReader(source)
                    archive.addfile(info, hashed)
                    if hashed.digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"Source changed since inventory: {path}")
                if index % 100 == 0 or index == len(selected):
                    print(f"PACK {index}/{len(selected)}", flush=True)
        temp.replace(args.output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"ARCHIVE {args.output} files={len(selected)} bytes={args.output.stat().st_size}")


def apply(args):
    if args.mode == "local_authoritative" and args.backup_dir is None:
        raise ValueError("--backup-dir is required for local_authoritative mode")
    with tarfile.open(args.archive, "r:gz") as archive:
        first = archive.next()
        if first is None or first.name != MANIFEST_MEMBER or not first.isfile():
            raise ValueError("Archive lacks an initial manifest")
        expected = {
            item["key"]: item
            for item in (json.loads(line) for line in archive.extractfile(first))
        }
        if len(expected) == 0:
            print("APPLY archive has zero files")
        seen = set()
        written = skipped = backed_up = 0
        while member := archive.next():
            key = check_key(member.name)
            if key not in expected or key in seen or not member.isfile():
                raise ValueError(f"Unexpected archive member: {key}")
            seen.add(key)
            dest = path_for(args.root, args.side, key)
            root_for_key = path_for(args.root, args.side, key.split("/", 1)[0] + "/probe").parent
            if not dest.parent.resolve().is_relative_to(root_for_key.resolve()):
                raise ValueError(f"Destination escapes root: {dest}")
            Path(io_path(dest.parent)).mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".sync_part")
            if os.path.exists(io_path(temp)):
                raise FileExistsError(temp)
            digest = hashlib.sha256()
            with open(io_path(temp), "wb") as output:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read {key}")
                while chunk := source.read(4 * 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            item = expected[key]
            if os.stat(io_path(temp)).st_size != item["size"] or digest.hexdigest() != item["sha256"]:
                Path(io_path(temp)).unlink(missing_ok=True)
                raise ValueError(f"Archive checksum mismatch: {key}")
            if os.path.exists(io_path(dest)):
                if os.path.islink(io_path(dest)):
                    Path(io_path(temp)).unlink(missing_ok=True)
                    raise ValueError(f"Refusing to overwrite symlink: {dest}")
                if sha256_file(dest) == item["sha256"]:
                    Path(io_path(temp)).unlink()
                    skipped += 1
                    continue
                if args.mode == "server_only":
                    Path(io_path(temp)).unlink()
                    raise FileExistsError(f"Existing local file differs: {dest}")
                backup = args.backup_dir / key
                Path(io_path(backup.parent)).mkdir(parents=True, exist_ok=True)
                if os.path.exists(io_path(backup)):
                    Path(io_path(temp)).unlink()
                    raise FileExistsError(f"Backup already exists: {backup}")
                shutil.copy2(io_path(dest), io_path(backup))
                backed_up += 1
            os.replace(io_path(temp), io_path(dest))
            written += 1
            if (written + skipped) % 100 == 0:
                print(f"APPLY {written + skipped}/{len(expected)}", flush=True)
        if seen != set(expected):
            raise ValueError(f"Archive missing {len(set(expected) - seen)} listed files")
    print(json.dumps({"written": written, "already_same": skipped,
                      "backed_up": backed_up}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--local-manifest", type=Path, required=True)
    p.add_argument("--server-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(function=plan)
    p = commands.add_parser("pack")
    p.add_argument("--side", choices=("local", "server"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=pack)
    p = commands.add_parser("apply")
    p.add_argument("--side", choices=("local", "server"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--mode", choices=("server_only", "local_authoritative"), required=True)
    p.add_argument("--backup-dir", type=Path)
    p.set_defaults(function=apply)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
