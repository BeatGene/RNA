#!/usr/bin/env python3
"""Preview or install the scoped RNA pipeline V2 update archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def checked_files(archive: ZipFile) -> list[tuple[str, bytes, str]]:
    names = set(archive.namelist())
    lines = archive.read("SHA256SUMS.txt").decode("utf-8").splitlines()
    result = []
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.parts[0] not in {"Code", "Code_Flow_matching"}
                or relative not in names):
            raise ValueError(f"unsafe or missing archive path: {relative}")
        payload = archive.read(relative)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {relative}")
        result.append((relative, payload, digest))
    if len(result) != len({item[0] for item in result}):
        raise ValueError("duplicate files in SHA256SUMS.txt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target-root", type=Path, default=Path.home())
    parser.add_argument("--apply", action="store_true", help="install after preview")
    args = parser.parse_args()
    root = args.target_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    with ZipFile(args.archive.expanduser()) as archive:
        files = checked_files(archive)
    plan = []
    for relative, payload, digest in files:
        destination = root.joinpath(*PurePosixPath(relative).parts)
        resolved = destination.resolve(strict=False)
        if root not in resolved.parents:
            raise ValueError(f"destination escapes target root: {destination}")
        if destination.is_symlink():
            raise ValueError(f"refusing to overwrite symlink: {destination}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"destination is not a file: {destination}")
        status = "same" if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest else "replace" if destination.exists() else "new"
        plan.append((relative, payload, destination, status))
        print(f"{status:7s} {relative}")
    if not args.apply:
        print("PREVIEW_ONLY: no files changed")
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = root / "Code" / "pipeline_reports" / f"PIPELINE_V2_DEPLOY_{timestamp}"
    report.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, payload, destination, status in plan:
        if status == "same":
            records.append({"path": relative, "action": "same"})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = report / "backup" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        try:
            os.chmod(
                temporary,
                stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644,
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        records.append({"path": relative, "action": status})
    (report / "install_manifest.json").write_text(
        json.dumps({"archive": str(args.archive.expanduser().resolve()),
                    "target_root": str(root), "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"INSTALLED report={report}")


if __name__ == "__main__":
    main()
