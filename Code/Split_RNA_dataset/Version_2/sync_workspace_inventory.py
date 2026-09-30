"""Inventory the two code trees with a stable reports-directory mapping.

Run on either the local Windows workspace or the laboratory server. The
logical ``reports/`` tree maps to ``Code/New_Data_pipeline_reports`` locally
and ``Code/pipeline_reports`` on the server. This script only reads source
files; it never copies or overwrites them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def io_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def inventory_tree(root: Path, prefix: str, excluded_dir: str, output):
    scanned = 0
    total_bytes = 0
    symlinks = []
    root_io = io_path(root)
    if not os.path.isdir(root_io):
        raise FileNotFoundError(root)
    stack = [root_io]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            relative = os.path.relpath(entry.path, root_io).replace("\\", "/")
            if entry.is_symlink():
                symlinks.append(prefix + relative)
                continue
            if entry.is_dir(follow_symlinks=False):
                if directory == root_io and entry.name == excluded_dir:
                    continue
                if prefix == "reports/" and entry.name.startswith("CODE_SYNC_"):
                    continue
                stack.append(entry.path)
            elif entry.is_file(follow_symlinks=False):
                digest = hashlib.sha256()
                with open(entry.path, "rb") as source:
                    while chunk := source.read(4 * 1024 * 1024):
                        digest.update(chunk)
                size = entry.stat(follow_symlinks=False).st_size
                output.write(json.dumps(
                    {"key": prefix + relative, "size": size, "sha256": digest.hexdigest()},
                    ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                scanned += 1
                total_bytes += size
                if scanned % 1000 == 0:
                    print(f"{prefix} files={scanned} GiB={total_bytes / 2**30:.3f}", flush=True)
            else:
                raise ValueError(f"Unsupported filesystem entry: {entry.path}")
    return {"files": scanned, "bytes": total_bytes, "symlinks": symlinks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("local", "server"), required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Local RNA workspace root or server home directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = args.root / "Code"
    flow = args.root / "Code_Flow_matching"
    reports = code / ("New_Data_pipeline_reports" if args.side == "local" else "pipeline_reports")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temp = args.output.with_name(args.output.name + ".part")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as output:
            stats = {
                "code": inventory_tree(code, "code/", reports.name, output),
                "flow": inventory_tree(flow, "flow/", "", output),
                "reports": inventory_tree(reports, "reports/", "", output),
            }
        temp.replace(args.output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    summary = {"side": args.side, "root": str(args.root.resolve()),
               "manifest": str(args.output.resolve()), "trees": stats}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if any(tree["symlinks"] for tree in stats.values()):
        print("WARNING: symlinks were listed in summary but not inventoried", file=sys.stderr)


if __name__ == "__main__":
    main()
