"""Create a small reproducible server upload archive; exclude data and secrets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
from pathlib import Path
import tarfile
import time


def build_archive(output, fix_only=False):
    project = Path(__file__).resolve().parents[1]
    script_names = (
        "evaluate_refinement.py", "evaluate_refinement_v3.py", "export_rank1_pymol_v3.py",
        "build_rank1_pymol_diagnostics.py", "rna_pymol_viewer.py",
        "repack_rank1_pymol_bundle.py",
        "run_rank1_pymol_diagnostics.sh", "RANK1_PYMOL_DIAGNOSTICS.md",
        "test_rank1_pymol_diagnostics.py", "package_rank1_pymol_diagnostics.py",
    )
    if fix_only:
        script_names = ("build_rank1_pymol_diagnostics.py", "repack_rank1_pymol_bundle.py",
                        "run_rank1_pymol_diagnostics.sh", "RANK1_PYMOL_DIAGNOSTICS.md",
                        "package_rank1_pymol_diagnostics.py")
    files = [project / "scripts" / name for name in script_names]
    if not fix_only:
        files += [project / "config" / "RNA_train_residual_v3.yaml"]
        files += sorted(path for path in (project / "etflow").rglob("*.py")
                        if not path.name.startswith("test_"))
    contents = []
    for path in files:
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        if path.suffix == ".py":
            ast.parse(data.decode("utf-8"), filename=str(path))
        contents.append((path.relative_to(project).as_posix(), data))
    manifest = "\n".join(hashlib.sha256(data).hexdigest() + "  " + name for name, data in contents) + "\n"
    contents.append(("scripts/RANK1_PYMOL_UPLOAD_SHA256SUMS", manifest.encode("utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for name, data in contents:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = int(time.time())
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output, "r:gz") as archive:
        assert sorted(archive.getnames()) == sorted(name for name, _ in contents)
        for name, data in contents:
            assert archive.extractfile(name).read() == data
    print(f"UPLOAD_ARCHIVE_READY files={len(contents)} bytes={output.stat().st_size} path={output}")
    print(f"SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] /
                        "deliverables/rank1_pymol_diagnostics_20261013.tar.gz")
    parser.add_argument("--fix-only", action="store_true", help="ship only the ZIP timestamp fix and recovery command")
    args = parser.parse_args()
    build_archive(args.output.resolve(), args.fix_only)
