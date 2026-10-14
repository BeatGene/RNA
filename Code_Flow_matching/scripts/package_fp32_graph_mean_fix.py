"""Build and verify the narrow graph-mean repair upload, with current mtimes."""
import argparse
import ast
import hashlib
import io
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "etflow/commons/reductions.py", "etflow/models/utils.py", "etflow/models/model.py",
    "etflow/models/loss.py", "etflow/networks/torchmd_net/model_dynamics.py",
    "etflow/networks/torchmd_net/modules.py",
    "scripts/check_bf16_graph_mean.py", "scripts/probe_rank1_mobility_gradients.py",
    "scripts/test_fp32_graph_reductions.py", "scripts/test_probe_rank1_mobility_gradients.py",
    "scripts/test_masked_rms.py", "scripts/test_vector_output_amp.py", "scripts/test_plane_loss_amp.py",
    "scripts/run_fp32_graph_mean_checks.sh", "scripts/FP32_GRAPH_MEAN_FIX.md",
    "scripts/package_fp32_graph_mean_fix.py", "scripts/smoke_test_synthetic.py",
]


def build_archive(output):
    contents = []
    for name in FILES:
        data = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
        contents.append((name, data))
    manifest = "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name, data in contents)
    (ROOT / "scripts/FP32_GRAPH_MEAN_SHA256SUMS").write_text(manifest, encoding="utf-8", newline="\n")
    contents.append(("scripts/FP32_GRAPH_MEAN_SHA256SUMS", manifest.encode("utf-8")))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, data in contents:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(time.time())
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == [name for name, _ in contents]
        for name, data in contents:
            assert archive.extractfile(name).read() == data
    print("UPLOAD_ARCHIVE_READY", output, "FILES", len(contents), "BYTES", output.stat().st_size)
    print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / "deliverables/rna_fp32_graph_mean_fix_20261013_v2.tar.gz")
    build_archive(parser.parse_args().output)
