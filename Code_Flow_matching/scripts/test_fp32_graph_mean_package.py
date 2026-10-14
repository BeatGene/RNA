"""Catch omitted runner dependencies in the server upload archive."""
import re
import tarfile
import tempfile
import unittest
from pathlib import Path

import package_fp32_graph_mean_fix as package


class PackageTest(unittest.TestCase):
    def test_runner_required_scripts_are_packaged(self):
        runner = (package.ROOT / "scripts/run_fp32_graph_mean_checks.sh").read_text(encoding="utf-8")
        # Optional full evaluation uses the evaluator already on the server.
        required = set(re.findall(r"python (scripts/[a-zA-Z0-9_]+\.py)", runner))
        required.discard("scripts/evaluate_refinement_v3.py")
        for loop in re.findall(r"for test_name in ([^;]+); do", runner):
            required.update("scripts/" + name + ".py" for name in loop.split())
        self.assertIn("scripts/smoke_test_synthetic.py", required)
        self.assertFalse(required - set(package.FILES), f"missing upload files: {required - set(package.FILES)}")

    def test_archive_has_smoke_test_and_correct_manifest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fix.tar.gz"
            package.build_archive(output)
            with tarfile.open(output) as archive:
                self.assertIn("scripts/smoke_test_synthetic.py", archive.getnames())
                manifest = archive.extractfile("scripts/FP32_GRAPH_MEAN_SHA256SUMS").read().decode("utf-8")
                for line in manifest.splitlines():
                    checksum, name = line.split("  ", 1)
                    data = archive.extractfile(name).read()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), checksum)
                    self.assertNotIn(b"\r\n", data)
                    self.assertGreater(archive.getmember(name).mtime, 315532800)


if __name__ == "__main__":
    unittest.main()
