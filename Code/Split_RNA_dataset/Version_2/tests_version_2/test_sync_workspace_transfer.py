import argparse
import contextlib
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sync_workspace_inventory import inventory_tree
from sync_workspace_transfer import apply, pack, plan


class WorkspaceSyncTest(unittest.TestCase):
    def test_long_windows_report_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "reports/" + "/".join(("a" * 90, "b" * 90, "c" * 90 + ".txt"))
            content = b"long path"
            record = {"key": key, "size": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()}
            archive_path = root / "long.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = (json.dumps(record) + "\n").encode()
                info = tarfile.TarInfo("SYNC_MANIFEST.jsonl")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                info = tarfile.TarInfo(key)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            with contextlib.redirect_stdout(io.StringIO()):
                apply(argparse.Namespace(side="local", root=root, archive=archive_path,
                                         mode="server_only", backup_dir=None))

    def test_server_only_download_then_local_authoritative_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local = base / "local"
            server = base / "server"
            files = {
                local / "Code" / "script.py": b"local code",
                local / "Code" / "New_Data_pipeline_reports" / "audit.txt": b"local report",
                local / "Code_Flow_matching" / "shared.txt": b"same",
                server / "Code" / "script.py": b"old server code",
                server / "Code" / "only_server.txt": b"download me",
                server / "Code" / "pipeline_reports" / "audit.txt": b"old report",
                server / "Code_Flow_matching" / "shared.txt": b"same",
            }
            for path, data in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            manifests = {}
            for side, root, reports in (
                ("local", local, "New_Data_pipeline_reports"),
                ("server", server, "pipeline_reports"),
            ):
                manifest = base / f"{side}.jsonl"
                with manifest.open("w", encoding="utf-8") as output:
                    inventory_tree(root / "Code", "code/", reports, output)
                    inventory_tree(root / "Code_Flow_matching", "flow/", "", output)
                    inventory_tree(root / "Code" / reports, "reports/", "", output)
                manifests[side] = manifest
            report = base / "plan"
            with contextlib.redirect_stdout(io.StringIO()):
                plan(argparse.Namespace(local_manifest=manifests["local"],
                                        server_manifest=manifests["server"], output_dir=report))
            summary = json.loads((report / "plan_summary.json").read_text())
            self.assertEqual((summary["server_only"], summary["different_local_wins"],
                              summary["same"]), (1, 2, 1))
            with contextlib.redirect_stdout(io.StringIO()):
                pack(argparse.Namespace(side="server", root=server,
                                        manifest=manifests["server"], keys=report / "server_only.txt",
                                        output=base / "server_only.tar.gz"))
                apply(argparse.Namespace(side="local", root=local,
                                         archive=base / "server_only.tar.gz", mode="server_only",
                                         backup_dir=None))
                pack(argparse.Namespace(side="local", root=local,
                                        manifest=manifests["local"], keys=report / "local_push.txt",
                                        output=base / "local_push.tar.gz"))
                apply(argparse.Namespace(side="server", root=server,
                                         archive=base / "local_push.tar.gz", mode="local_authoritative",
                                         backup_dir=base / "backups"))
            self.assertEqual((local / "Code" / "only_server.txt").read_bytes(), b"download me")
            self.assertEqual((server / "Code" / "script.py").read_bytes(), b"local code")
            self.assertEqual((server / "Code" / "pipeline_reports" / "audit.txt").read_bytes(),
                             b"local report")
            self.assertEqual((base / "backups" / "code" / "script.py").read_bytes(),
                             b"old server code")
            self.assertEqual((base / "backups" / "reports" / "audit.txt").read_bytes(),
                             b"old report")


if __name__ == "__main__":
    unittest.main()
