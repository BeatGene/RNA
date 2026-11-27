#!/usr/bin/env bash
# Run locally. SSH asks for the laboratory password in your terminal.
set -euo pipefail

local_root=$(cd "$(dirname "$0")/.." && pwd)
remote_user_host=tinghao.xia@dubhe.lglab.ac.cn
port=32344
stamp=$(date +%Y%m%d_%H%M%S)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/rna-upload.XXXXXX")
upload_id="${stamp}_$(basename "$work_dir")"
control_path="$work_dir/ssh.sock"
ssh_args=(-p "$port" -o "ControlPath=$control_path" -o ControlMaster=auto -o ControlPersist=600)

cleanup() {
    ssh "${ssh_args[@]}" -O exit "$remote_user_host" >/dev/null 2>&1 || true
    # Keep archive and manifest for inspection/retry.
    printf 'Local upload artifacts: %s\n' "$work_dir"
}
trap cleanup EXIT

# Hash source files so the server can verify both staging and installed content.
python3 - "$local_root" "$work_dir/SHA256SUMS" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
with open(sys.argv[2], 'w') as out:
    for name in ('config', 'etflow', 'scripts'):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit('Missing directory: ' + str(folder))
        for path in sorted(folder.rglob('*')):
            rel = path.relative_to(root)
            if ('__pycache__' in rel.parts or path.name == '.DS_Store'
                    or path.suffix in ('.pyc', '.pyo')):
                continue
            if path.is_symlink():
                raise SystemExit('Inspect symlink before upload: ' + str(path))
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                out.write(digest + '  ' + rel.as_posix() + '\n')
PY

COPYFILE_DISABLE=1 tar -czf "$work_dir/code.tar.gz" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
    --exclude='.DS_Store' \
    -C "$local_root" config etflow scripts \
    -C "$work_dir" SHA256SUMS

# Authenticate once; scp and subsequent ssh reuse this connection.
ssh "${ssh_args[@]}" -fN "$remote_user_host"
ssh "${ssh_args[@]}" "$remote_user_host" "mkdir -p \"\$HOME/.rna-upload/$upload_id\""
scp -P "$port" -o "ControlPath=$control_path" \
    "$work_dir/code.tar.gz" "$remote_user_host:.rna-upload/$upload_id/code.tar.gz"

ssh "${ssh_args[@]}" "$remote_user_host" bash -s -- "$upload_id" <<'REMOTE'
set -euo pipefail
upload_id=$1
root="$HOME/Code_Flow_matching"
stage="$HOME/.rna-upload/$upload_id"
backup="$HOME/Code_Flow_matching_backup_$upload_id"
[[ -d "$root" && ! -L "$root" ]] || { echo "Unexpected project path: $root" >&2; exit 1; }
for name in config etflow scripts; do
    [[ ! -L "$root/$name" ]] || { echo "Refusing symlink: $root/$name" >&2; exit 1; }
done
tar -xzf "$stage/code.tar.gz" -C "$stage"
(cd "$stage" && sha256sum --quiet -c SHA256SUMS)

mkdir "$backup"
# Preserve server-only scripts, while retaining a complete pre-upload backup.
if [[ -d "$root/scripts" ]]; then cp -a "$root/scripts" "$backup/scripts"; fi
for name in config etflow; do
    if [[ -e "$root/$name" ]]; then mv "$root/$name" "$backup/$name"; fi
    mv "$stage/$name" "$root/$name"
done
mkdir -p "$root/scripts"
cp -a "$stage/scripts/." "$root/scripts/"
(cd "$root" && sha256sum --quiet -c "$stage/SHA256SUMS")
echo "Upload verified: $root"
echo "Old config/etflow and scripts backup: $backup"
echo "Server-only scripts have been preserved."
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || true
echo "Dataset paths:"
ls -ld /storage9920/home/tinghao.xia/Data_PT_V2/{train,val} || true
REMOTE
