#!/usr/bin/env bash
# Persistent Protenix v1.0.5 runtime shared by foreground and nohup jobs.
# This file must be sourced, not executed in a transient Docker container.

PROTENIX_USER_ROOT="${PROTENIX_USER_ROOT:-/storage9920/home/tinghao.xia}"
PROTENIX_ENV_PREFIX="${PROTENIX_ENV_PREFIX:-$PROTENIX_USER_ROOT/miniconda3/envs/protenix-1.0.5}"
CUDA_TARGET_DIR="$PROTENIX_ENV_PREFIX/targets/x86_64-linux"

if [[ ! -x "$PROTENIX_ENV_PREFIX/bin/python" ]]; then
  echo "Persistent Protenix Python not found: $PROTENIX_ENV_PREFIX/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$PROTENIX_ENV_PREFIX/bin:$PATH"
export CUDA_HOME="$PROTENIX_ENV_PREFIX"
export CUDACXX="$PROTENIX_ENV_PREFIX/bin/nvcc"
export CPATH="$CUDA_TARGET_DIR/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_TARGET_DIR/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$PROTENIX_ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_TARGET_DIR/lib:$PROTENIX_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PROTENIX_ROOT_DIR="$PROTENIX_USER_ROOT/protenix_data"
export PYTHONUNBUFFERED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="$PROTENIX_USER_ROOT/.cache/protenix/triton"
export TORCH_EXTENSIONS_DIR="$PROTENIX_USER_ROOT/.cache/protenix/torch_extensions"

mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
umask 0002

protenix_memory_snapshot() {
  local current_file=/sys/fs/cgroup/memory.current
  local peak_file=/sys/fs/cgroup/memory.peak
  local max_file=/sys/fs/cgroup/memory.max
  [[ -r "$current_file" && -r "$max_file" ]] || return 0
  local current peak maximum
  current="$(<"$current_file")"
  peak="$(cat "$peak_file" 2>/dev/null || echo 0)"
  maximum="$(<"$max_file")"
  if [[ "$maximum" == "max" ]]; then
    printf 'CGROUP_MEMORY current_bytes=%s peak_bytes=%s max=unlimited\n' "$current" "$peak"
  else
    awk -v c="$current" -v p="$peak" -v m="$maximum" 'BEGIN {
      printf "CGROUP_MEMORY current_gib=%.1f peak_gib=%.1f max_gib=%.1f used_percent=%.1f\n", c/1073741824, p/1073741824, m/1073741824, 100*c/m
    }'
  fi
}

protenix_require_memory_below() {
  local limit_percent="${1:-70}"
  local current_file=/sys/fs/cgroup/memory.current
  local max_file=/sys/fs/cgroup/memory.max
  [[ -r "$current_file" && -r "$max_file" ]] || return 0
  local current maximum
  current="$(<"$current_file")"
  maximum="$(<"$max_file")"
  [[ "$maximum" != "max" ]] || return 0
  if ! awk -v c="$current" -v m="$maximum" -v l="$limit_percent" 'BEGIN { exit !(100*c/m < l) }'; then
    protenix_memory_snapshot >&2
    echo "Refusing to start: cgroup memory is at or above ${limit_percent}%" >&2
    return 1
  fi
}

# Replace small state files through the parent directory instead of opening an
# existing file in place.  This also recovers cleanly from legacy root-owned
# files left by the former Docker-in-Docker runtime, provided the directory is
# writable by the current user.
protenix_atomic_write() {
  local target="$1"
  local value="$2"
  local parent temp
  parent="$(dirname -- "$target")"
  mkdir -p "$parent"
  temp="${target}.tmp.$$.${RANDOM}"
  if ! printf '%s\n' "$value" > "$temp"; then
    command rm -f -- "$temp" 2>/dev/null || true
    return 1
  fi
  if ! command mv -f -- "$temp" "$target"; then
    command rm -f -- "$temp" 2>/dev/null || true
    return 1
  fi
}
