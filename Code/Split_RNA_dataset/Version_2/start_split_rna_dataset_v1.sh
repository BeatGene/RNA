#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash start_split_rna_dataset_v1.sh {dry-run|execute} [pipeline options...]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
mode="$1"
shift

case "$mode" in
    dry-run)
        execute_args=()
        ;;
    execute)
        execute_args=(--execute)
        ;;
    *)
        usage
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pipeline_script="$script_dir/split_rna_dataset_v1.py"
[[ -f "$pipeline_script" ]] || {
    echo "Pipeline script not found: $pipeline_script" >&2
    exit 1
}

python_bin="${PYTHON_BIN:-python3}"
report_root="${REPORT_ROOT:-$HOME/Code/pipeline_reports}"
launcher_root="$report_root/background_launches"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
launcher_dir="$launcher_root/DATA_SPLIT_V1_${timestamp}_${mode^^}"
mkdir -p "$launcher_dir"

command=(
    "$python_bin"
    "$pipeline_script"
    "${execute_args[@]}"
    --report-root "$report_root"
    "$@"
)

printf '%q ' "${command[@]}" > "$launcher_dir/command.txt"
printf '\n' >> "$launcher_dir/command.txt"

nohup nice -n 10 env \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${command[@]}" \
    > "$launcher_dir/console.log" 2>&1 < /dev/null &

pid="$!"
printf '%s\n' "$pid" > "$launcher_dir/pid"
printf '%s\n' "$launcher_dir" > "$report_root/current_data_split_v1_run.txt"

echo "Started Data_V1 $mode in background."
echo "PID: $pid"
echo "Launcher directory: $launcher_dir"
echo "Console log: $launcher_dir/console.log"
echo "The timestamped pipeline report directory will be printed near the top of console.log."
