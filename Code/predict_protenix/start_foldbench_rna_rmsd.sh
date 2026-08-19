#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
FOLDBENCH_ENV_PREFIX="${FOLDBENCH_ENV_PREFIX:-$HOME/miniconda3/envs/foldbench}"
PYTHON_BIN="$FOLDBENCH_ENV_PREFIX/bin/python"
SCRIPT_ROOT="${SCRIPT_ROOT:-$HOME/Code/predict_protenix}"
OUTPUT_ROOT="${RMSD_OUTPUT_ROOT:-$HOME/Json_data/Foldbench_evaluation/rmsd}"
REPORT_ROOT="${RMSD_REPORT_ROOT:-$HOME/Code/pipeline_reports/FOLDBENCH_RNA_RMSD}"
WORKERS="${RMSD_WORKERS:-16}"
TIMEOUT_SECONDS="${RMSD_TIMEOUT_SECONDS:-1800}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found or not executable: $PYTHON_BIN" >&2
    exit 2
fi
mkdir -p "$REPORT_ROOT/runs" "$OUTPUT_ROOT"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPORT_ROOT/runs/${RUN_STAMP}_${MODE}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$REPORT_ROOT/current_run.txt"

COMMON_ENV=(
    "OMP_NUM_THREADS=1"
    "OPENBLAS_NUM_THREADS=1"
    "MKL_NUM_THREADS=1"
    "NUMEXPR_NUM_THREADS=1"
)

case "$MODE" in
    smoke)
        if [[ ! -f "$SCRIPT_ROOT/run_foldbench_rna_rmsd.py" ]]; then
            echo "Runner not found: $SCRIPT_ROOT/run_foldbench_rna_rmsd.py" >&2
            exit 2
        fi
        COMMAND=(
            "$PYTHON_BIN" "$SCRIPT_ROOT/run_foldbench_rna_rmsd.py"
            --output-root "$OUTPUT_ROOT"
            --workers "${RMSD_SMOKE_WORKERS:-4}"
            --timeout-seconds "$TIMEOUT_SECONDS"
            --limit "${RMSD_SMOKE_LIMIT:-50}"
            --progress-every 10
        )
        ;;
    run)
        if [[ ! -f "$SCRIPT_ROOT/run_foldbench_rna_rmsd.py" ]]; then
            echo "Runner not found: $SCRIPT_ROOT/run_foldbench_rna_rmsd.py" >&2
            exit 2
        fi
        COMMAND=(
            "$PYTHON_BIN" "$SCRIPT_ROOT/run_foldbench_rna_rmsd.py"
            --output-root "$OUTPUT_ROOT"
            --workers "$WORKERS"
            --timeout-seconds "$TIMEOUT_SECONDS"
            --progress-every 100
        )
        ;;
    summarize)
        if [[ ! -f "$SCRIPT_ROOT/summarize_foldbench_rna_rmsd.py" ]]; then
            echo "Summary script not found: $SCRIPT_ROOT/summarize_foldbench_rna_rmsd.py" >&2
            exit 2
        fi
        COMMAND=(
            "$PYTHON_BIN" "$SCRIPT_ROOT/summarize_foldbench_rna_rmsd.py"
            --output-root "$OUTPUT_ROOT"
        )
        ;;
    rescue)
        if [[ ! -f "$SCRIPT_ROOT/rescue_foldbench_rna_rmsd.py" ]]; then
            echo "Rescue script not found: $SCRIPT_ROOT/rescue_foldbench_rna_rmsd.py" >&2
            exit 2
        fi
        COMMAND=(
            "$PYTHON_BIN" "$SCRIPT_ROOT/rescue_foldbench_rna_rmsd.py"
            --output-root "$OUTPUT_ROOT"
            --workers "${RMSD_RESCUE_WORKERS:-16}"
            --timeout-seconds "$TIMEOUT_SECONDS"
        )
        ;;
    final-report)
        if [[ ! -f "$SCRIPT_ROOT/build_foldbench_rna_rmsd_final_report.py" ]]; then
            echo "Final-report script not found: $SCRIPT_ROOT/build_foldbench_rna_rmsd_final_report.py" >&2
            exit 2
        fi
        COMMAND=(
            "$PYTHON_BIN" "$SCRIPT_ROOT/build_foldbench_rna_rmsd_final_report.py"
            --rmsd-root "$OUTPUT_ROOT"
            --expected-targets 2199
            --expected-excluded 32
            --expected-valid 2167
        )
        ;;
    *)
        echo "Usage: $0 {smoke|run|rescue|summarize|final-report}" >&2
        exit 2
        ;;
esac

printf '%q ' nice -n 10 env "${COMMON_ENV[@]}" "${COMMAND[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
printf 'RUNNING\n' > "$RUN_DIR/exit_code"

nohup bash -c '
    EXIT_FILE="$1"
    shift
    set +e
    "$@"
    JOB_EXIT=$?
    printf "%s\n" "$JOB_EXIT" > "$EXIT_FILE"
    exit "$JOB_EXIT"
' bash "$RUN_DIR/exit_code" nice -n 10 env "${COMMON_ENV[@]}" "${COMMAND[@]}" \
    > "$RUN_DIR/console.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf '%s\n' "$LAUNCH_PID" > "$RUN_DIR/pid"

echo "Started mode=$MODE pid=$LAUNCH_PID"
echo "Run directory: $RUN_DIR"
echo "Console log: $RUN_DIR/console.log"
echo "Stable evaluation output: $OUTPUT_ROOT"
