#!/usr/bin/env bash
set -euo pipefail

REPORT_ROOT="${RMSD_REPORT_ROOT:-$HOME/Code/pipeline_reports/FOLDBENCH_RNA_RMSD}"
OUTPUT_ROOT="${RMSD_OUTPUT_ROOT:-$HOME/Json_data/Foldbench_evaluation/rmsd}"
CURRENT_FILE="$REPORT_ROOT/current_run.txt"

if [[ ! -f "$CURRENT_FILE" ]]; then
    echo "No run has been launched: $CURRENT_FILE does not exist" >&2
    exit 2
fi

RUN_DIR="$(<"$CURRENT_FILE")"
echo "Run directory: $RUN_DIR"

if [[ -f "$RUN_DIR/pid" ]]; then
    RUN_PID="$(<"$RUN_DIR/pid")"
    if ps -p "$RUN_PID" -o pid=,etime=,stat=,pcpu=,pmem=,cmd=; then
        echo "Launcher process is present"
    else
        echo "Launcher process is no longer present"
    fi
fi

if [[ -f "$RUN_DIR/exit_code" ]]; then
    echo "Exit code: $(<"$RUN_DIR/exit_code")"
fi

if [[ -f "$OUTPUT_ROOT/progress.json" ]]; then
    echo "Progress:"
    python -m json.tool "$OUTPUT_ROOT/progress.json"
fi

if [[ -f "$OUTPUT_ROOT/run_summary.json" ]]; then
    echo "Last completed run summary:"
    python -m json.tool "$OUTPUT_ROOT/run_summary.json"
fi

if [[ -f "$OUTPUT_ROOT/rigid_only_rescue/progress.json" ]]; then
    echo "Rigid-only rescue progress:"
    python -m json.tool "$OUTPUT_ROOT/rigid_only_rescue/progress.json"
fi

if [[ -f "$OUTPUT_ROOT/rigid_only_rescue/run_summary.json" ]]; then
    echo "Last completed rigid-only rescue summary:"
    python -m json.tool "$OUTPUT_ROOT/rigid_only_rescue/run_summary.json"
fi

if [[ -f "$OUTPUT_ROOT/final_report/final_report_summary.txt" ]]; then
    echo "Final RMSD report summary:"
    cat "$OUTPUT_ROOT/final_report/final_report_summary.txt"
fi

if [[ -f "$RUN_DIR/console.log" ]]; then
    echo "Latest console output:"
    tail -n 30 "$RUN_DIR/console.log"
fi
