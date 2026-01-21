#!/bin/bash
# Run rubric evaluations for all benchmarks
set -e

LOG_DIR="logs/rubrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Running Rubric Evaluations for All Benchmarks"
echo "============================================================"
echo "Using model: openai:gpt-5.2"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run rubric evaluation
run_rubric() {
    local name=$1
    local rubric=$2
    shift 2
    local trace_files=("$@")
    local logfile="$LOG_DIR/${name}_rubric.log"

    echo "[$(date +%H:%M:%S)] Starting $name rubric evaluation..."
    echo "   - Trace files: ${#trace_files[@]}"
    echo "   - Rubric: $rubric"

    # Build command with multiple --trace-file arguments
    local cmd="python scripts/eval_rubric.py"
    for file in "${trace_files[@]}"; do
        cmd="$cmd --trace-file $file"
    done
    cmd="$cmd --rubric $rubric --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"

    if eval "$cmd" > "$logfile" 2>&1; then
        # Extract summary from log
        local failed_count=$(grep -c "Failed task" "$logfile" 2>/dev/null || echo "0")
        local eval_count=$(grep -c "Evaluating task" "$logfile" 2>/dev/null || echo "0")
        echo "[$(date +%H:%M:%S)] ✓ $name completed"
        echo "   - Tasks evaluated: $eval_count"
        echo "   - Full log: $logfile"
    else
        echo "[$(date +%H:%M:%S)] ✗ $name FAILED"
        echo "   - Check log: $logfile"
        echo "   - Last 15 lines:"
        tail -15 "$logfile" | sed 's/^/     /'
        return 1
    fi
    echo ""
}

# CoreBench
echo "=== CoreBench ==="
prop_files=(traces/prop_corebench_hard_*_MERGED_UPLOAD.json)
iter1_files=(traces/iter1_corebench_hard_*_MERGED_UPLOAD.json)
all_corebench=("${prop_files[@]}" "${iter1_files[@]}")
run_rubric "CoreBench" "rubric_templates/corebench.txt" "${all_corebench[@]}"

# SciCode
echo "=== SciCode ==="
lady_files=(traces/scicode_lady_*_MERGED_UPLOAD.json)
honey_files=(traces/scicode_honey_*_MERGED_UPLOAD.json)
all_scicode=("${lady_files[@]}" "${honey_files[@]}")
run_rubric "SciCode" "rubric_templates/scicode.txt" "${all_scicode[@]}"

# ScienceAgentBench
echo "=== ScienceAgentBench ==="
mate_files=(traces/sab_mate_*_MERGED_UPLOAD.json)
cow_files=(traces/sab_cow_*_MERGED_UPLOAD.json)
husky_files=(traces/sab_husky_*_MERGED_UPLOAD.json)
all_sab=("${mate_files[@]}" "${cow_files[@]}" "${husky_files[@]}")
run_rubric "ScienceAgentBench" "rubric_templates/scienceagentbench.txt" "${all_sab[@]}"

# ColBench
echo "=== ColBench ==="
ivy_files=(traces/col_ivy_*_MERGED_UPLOAD.json)
zuck_files=(traces/col_zuck_*_MERGED_UPLOAD.json)
all_colbench=("${ivy_files[@]}" "${zuck_files[@]}")
run_rubric "ColBench" "rubric_templates/colbench.txt" "${all_colbench[@]}"

echo "============================================================"
echo "ALL RUBRIC EVALUATIONS COMPLETED!"
echo "============================================================"
echo ""
echo "Results are saved in: rubrics_output/"
echo ""
echo "To view results:"
echo "  ls -lh rubrics_output/*/"
echo ""
echo "Full logs in: $LOG_DIR/"
echo "============================================================"
