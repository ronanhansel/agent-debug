#!/bin/bash
# Master script to run all benchmark merge and fetch operations sequentially
# Only shows summaries and errors, not full output

set -e

LOG_DIR="logs/merge_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Running All Benchmark Merge and Fetch Operations"
echo "============================================================"
echo "Log directory: $LOG_DIR"
echo ""
echo "This will process 4 benchmarks:"
echo "  1. CoreBench (prop + iter1) - 20 merged traces"
echo "  2. SciCode (lady + honey) - 20 merged traces"
echo "  3. ScienceAgentBench (mate + cow + husky) - 18 merged traces"
echo "  4. ColBench (ivy + zuck) - 18 merged traces"
echo ""
echo "Total: 76 merged trace files"
echo "============================================================"
echo ""

# Function to run a benchmark script and capture output
run_benchmark() {
    local script=$1
    local name=$2
    local logfile="$LOG_DIR/${name}.log"

    echo "[$(date +%H:%M:%S)] Starting $name..."

    if bash "$script" > "$logfile" 2>&1; then
        # Extract summary info from log
        local merged_count=$(grep -c "Merged [0-9]* traces" "$logfile" 2>/dev/null || echo "0")
        local wrote_count=$(grep -c "Wrote /home" "$logfile" 2>/dev/null || echo "0")
        echo "[$(date +%H:%M:%S)] ✓ $name completed"
        echo "   - Merge operations: $merged_count"
        echo "   - Files written: $wrote_count"
        echo "   - Full log: $logfile"
    else
        echo "[$(date +%H:%M:%S)] ✗ $name FAILED"
        echo "   - Check log for details: $logfile"
        echo "   - Last 20 lines of log:"
        tail -20 "$logfile" | sed 's/^/     /'
        return 1
    fi
    echo ""
}

# Run benchmarks sequentially
run_benchmark "./run_corebench_merge.sh" "CoreBench"
run_benchmark "./run_scicode_merge.sh" "SciCode"
run_benchmark "./run_sab_merge.sh" "ScienceAgentBench"
run_benchmark "./run_colbench_merge.sh" "ColBench"

echo "============================================================"
echo "ALL BENCHMARKS COMPLETED!"
echo "============================================================"
echo ""
echo "Merged trace files created in traces/ directory:"
ls -1 traces/*_MERGED_UPLOAD.json 2>/dev/null | wc -l | xargs echo "  Total merged files:"
echo ""
echo "Next steps - Run rubric evaluations:"
echo ""
echo "# CoreBench:"
echo "python scripts/eval_rubric.py --trace-file traces/prop_corebench_hard_*_MERGED_UPLOAD.json traces/iter1_corebench_hard_*_MERGED_UPLOAD.json --rubric rubric_templates/corebench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# SciCode:"
echo "python scripts/eval_rubric.py --trace-file traces/scicode_lady_*_MERGED_UPLOAD.json traces/scicode_honey_*_MERGED_UPLOAD.json --rubric rubric_templates/scicode.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# ScienceAgentBench:"
echo "python scripts/eval_rubric.py --trace-file traces/sab_mate_*_MERGED_UPLOAD.json traces/sab_cow_*_MERGED_UPLOAD.json traces/sab_husky_*_MERGED_UPLOAD.json --rubric rubric_templates/scienceagentbench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# ColBench:"
echo "python scripts/eval_rubric.py --trace-file traces/col_ivy_*_MERGED_UPLOAD.json traces/col_zuck_*_MERGED_UPLOAD.json --rubric rubric_templates/colbench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "============================================================"
