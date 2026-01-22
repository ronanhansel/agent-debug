#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs all benchmarks concurrently with centralized logging
#
# Usage: ./run_all_benchmarks.sh [prefix] [parallel]
# Example: ./run_all_benchmarks.sh moon2_ 20
#

set -o pipefail

# Configuration
PREFIX="${1:-moon1_}"
PARALLEL="${2:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/Data/home/v-qizhengli/workspace/agent-debug/logs/benchmark_run_${TIMESTAMP}"
PARALLEL_MODELS=$PARALLEL
PARALLEL_TASKS=$PARALLEL

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Benchmarks to run
BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}       COMPREHENSIVE BENCHMARK RUNNER${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Timestamp:${NC} $TIMESTAMP"
echo -e "${BLUE}Prefix:${NC} $PREFIX"
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo -e "${BLUE}Parallel Models:${NC} $PARALLEL_MODELS"
echo -e "${BLUE}Parallel Tasks:${NC} $PARALLEL_TASKS"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Save configuration
cat > "$LOG_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "prefix": "$PREFIX",
    "parallel_models": $PARALLEL_MODELS,
    "parallel_tasks": $PARALLEL_TASKS,
    "benchmarks": ["scicode", "scienceagentbench", "corebench", "colbench"]
}
EOF

# Function to filter and colorize log output
filter_log() {
    local benchmark=$1
    local color=$2

    while IFS= read -r line; do
        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Always show these patterns
        if echo "$line" | grep -qiE "error|exception|failed|traceback|401|unauthorized"; then
            echo -e "${RED}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "success|completed|finished"; then
            echo -e "${GREEN}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "starting|started|running|task.*[0-9]+/[0-9]+|\[hal\]|\[main\]"; then
            echo -e "${color}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "warning|warn"; then
            echo -e "${YELLOW}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "trace saved|upload"; then
            echo -e "${GREEN}[$benchmark] $line${NC}"
        fi
        # Other lines are written to log but not displayed
    done
}

# Function to run a single benchmark
run_benchmark() {
    local benchmark=$1
    local color=$2
    local log_file="$LOG_DIR/${benchmark}.log"
    local err_file="$LOG_DIR/${benchmark}.err"
    local pid_file="$LOG_DIR/${benchmark}.pid"

    echo -e "${color}[$(date +%H:%M:%S)] Starting $benchmark benchmark...${NC}"

    # Run benchmark and capture output
    (
        cd /Data/home/v-qizhengli/workspace/agent-debug
        ./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py \
            --benchmark "$benchmark" \
            --all-configs \
            --all-tasks \
            --prefix "${benchmark}_${PREFIX}" \
            --docker \
            --parallel-models "$PARALLEL_MODELS" \
            --parallel-tasks "$PARALLEL_TASKS" \
            2>&1 | tee "$log_file" | filter_log "$benchmark" "$color"

        exit_code=${PIPESTATUS[0]}
        echo "$exit_code" > "$LOG_DIR/${benchmark}.exit_code"

        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}[$(date +%H:%M:%S)] $benchmark COMPLETED SUCCESSFULLY${NC}"
        else
            echo -e "${RED}[$(date +%H:%M:%S)] $benchmark FAILED with exit code $exit_code${NC}"
        fi
    ) &

    local pid=$!
    echo $pid > "$pid_file"
    echo -e "${color}[$(date +%H:%M:%S)] $benchmark started with PID $pid${NC}"
}

# Function to monitor progress
monitor_progress() {
    local check_interval=60  # Check every 60 seconds

    while true; do
        sleep $check_interval

        echo ""
        echo -e "${CYAN}[$(date +%H:%M:%S)] === PROGRESS UPDATE ===${NC}"

        all_done=true
        for benchmark in "${BENCHMARKS[@]}"; do
            pid_file="$LOG_DIR/${benchmark}.pid"
            exit_code_file="$LOG_DIR/${benchmark}.exit_code"

            if [ -f "$exit_code_file" ]; then
                exit_code=$(cat "$exit_code_file")
                if [ "$exit_code" -eq 0 ]; then
                    echo -e "${GREEN}  $benchmark: COMPLETED${NC}"
                else
                    echo -e "${RED}  $benchmark: FAILED (exit code: $exit_code)${NC}"
                fi
            elif [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null 2>&1; then
                    # Get last few relevant lines from log
                    last_activity=$(grep -iE "task|success|error|\[hal\]|\[main\]" "$LOG_DIR/${benchmark}.log" 2>/dev/null | tail -1)
                    echo -e "${BLUE}  $benchmark: RUNNING (PID: $pid)${NC}"
                    [ -n "$last_activity" ] && echo -e "    Last: $last_activity"
                    all_done=false
                else
                    echo -e "${YELLOW}  $benchmark: STOPPED (no exit code)${NC}"
                fi
            else
                echo -e "${YELLOW}  $benchmark: NOT STARTED${NC}"
                all_done=false
            fi
        done

        # Show Docker container count
        container_count=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -c "agentrun" || echo "0")
        echo -e "${CYAN}  Active containers: $container_count${NC}"

        if $all_done; then
            echo -e "${GREEN}[$(date +%H:%M:%S)] All benchmarks completed!${NC}"
            break
        fi

        echo -e "${CYAN}===============================${NC}"
    done
}

# Trap to handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}[$(date +%H:%M:%S)] Received interrupt signal. Cleaning up...${NC}"

    for benchmark in "${BENCHMARKS[@]}"; do
        pid_file="$LOG_DIR/${benchmark}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${YELLOW}Stopping $benchmark (PID: $pid)...${NC}"
                kill -TERM $pid 2>/dev/null
            fi
        fi
    done

    echo -e "${YELLOW}Logs saved to: $LOG_DIR${NC}"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Start all benchmarks
echo -e "${CYAN}[$(date +%H:%M:%S)] Starting all benchmarks...${NC}"
echo ""

# Assign colors to benchmarks
COLORS=("$BLUE" "$GREEN" "$YELLOW" "$CYAN")

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    color="${COLORS[$i]}"
    run_benchmark "$benchmark" "$color"
    sleep 2  # Small delay between starts to avoid race conditions
done

echo ""
echo -e "${CYAN}[$(date +%H:%M:%S)] All benchmarks started. Monitoring progress...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop all benchmarks${NC}"
echo ""

# Start monitoring in background
monitor_progress &
monitor_pid=$!

# Wait for all benchmark processes to complete
wait_for_benchmarks() {
    for benchmark in "${BENCHMARKS[@]}"; do
        pid_file="$LOG_DIR/${benchmark}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            wait $pid 2>/dev/null
        fi
    done
}

wait_for_benchmarks

# Kill monitor
kill $monitor_pid 2>/dev/null

# Final summary
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}                    FINAL SUMMARY${NC}"
echo -e "${CYAN}============================================================${NC}"

total_success=0
total_failed=0

for benchmark in "${BENCHMARKS[@]}"; do
    exit_code_file="$LOG_DIR/${benchmark}.exit_code"
    log_file="$LOG_DIR/${benchmark}.log"

    if [ -f "$exit_code_file" ]; then
        exit_code=$(cat "$exit_code_file")
        if [ "$exit_code" -eq 0 ]; then
            echo -e "${GREEN}  $benchmark: SUCCESS${NC}"
            ((total_success++))
        else
            echo -e "${RED}  $benchmark: FAILED (exit code: $exit_code)${NC}"
            ((total_failed++))
            # Show last error
            last_error=$(grep -iE "error|exception|failed" "$log_file" 2>/dev/null | tail -3)
            [ -n "$last_error" ] && echo -e "${RED}    Last errors:${NC}"
            [ -n "$last_error" ] && echo "$last_error" | sed 's/^/      /'
        fi
    else
        echo -e "${YELLOW}  $benchmark: UNKNOWN (no exit code)${NC}"
        ((total_failed++))
    fi
done

echo ""
echo -e "${CYAN}Total: $total_success succeeded, $total_failed failed${NC}"
echo -e "${CYAN}Logs saved to: $LOG_DIR${NC}"
echo -e "${CYAN}============================================================${NC}"

# Create summary file
cat > "$LOG_DIR/summary.txt" << EOF
Benchmark Run Summary
=====================
Timestamp: $TIMESTAMP
Prefix: $PREFIX

Results:
$(for benchmark in "${BENCHMARKS[@]}"; do
    exit_code_file="$LOG_DIR/${benchmark}.exit_code"
    if [ -f "$exit_code_file" ]; then
        exit_code=$(cat "$exit_code_file")
        echo "  $benchmark: exit_code=$exit_code"
    else
        echo "  $benchmark: no exit code"
    fi
done)

Success: $total_success
Failed: $total_failed
EOF

exit $total_failed
