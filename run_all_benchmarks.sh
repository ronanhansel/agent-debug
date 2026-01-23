#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs all benchmarks concurrently with centralized logging
#
# Usage: ./run_all_benchmarks.sh [options] [prefix] [parallel]
#
# Options:
#   --continue    Continue from most recent failed run (reuses prefix and log dir)
#
# Examples:
#   ./run_all_benchmarks.sh moon2_ 20           # Fresh run with moon2_ prefix
#   ./run_all_benchmarks.sh --continue          # Continue most recent failed run
#   ./run_all_benchmarks.sh --continue moon4_   # Continue run with specific prefix
#

set -o pipefail

# Get script directory (works on different machines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CONTINUE_MODE=false
PREFIX=""
PARALLEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        *)
            if [ -z "$PREFIX" ]; then
                PREFIX="$1"
            elif [ -z "$PARALLEL" ]; then
                PARALLEL="$1"
            fi
            shift
            ;;
    esac
done

# Defaults
PREFIX="${PREFIX:-moon1_}"
PARALLEL="${PARALLEL:-10}"
PARALLEL_MODELS=$PARALLEL
PARALLEL_TASKS=$PARALLEL

# Use host networking to avoid docker0 bridge limits (allows 1000+ containers)
export HAL_DOCKER_NETWORK_MODE=host

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Benchmarks to run
ALL_BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")
BENCHMARKS_TO_RUN=()

# =============================================================================
# CONTINUE MODE LOGIC
# =============================================================================

if $CONTINUE_MODE; then
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}       CONTINUE MODE - Finding previous run...${NC}"
    echo -e "${CYAN}============================================================${NC}"

    # Find the most recent log directory
    LATEST_LOG_DIR=$(ls -td "$SCRIPT_DIR/logs"/benchmark_run_* 2>/dev/null | head -1)

    if [ -z "$LATEST_LOG_DIR" ]; then
        echo -e "${RED}No previous runs found in logs/. Starting fresh run.${NC}"
        CONTINUE_MODE=false
    else
        echo -e "${BLUE}Found previous run:${NC} $LATEST_LOG_DIR"

        # Extract prefix from config.json if available
        if [ -f "$LATEST_LOG_DIR/config.json" ]; then
            SAVED_PREFIX=$(grep -o '"prefix": *"[^"]*"' "$LATEST_LOG_DIR/config.json" | cut -d'"' -f4)
            if [ -n "$SAVED_PREFIX" ]; then
                PREFIX="$SAVED_PREFIX"
                echo -e "${BLUE}Using saved prefix:${NC} $PREFIX"
            fi
        fi

        # Use the existing log directory
        LOG_DIR="$LATEST_LOG_DIR"
        TIMESTAMP=$(basename "$LOG_DIR" | sed 's/benchmark_run_//')

        # Find failed/incomplete benchmarks
        echo ""
        echo -e "${BLUE}Checking benchmark status:${NC}"
        for benchmark in "${ALL_BENCHMARKS[@]}"; do
            exit_code_file="$LOG_DIR/${benchmark}.exit_code"
            pid_file="$LOG_DIR/${benchmark}.pid"

            if [ -f "$exit_code_file" ]; then
                exit_code=$(cat "$exit_code_file")
                if [ "$exit_code" -eq 0 ]; then
                    echo -e "  ${GREEN}[OK]${NC} $benchmark - completed successfully, skipping"
                else
                    echo -e "  ${RED}[FAILED]${NC} $benchmark - exit code $exit_code, will retry"
                    BENCHMARKS_TO_RUN+=("$benchmark")
                fi
            elif [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null 2>&1; then
                    echo -e "  ${YELLOW}[RUNNING]${NC} $benchmark - still running (PID: $pid), killing..."
                    kill -TERM $pid 2>/dev/null
                    sleep 2
                    kill -9 $pid 2>/dev/null
                fi
                echo -e "  ${YELLOW}[INCOMPLETE]${NC} $benchmark - will retry"
                BENCHMARKS_TO_RUN+=("$benchmark")
            else
                echo -e "  ${YELLOW}[NOT STARTED]${NC} $benchmark - will run"
                BENCHMARKS_TO_RUN+=("$benchmark")
            fi
        done

        if [ ${#BENCHMARKS_TO_RUN[@]} -eq 0 ]; then
            echo ""
            echo -e "${GREEN}All benchmarks completed successfully! Nothing to continue.${NC}"
            exit 0
        fi

        echo ""
        echo -e "${BLUE}Will run ${#BENCHMARKS_TO_RUN[@]} benchmark(s):${NC} ${BENCHMARKS_TO_RUN[*]}"
    fi
fi

# If not continue mode or no previous run found, run all benchmarks
if ! $CONTINUE_MODE; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="$SCRIPT_DIR/logs/benchmark_run_${TIMESTAMP}"
    BENCHMARKS_TO_RUN=("${ALL_BENCHMARKS[@]}")
fi

# Create log directory
mkdir -p "$LOG_DIR"

# =============================================================================
# PRE-BUILD FUNCTION
# =============================================================================

run_comprehensive_prebuild() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}[$(date +%H:%M:%S)] PHASE 1: PRE-BUILDING ALL DOCKER IMAGES${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""

    if [ -f "$SCRIPT_DIR/prebuild_all_images.sh" ]; then
        if bash "$SCRIPT_DIR/prebuild_all_images.sh"; then
            echo -e "${GREEN}[prebuild] All Docker images ready!${NC}"
            return 0
        else
            echo -e "${RED}[prebuild] CRITICAL: Prebuild failed!${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}[prebuild] prebuild_all_images.sh not found, skipping...${NC}"
        return 0
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Pre-build ALL images before starting benchmarks
if ! run_comprehensive_prebuild; then
    echo ""
    echo -e "${RED}================================================================${NC}"
    echo -e "${RED}  ABORTING: Cannot start benchmarks without pre-built images${NC}"
    echo -e "${RED}================================================================${NC}"
    exit 1
fi
echo ""

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}       COMPREHENSIVE BENCHMARK RUNNER${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Script Dir:${NC} $SCRIPT_DIR"
echo -e "${BLUE}Timestamp:${NC} $TIMESTAMP"
echo -e "${BLUE}Prefix:${NC} $PREFIX"
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo -e "${BLUE}Parallel Models:${NC} $PARALLEL_MODELS"
echo -e "${BLUE}Parallel Tasks:${NC} $PARALLEL_TASKS"
echo -e "${BLUE}Continue Mode:${NC} $CONTINUE_MODE"
echo -e "${BLUE}Benchmarks:${NC} ${BENCHMARKS_TO_RUN[*]}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Save configuration
cat > "$LOG_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "prefix": "$PREFIX",
    "parallel_models": $PARALLEL_MODELS,
    "parallel_tasks": $PARALLEL_TASKS,
    "benchmarks": ["scicode", "scienceagentbench", "corebench", "colbench"],
    "continue_mode": $CONTINUE_MODE
}
EOF

# Function to filter log output (minimal - only critical messages)
filter_log() {
    local benchmark=$1
    local color=$2

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if echo "$line" | grep -qiE "error|exception|failed|traceback|401|unauthorized"; then
            echo -e "${RED}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "COMPLETED|FINISHED|All.*done"; then
            echo -e "${GREEN}[$benchmark] $line${NC}"
        fi
    done
}

# Function to run a single benchmark
run_benchmark() {
    local benchmark=$1
    local color=$2
    local log_file="$LOG_DIR/${benchmark}.log"
    local pid_file="$LOG_DIR/${benchmark}.pid"

    echo -e "${color}[$(date +%H:%M:%S)] Starting $benchmark benchmark...${NC}"

    # Run benchmark and capture output
    (
        cd "$SCRIPT_DIR"
        ./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py \
            --benchmark "$benchmark" \
            --all-configs \
            --all-tasks \
            --prefix "${benchmark}_${PREFIX}" \
            --docker \
            --parallel-models "$PARALLEL_MODELS" \
            --parallel-tasks "$PARALLEL_TASKS" \
            2>&1 | tee -a "$log_file" | filter_log "$benchmark" "$color"

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

# Function to wait for completion
wait_for_completion() {
    while true; do
        sleep 30
        all_done=true
        for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
            pid_file="$LOG_DIR/${benchmark}.pid"
            exit_code_file="$LOG_DIR/${benchmark}.exit_code"
            if [ -f "$exit_code_file" ]; then
                continue
            elif [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null 2>&1; then
                    all_done=false
                fi
            else
                all_done=false
            fi
        done
        if $all_done; then
            break
        fi
    done
}

# Trap to handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}[$(date +%H:%M:%S)] Received interrupt signal. Cleaning up...${NC}"

    for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
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

# Start benchmarks
echo -e "${CYAN}[$(date +%H:%M:%S)] Starting ${#BENCHMARKS_TO_RUN[@]} benchmark(s)...${NC}"
echo ""

# Assign colors to benchmarks
declare -A BENCHMARK_COLORS
BENCHMARK_COLORS["scicode"]="$BLUE"
BENCHMARK_COLORS["scienceagentbench"]="$GREEN"
BENCHMARK_COLORS["corebench"]="$YELLOW"
BENCHMARK_COLORS["colbench"]="$CYAN"

for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
    color="${BENCHMARK_COLORS[$benchmark]:-$WHITE}"
    run_benchmark "$benchmark" "$color"
    sleep 2
done

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}[$(date +%H:%M:%S)] All benchmarks started!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${GREEN}Phase: RUNNING EVALUATIONS${NC}"
echo -e "${BLUE}Status: ${#BENCHMARKS_TO_RUN[@]} benchmark(s) running${NC}"
echo ""
echo -e "${YELLOW}To monitor progress, open another terminal and run:${NC}"
echo -e "  ${WHITE}./watch_all.sh${NC}           # Dashboard view (default)"
echo -e "  ${WHITE}./watch_all.sh logs${NC}      # Detailed log output"
echo -e "  ${WHITE}./watch_all.sh errors${NC}    # Errors only"
echo ""
echo -e "${BLUE}Log directory:${NC} $LOG_DIR"
echo -e "${CYAN}Press Ctrl+C to stop all benchmarks${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Wait in background
wait_for_completion &
monitor_pid=$!

# Wait for all benchmark processes
for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
    pid_file="$LOG_DIR/${benchmark}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        wait $pid 2>/dev/null
    fi
done

kill $monitor_pid 2>/dev/null

# Final summary
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}                    FINAL SUMMARY${NC}"
echo -e "${CYAN}============================================================${NC}"

total_success=0
total_failed=0

for benchmark in "${ALL_BENCHMARKS[@]}"; do
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
if [ $total_failed -gt 0 ]; then
    echo -e "${YELLOW}To retry failed benchmarks: ./run_all_benchmarks.sh --continue${NC}"
fi
echo -e "${CYAN}============================================================${NC}"

# Create summary file
cat > "$LOG_DIR/summary.txt" << EOF
Benchmark Run Summary
=====================
Timestamp: $TIMESTAMP
Prefix: $PREFIX
Continue Mode: $CONTINUE_MODE

Results:
$(for benchmark in "${ALL_BENCHMARKS[@]}"; do
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
