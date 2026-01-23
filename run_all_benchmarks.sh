#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs all benchmarks concurrently with centralized logging
#
# Usage: ./run_all_benchmarks.sh [prefix] [parallel]
# Example: ./run_all_benchmarks.sh moon2_ 20
#

set -o pipefail

# Get script directory (works on different machines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PREFIX="${1:-moon1_}"
PARALLEL="${2:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs/benchmark_run_${TIMESTAMP}"
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
NC='\033[0m' # No Color

# Benchmarks to run
BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")

# Create log directory
mkdir -p "$LOG_DIR"

# =============================================================================
# PRE-BUILD FUNCTION
# =============================================================================

# Comprehensive prebuild using the dedicated script
run_comprehensive_prebuild() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}[$(date +%H:%M:%S)] PHASE 1: PRE-BUILDING ALL DOCKER IMAGES${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo -e "${YELLOW}This ensures 800+ parallel processes don't race to build images.${NC}"
    echo ""

    # Use the comprehensive prebuild script
    if [ -f "$SCRIPT_DIR/prebuild_all_images.sh" ]; then
        if bash "$SCRIPT_DIR/prebuild_all_images.sh"; then
            echo -e "${GREEN}[prebuild] All Docker images ready!${NC}"
            return 0
        else
            echo -e "${RED}[prebuild] CRITICAL: Prebuild failed!${NC}"
            echo -e "${RED}Cannot proceed with benchmarks - image builds may race and fail.${NC}"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  1. Fix the errors above and re-run"
            echo "  2. Run ./prebuild_all_images.sh --force to rebuild all images"
            echo "  3. Check network connectivity and Docker daemon status"
            return 1
        fi
    else
        echo -e "${YELLOW}[prebuild] prebuild_all_images.sh not found, using legacy method...${NC}"
        # Legacy fallback
        if [ -f "$SCRIPT_DIR/prebuild_agent_envs.sh" ]; then
            bash "$SCRIPT_DIR/prebuild_agent_envs.sh"
        fi
        return 0
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Pre-build ALL images before starting benchmarks (CRITICAL for parallel execution)
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

# Function to filter log output (minimal - only critical messages)
filter_log() {
    local benchmark=$1
    local color=$2

    while IFS= read -r line; do
        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Only show critical messages (errors and completion)
        if echo "$line" | grep -qiE "error|exception|failed|traceback|401|unauthorized"; then
            echo -e "${RED}[$benchmark] $line${NC}"
        elif echo "$line" | grep -qiE "COMPLETED|FINISHED|All.*done"; then
            echo -e "${GREEN}[$benchmark] $line${NC}"
        fi
        # All other lines are only written to log, not displayed
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
        cd "$SCRIPT_DIR"
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

# Function to wait for completion (silent, no progress output)
wait_for_completion() {
    while true; do
        sleep 30  # Check every 30 seconds

        all_done=true
        for benchmark in "${BENCHMARKS[@]}"; do
            pid_file="$LOG_DIR/${benchmark}.pid"
            exit_code_file="$LOG_DIR/${benchmark}.exit_code"

            if [ -f "$exit_code_file" ]; then
                # Completed (success or failure)
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
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}[$(date +%H:%M:%S)] All benchmarks started!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${GREEN}Phase: RUNNING EVALUATIONS${NC}"
echo -e "${BLUE}Status: All 4 benchmarks running in parallel${NC}"
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

# Start waiting in background (silent)
wait_for_completion &
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
