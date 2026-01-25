#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs all benchmarks concurrently with centralized logging
#
# Usage: ./run_all_benchmarks.sh [options] [prefix] [parallel]
#
# Options:
#   --continue    Continue from most recent failed run (reuses prefix and log dir)
#   --resume      Resume HAL runs (reuse run_id per config when available)
#   --prefix PFX  Prefix for run IDs and output files (default: moon1_)
#   --benchmarks  Comma-separated list of benchmarks to run (e.g., colbench,scicode)
#   --parallel    Total parallel tasks to run (caps at 800, ignores --parallel-models)
#   --parallel-models N  Number of model configs to run concurrently
#   --parallel-tasks N   Number of tasks to run concurrently per model
#   --trace-mode MODE    Set HAL_TRACE_MODE (e.g., local)
#   --sample-tasks N     Randomly sample N tasks from each benchmark dataset
#   --sample-seed N      Seed for --sample-tasks to make selection reproducible
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

# Log directory detection (respects DATA_PATH/HAL_DATA_ROOT)
local_logs_root() {
    local candidate="$SCRIPT_DIR/logs"
    if [ -L "$candidate" ] && [ ! -e "$candidate" ]; then
        echo "$SCRIPT_DIR/.logs"
        return
    fi
    echo "$candidate"
}

detect_logs_root() {
    if [ -n "${DATA_PATH:-}" ] && [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$DATA_PATH/hal_runs/$namespace/$(basename "$SCRIPT_DIR")/logs"
        return
    fi
    if [ -n "${HAL_DATA_ROOT:-}" ] && [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$HAL_DATA_ROOT/hal_runs/$namespace/$(basename "$SCRIPT_DIR")/logs"
        return
    fi
    local_logs_root
}

LOGS_BASE="$(detect_logs_root)"
if ! mkdir -p "$LOGS_BASE" 2>/dev/null; then
    LOGS_BASE="$(local_logs_root)"
    mkdir -p "$LOGS_BASE" 2>/dev/null || true
fi

# Parse arguments
CONTINUE_MODE=false
RESUME_MODE=false
PREFIX=""
PREFIX_FROM_ARG=false
PARALLEL=""
PARALLEL_TOTAL=""
PARALLEL_FLAG=false
PARALLEL_MODELS=""
PARALLEL_TASKS=""
TRACE_MODE=""
SAMPLE_TASKS=""
SAMPLE_SEED=""
REQUESTED_BENCHMARKS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --prefix)
            shift
            PREFIX="${1:-}"
            PREFIX_FROM_ARG=true
            shift
            ;;
        --benchmarks|--tags)
            shift
            IFS=',' read -r -a REQUESTED_BENCHMARKS <<< "${1:-}"
            shift
            ;;
        --parallel)
            shift
            PARALLEL_TOTAL="${1:-}"
            PARALLEL_FLAG=true
            shift
            ;;
        --parallel-models)
            shift
            PARALLEL_MODELS="${1:-}"
            shift
            ;;
        --parallel-tasks)
            shift
            PARALLEL_TASKS="${1:-}"
            shift
            ;;
        --trace-mode)
            shift
            TRACE_MODE="${1:-}"
            shift
            ;;
        --sample-tasks)
            shift
            SAMPLE_TASKS="${1:-}"
            shift
            ;;
        --sample-seed)
            shift
            SAMPLE_SEED="${1:-}"
            shift
            ;;
        *)
            if [ -z "$PREFIX" ]; then
                PREFIX="$1"
                PREFIX_FROM_ARG=true
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
PARALLEL_MODELS="${PARALLEL_MODELS:-$PARALLEL}"
PARALLEL_TASKS="${PARALLEL_TASKS:-$PARALLEL}"
PARALLEL_CAP=800

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

if $PARALLEL_FLAG; then
    if ! is_positive_int "$PARALLEL_TOTAL"; then
        echo "Invalid --parallel value: '$PARALLEL_TOTAL' (expected positive integer)"
        exit 1
    fi
    if [ "$PARALLEL_TOTAL" -gt "$PARALLEL_CAP" ]; then
        echo "Capping --parallel from $PARALLEL_TOTAL to $PARALLEL_CAP for stability."
        PARALLEL_TOTAL="$PARALLEL_CAP"
    fi
    if ! is_positive_int "$PARALLEL_MODELS"; then
        echo "Invalid --parallel-models value: '$PARALLEL_MODELS' (expected positive integer)"
        exit 1
    fi
    if [ "$PARALLEL_MODELS" -gt "$PARALLEL_TOTAL" ]; then
        PARALLEL_MODELS="$PARALLEL_TOTAL"
    fi
    PARALLEL_TASKS=$(( (PARALLEL_TOTAL + PARALLEL_MODELS - 1) / PARALLEL_MODELS ))
    if [ "$PARALLEL_TASKS" -lt 1 ]; then
        PARALLEL_TASKS=1
    fi
    PARALLEL_MODELS=$(( PARALLEL_TOTAL / PARALLEL_TASKS ))
    if [ "$PARALLEL_MODELS" -lt 1 ]; then
        PARALLEL_MODELS=1
    fi
else
    if ! is_positive_int "$PARALLEL_MODELS"; then
        echo "Invalid --parallel-models value: '$PARALLEL_MODELS' (expected positive integer)"
        exit 1
    fi
    if ! is_positive_int "$PARALLEL_TASKS"; then
        echo "Invalid --parallel-tasks value: '$PARALLEL_TASKS' (expected positive integer)"
        exit 1
    fi
fi

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

benchmark_in_list() {
    local target="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

if [ ${#REQUESTED_BENCHMARKS[@]} -gt 0 ]; then
    for b in "${REQUESTED_BENCHMARKS[@]}"; do
        if ! benchmark_in_list "$b" "${ALL_BENCHMARKS[@]}"; then
            echo "Unknown benchmark: $b"
            echo "Valid benchmarks: ${ALL_BENCHMARKS[*]}"
            exit 1
        fi
    done
fi

# =============================================================================
# CONTINUE MODE LOGIC
# =============================================================================

if $CONTINUE_MODE; then
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}       CONTINUE MODE - Finding previous run...${NC}"
    echo -e "${CYAN}============================================================${NC}"

    has_incomplete_marker() {
        local log_file="$1"
        [ -f "$log_file" ] || return 1
        grep -qiE "tasks are incomplete|incomplete tasks|continue-run flag to retry" "$log_file"
    }

    has_incomplete_results() {
        local benchmark="$1"
        local log_file="$2"
        python3 - "$benchmark" "$log_file" << 'PY'
import json
import re
import sys
from pathlib import Path

benchmark = sys.argv[1]
log_path = Path(sys.argv[2])
if not log_path.exists():
    sys.exit(0)

hal_map = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}
hal_benchmark = hal_map.get(benchmark, benchmark)

text = log_path.read_text(errors="ignore")
totals = [int(m.group(1)) for m in re.finditer(r"\((\d+) tasks\)", text)]
total_tasks = max(totals) if totals else None

run_ids = re.findall(r"Run ID: (\S+)", text)
if not run_ids:
    sys.exit(0)

run_ids = sorted(set(run_ids))

def find_run_dir(run_id: str) -> Path | None:
    for base in (Path("results") / hal_benchmark, Path("results") / benchmark):
        candidate = base / run_id
        if candidate.exists():
            return candidate
    return None

def count_completed(raw_path: Path) -> tuple[int, int]:
    completed = 0
    errors = 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            value = next(iter(obj.values()))
            if isinstance(value, str) and value.startswith("ERROR"):
                errors += 1
            else:
                completed += 1
    return completed, errors

for run_id in run_ids:
    run_dir = find_run_dir(run_id)
    if run_dir is None:
        sys.exit(0)
    raw_path = run_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
    if not raw_path.exists():
        sys.exit(0)
    completed, errors = count_completed(raw_path)
    if errors:
        sys.exit(0)
    if total_tasks is not None and completed < total_tasks:
        sys.exit(0)

sys.exit(1)
PY
    }

    find_log_dir_for_prefix() {
        local log_base="$1"
        local target_prefix="$2"
        local dir
        for dir in $(ls -td "$log_base"/benchmark_run_* 2>/dev/null); do
            local cfg="$dir/config.json"
            [ -f "$cfg" ] || continue
            local saved_prefix
            saved_prefix=$(grep -o '"prefix": *"[^"]*"' "$cfg" | cut -d'"' -f4)
            if [ "$saved_prefix" = "$target_prefix" ]; then
                echo "$dir"
                return 0
            fi
        done
        return 1
    }

    # Find the most recent log directory
    if $PREFIX_FROM_ARG; then
        LATEST_LOG_DIR=$(find_log_dir_for_prefix "$LOGS_BASE" "$PREFIX")
    else
        LATEST_LOG_DIR=$(ls -td "$LOGS_BASE"/benchmark_run_* 2>/dev/null | head -1)
    fi
    local_logs_root_value="$(local_logs_root)"
    if [ -z "$LATEST_LOG_DIR" ] && [ "$LOGS_BASE" != "$local_logs_root_value" ]; then
        original_logs_base="$LOGS_BASE"
        if $PREFIX_FROM_ARG; then
            FALLBACK_LOG_DIR=$(find_log_dir_for_prefix "$local_logs_root_value" "$PREFIX")
        else
            FALLBACK_LOG_DIR=$(ls -td "$local_logs_root_value"/benchmark_run_* 2>/dev/null | head -1)
        fi
        if [ -n "$FALLBACK_LOG_DIR" ]; then
            LOGS_BASE="$local_logs_root_value"
            LATEST_LOG_DIR="$FALLBACK_LOG_DIR"
            echo -e "${YELLOW}No runs found in ${original_logs_base}; using repo logs instead.${NC}"
        fi
    fi

    if [ -z "$LATEST_LOG_DIR" ]; then
        if $PREFIX_FROM_ARG; then
            echo -e "${RED}No previous runs found for prefix '${PREFIX}'. Starting fresh run.${NC}"
        else
            echo -e "${RED}No previous runs found in logs/. Starting fresh run.${NC}"
        fi
        CONTINUE_MODE=false
    else
        echo -e "${BLUE}Found previous run:${NC} $LATEST_LOG_DIR"

        # Extract prefix from config.json if available
        if [ -f "$LATEST_LOG_DIR/config.json" ]; then
            SAVED_PREFIX=$(grep -o '"prefix": *"[^"]*"' "$LATEST_LOG_DIR/config.json" | cut -d'"' -f4)
            if [ -n "$SAVED_PREFIX" ]; then
                if $PREFIX_FROM_ARG; then
                    if [ "$SAVED_PREFIX" != "$PREFIX" ]; then
                        echo -e "${YELLOW}Saved prefix '$SAVED_PREFIX' differs from requested '$PREFIX'. Using requested.${NC}"
                    fi
                else
                    PREFIX="$SAVED_PREFIX"
                    echo -e "${BLUE}Using saved prefix:${NC} $PREFIX"
                fi
            fi
            if ! $RESUME_MODE; then
                SAVED_RESUME=$(grep -o '"resume_mode": *[^,}]*' "$LATEST_LOG_DIR/config.json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
                if [ "$SAVED_RESUME" = "true" ]; then
                    RESUME_MODE=true
                    echo -e "${BLUE}Using saved resume mode:${NC} $RESUME_MODE"
                fi
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
            log_file="$LOG_DIR/${benchmark}.log"

            if [ -f "$exit_code_file" ]; then
                exit_code=$(cat "$exit_code_file")
                if [ "$exit_code" -eq 0 ]; then
                    if has_incomplete_marker "$log_file" || has_incomplete_results "$benchmark" "$log_file"; then
                        echo -e "  ${YELLOW}[INCOMPLETE]${NC} $benchmark - incomplete tasks detected, will retry"
                        BENCHMARKS_TO_RUN+=("$benchmark")
                    else
                        echo -e "  ${GREEN}[OK]${NC} $benchmark - completed successfully, skipping"
                    fi
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
    LOG_DIR="$LOGS_BASE/benchmark_run_${TIMESTAMP}"
    BENCHMARKS_TO_RUN=("${ALL_BENCHMARKS[@]}")
fi

if [ ${#REQUESTED_BENCHMARKS[@]} -gt 0 ]; then
    filtered=()
    for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
        if benchmark_in_list "$benchmark" "${REQUESTED_BENCHMARKS[@]}"; then
            filtered+=("$benchmark")
        fi
    done
    BENCHMARKS_TO_RUN=("${filtered[@]}")
    if [ ${#BENCHMARKS_TO_RUN[@]} -eq 0 ]; then
        echo "No benchmarks selected after filtering."
        exit 1
    fi
    echo -e "${BLUE}Using benchmark filter:${NC} ${BENCHMARKS_TO_RUN[*]}"
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
if [ -n "$TRACE_MODE" ]; then
    echo -e "${BLUE}HAL_TRACE_MODE:${NC} $TRACE_MODE"
fi
if [ -n "$SAMPLE_TASKS" ]; then
    if [ -n "$SAMPLE_SEED" ]; then
        echo -e "${BLUE}Sample Tasks:${NC} $SAMPLE_TASKS (seed=$SAMPLE_SEED)"
    else
        echo -e "${BLUE}Sample Tasks:${NC} $SAMPLE_TASKS (seed=random)"
    fi
fi
echo -e "${BLUE}Continue Mode:${NC} $CONTINUE_MODE"
echo -e "${BLUE}Resume Mode:${NC} $RESUME_MODE"
echo -e "${BLUE}Benchmarks:${NC} ${BENCHMARKS_TO_RUN[*]}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Save configuration
benchmarks_json=$(printf '"%s",' "${BENCHMARKS_TO_RUN[@]}")
benchmarks_json="[${benchmarks_json%,}]"
cat > "$LOG_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "prefix": "$PREFIX",
    "parallel_models": $PARALLEL_MODELS,
    "parallel_tasks": $PARALLEL_TASKS,
    "benchmarks": $benchmarks_json,
    "continue_mode": $CONTINUE_MODE,
    "resume_mode": $RESUME_MODE
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
        elif echo "$line" | grep -q "STEP:"; then
             # Allow progress steps to be visible
             echo -e "${color}[$benchmark] $line${NC}"
        fi
    done
}

# Function to run a single benchmark
run_benchmark() {
    local benchmark=$1
    local color=$2
    local log_file="$LOG_DIR/${benchmark}.log"
    local pid_file="$LOG_DIR/${benchmark}.pid"
    local extra_args=()

    if $RESUME_MODE; then
        extra_args+=(--resume)
    fi
    if [ -n "$TRACE_MODE" ]; then
        extra_args+=(--trace-mode "$TRACE_MODE")
    fi
    if [ -n "$SAMPLE_TASKS" ]; then
        extra_args+=(--sample-tasks "$SAMPLE_TASKS")
    fi
    if [ -n "$SAMPLE_SEED" ]; then
        extra_args+=(--sample-seed "$SAMPLE_SEED")
    fi

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
            "${extra_args[@]}" \
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

for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
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
$(for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
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
