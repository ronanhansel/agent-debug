#!/bin/bash
#
# Watch All Logs - Comprehensive real-time log viewer
#
# Usage: ./watch_all.sh [mode]
# Modes:
#   all      - Watch everything (default)
#   errors   - Only show errors
#   progress - Only show task progress
#   api      - Only show API calls/responses
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-all}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}           COMPREHENSIVE LOG WATCHER${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Mode:${NC} $MODE"
echo -e "${BLUE}Results:${NC} $RESULTS_DIR"
echo -e "${BLUE}Logs:${NC} $LOGS_DIR"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Set filter based on mode
case "$MODE" in
    errors)
        FILTER="error|exception|failed|traceback|401|403|429|500|502|503|504|timeout|unauthorized|denied"
        ;;
    progress)
        FILTER="task|starting|running|completed|success|finished|\[hal\]|\[main\]|SUCCESS|FAILED"
        ;;
    api)
        FILTER="openai|azure|api|request|response|token|model|gpt|o3|o4"
        ;;
    *)
        FILTER=""
        ;;
esac

# Format and colorize function
# Converts tail -f headers to "time + run_id" format
format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v white="$WHITE" -v nc="$NC" '
    BEGIN {
        current_run_id = ""
    }
    # Match tail -f file headers: ==> /path/to/benchmark/run_id/file.log <==
    /^==> .* <==/ {
        # Extract benchmark and run_id from path
        path = $2
        n = split(path, parts, "/")
        if (n >= 3) {
            # benchmark is parts[n-2], run_id is parts[n-1]
            current_benchmark = parts[n-2]
            current_run_id = parts[n-1]
        } else if (n >= 2) {
            current_benchmark = ""
            current_run_id = parts[n-1]
        } else {
            current_benchmark = ""
            current_run_id = path
        }
        # Silent switch - no separator line needed since each line has the prefix
        next
    }
    # Skip empty lines
    /^$/ { next }
    # Format regular lines with timestamp and run_id
    {
        timestamp = strftime("%H:%M:%S")
        # Format: benchmark/short_run_id (keep benchmark, truncate middle of run_id)
        if (current_benchmark != "") {
            # Extract key parts from run_id: model prefix and timestamp suffix
            run = current_run_id
            # Keep first 10 chars and last 15 chars of run_id if too long
            if (length(run) > 30) {
                run = substr(run, 1, 10) ".." substr(run, length(run)-14)
            }
            display_id = current_benchmark "/" run
        } else {
            display_id = current_run_id
        }
        prefix = sprintf("[%s %s] ", timestamp, display_id)

        line = $0
        # Strip redundant log prefix: "YYYY-MM-DD HH:MM:SS,mmm - agent_eval.verbose - DEBUG - "
        gsub(/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - [a-zA-Z_.]+ - (DEBUG|INFO|WARNING|ERROR) - /, "", line)
        # Colorize based on content
        if (tolower(line) ~ /error|exception|failed|traceback/) {
            printf "%s%s%s%s\n", red, prefix, line, nc
        } else if (tolower(line) ~ /401|403|429|500|502|503|504|timeout|unauthorized/) {
            printf "%s%s%s%s\n", magenta, prefix, line, nc
        } else if (tolower(line) ~ /success|completed|finished/) {
            printf "%s%s%s%s\n", green, prefix, line, nc
        } else if (tolower(line) ~ /warning|warn/) {
            printf "%s%s%s%s\n", yellow, prefix, line, nc
        } else if (tolower(line) ~ /starting|running|task/) {
            printf "%s%s%s%s\n", blue, prefix, line, nc
        } else if (tolower(line) ~ /\[scicode\]/) {
            printf "%s%s%s%s\n", cyan, prefix, line, nc
        } else if (tolower(line) ~ /\[corebench\]/) {
            printf "%s%s%s%s\n", green, prefix, line, nc
        } else if (tolower(line) ~ /\[colbench\]/) {
            printf "%s%s%s%s\n", yellow, prefix, line, nc
        } else if (tolower(line) ~ /\[scienceagentbench\]|\[sab\]/) {
            printf "%s%s%s%s\n", magenta, prefix, line, nc
        } else {
            printf "%s%s\n", prefix, line
        }
    }
    '
}

# Collect all log files
collect_logs() {
    local all_logs=""

    # Benchmark runner logs (centralized)
    if [ -d "$LOGS_DIR" ]; then
        for log_dir in $(ls -td "$LOGS_DIR"/benchmark_run_* 2>/dev/null | head -3); do
            for log in "$log_dir"/*.log; do
                [ -f "$log" ] && all_logs="$all_logs $log"
            done
        done
    fi

    # Per-run verbose logs (recent ones)
    for benchmark_dir in "$RESULTS_DIR"/*/; do
        if [ -d "$benchmark_dir" ]; then
            # Get most recent run directories
            for run_dir in $(ls -td "$benchmark_dir"*/ 2>/dev/null | head -5); do
                for log in "$run_dir"/*_verbose.log; do
                    [ -f "$log" ] && all_logs="$all_logs $log"
                done
            done
        fi
    done

    echo "$all_logs"
}

# Main watching loop
watch_logs() {
    while true; do
        LOG_FILES=$(collect_logs)

        if [ -z "$LOG_FILES" ]; then
            echo -e "${YELLOW}No log files found. Waiting...${NC}"
            sleep 5
            continue
        fi

        LOG_COUNT=$(echo $LOG_FILES | wc -w)
        echo -e "${CYAN}Watching $LOG_COUNT log files...${NC}"

        if [ -n "$FILTER" ]; then
            # Include file headers (==> ... <==) in filter so we can track which file each line is from
            tail -f $LOG_FILES 2>/dev/null | grep -iE --line-buffered "^==> .* <==|$FILTER" | format_and_colorize
        else
            tail -f $LOG_FILES 2>/dev/null | format_and_colorize
        fi

        # If tail exits, wait and retry
        sleep 2
    done
}

watch_logs
