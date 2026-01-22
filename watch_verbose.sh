#!/bin/bash
#
# Watch Verbose Logs - Real-time view of all agent verbose logs
#
# Usage: ./watch_verbose.sh [benchmark] [filter]
# Examples:
#   ./watch_verbose.sh              # Watch all benchmarks
#   ./watch_verbose.sh scicode      # Watch only scicode
#   ./watch_verbose.sh "" error     # Watch all, filter for errors
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARK_FILTER="${1:-}"
TEXT_FILTER="${2:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
BOLD_GREEN='\033[1;32m'
NC='\033[0m'

RESULTS_DIR="$SCRIPT_DIR/results"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}           VERBOSE LOG WATCHER${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Results dir:${NC} $RESULTS_DIR"
echo -e "${BLUE}Benchmark filter:${NC} ${BENCHMARK_FILTER:-all}"
echo -e "${BLUE}Text filter:${NC} ${TEXT_FILTER:-none}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Build the find pattern
if [ -n "$BENCHMARK_FILTER" ]; then
    FIND_PATH="$RESULTS_DIR/$BENCHMARK_FILTER*"
else
    FIND_PATH="$RESULTS_DIR"
fi

# Format and colorize function
# Converts tail -f headers to "time + run_id" format
format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v bold_green="$BOLD_GREEN" -v nc="$NC" '
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
        # PRIORITY 1: HAL evaluation results - BOLD GREEN spotlight
        if (line ~ /Results:.*\{/ || line ~ /"accuracy"/ || line ~ /"score"/ || \
            line ~ /Evaluation completed/ || line ~ /successful_tasks/ || line ~ /failed_tasks/) {
            printf "%s%s%s%s\n", bold_green, prefix, line, nc
        }
        # PRIORITY 2: Errors
        else if (tolower(line) ~ /error|exception|failed|traceback/) {
            printf "%s%s%s%s\n", red, prefix, line, nc
        } else if (tolower(line) ~ /401|403|429|500|502|503|504|timeout|unauthorized/) {
            printf "%s%s%s%s\n", magenta, prefix, line, nc
        } else if (tolower(line) ~ /success|completed|finished/) {
            printf "%s%s%s%s\n", green, prefix, line, nc
        } else if (tolower(line) ~ /warning|warn/) {
            printf "%s%s%s%s\n", yellow, prefix, line, nc
        } else if (tolower(line) ~ /starting|running|task/) {
            printf "%s%s%s%s\n", blue, prefix, line, nc
        } else {
            printf "%s%s\n", prefix, line
        }
    }
    '
}

# Find all verbose log files and tail them
find_and_tail() {
    # Find the most recent verbose logs (modified in last 2 hours)
    LOG_FILES=$(find $FIND_PATH -name "*_verbose.log" -mmin -120 2>/dev/null | sort -r | head -20)

    if [ -z "$LOG_FILES" ]; then
        echo -e "${YELLOW}No recent verbose logs found. Waiting for new logs...${NC}"
        sleep 5
        return
    fi

    echo -e "${CYAN}Watching $(echo "$LOG_FILES" | wc -l) log files...${NC}"
    echo ""

    # Tail all files with headers
    if [ -n "$TEXT_FILTER" ]; then
        # Include file headers (==> ... <==) in filter so we can track which file each line is from
        tail -f $LOG_FILES 2>/dev/null | grep -iE --line-buffered "^==> .* <==|$TEXT_FILTER" | format_and_colorize
    else
        tail -f $LOG_FILES 2>/dev/null | format_and_colorize
    fi
}

# Main loop - restart tail if files change
while true; do
    find_and_tail
    sleep 10
done
