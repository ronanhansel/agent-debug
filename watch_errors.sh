#!/bin/bash
#
# Real-time Error Watcher - Shows errors from all benchmark logs
#
# Usage: ./watch_errors.sh [log_dir]
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Find the most recent log directory if not specified
if [ -z "$1" ]; then
    LOG_DIR=$(ls -td "$LOGS_BASE/benchmark_run_"* 2>/dev/null | head -1)
    if [ -z "$LOG_DIR" ]; then
        echo "No benchmark runs found."
        exit 1
    fi
else
    LOG_DIR="$1"
fi

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo "Watching for errors in: $LOG_DIR"
echo "Press Ctrl+C to stop"
echo "============================================================"

# Format and colorize function for errors
# Converts tail -f headers to "time + run_id" format
format_errors() {
    awk -v red="$RED" -v yellow="$YELLOW" -v cyan="$CYAN" -v magenta="$MAGENTA" -v nc="$NC" '
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
        # Silent switch - no separator line needed
        next
    }
    # Skip empty lines
    /^$/ { next }
    # Only show error lines
    tolower($0) ~ /error|exception|failed|traceback|401|unauthorized/ {
        timestamp = strftime("%H:%M:%S")
        # Format: benchmark/short_run_id
        if (current_benchmark != "") {
            run = current_run_id
            if (length(run) > 30) {
                run = substr(run, 1, 10) ".." substr(run, length(run)-14)
            }
            display_id = current_benchmark "/" run
        } else {
            display_id = current_run_id
        }
        prefix = sprintf("[%s %s] ", timestamp, display_id)

        # Strip redundant log prefix: "YYYY-MM-DD HH:MM:SS,mmm - agent_eval.verbose - DEBUG - "
        line = $0
        gsub(/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - [a-zA-Z_.]+ - (DEBUG|INFO|WARNING|ERROR) - /, "", line)

        if (tolower(line) ~ /401|unauthorized/) {
            printf "%s%s%s%s\n", magenta, prefix, line, nc
        } else {
            printf "%s%s%s%s\n", red, prefix, line, nc
        }
    }
    '
}

# Tail all log files and filter for errors
tail -f "$LOG_DIR"/*.log 2>/dev/null | format_errors
