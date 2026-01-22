#!/bin/bash
#
# Real-time Error Watcher - Shows errors from all benchmark logs
#
# Usage: ./watch_errors.sh [log_dir]
#

# Find the most recent log directory if not specified
if [ -z "$1" ]; then
    LOG_DIR=$(ls -td /Data/home/v-qizhengli/workspace/agent-debug/logs/benchmark_run_* 2>/dev/null | head -1)
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
NC='\033[0m'

echo "Watching for errors in: $LOG_DIR"
echo "Press Ctrl+C to stop"
echo "============================================================"

# Tail all log files and filter for errors
tail -f "$LOG_DIR"/*.log 2>/dev/null | while read line; do
    if echo "$line" | grep -qiE "error|exception|failed|traceback|401|unauthorized"; then
        # Extract benchmark name from the line
        if echo "$line" | grep -q "==>"; then
            echo -e "${YELLOW}$line${NC}"
        else
            echo -e "${RED}$line${NC}"
        fi
    fi
done
