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

# Function to colorize output based on content
colorize() {
    while IFS= read -r line; do
        if echo "$line" | grep -qiE "error|exception|failed|traceback"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -qiE "success|completed|finished"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -qiE "warning|warn"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qiE "starting|running|task"; then
            echo -e "${BLUE}$line${NC}"
        elif echo "$line" | grep -qiE "401|403|429|500|502|503|504|timeout|unauthorized"; then
            echo -e "${MAGENTA}$line${NC}"
        else
            echo "$line"
        fi
    done
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
        tail -f $LOG_FILES 2>/dev/null | grep -i --line-buffered "$TEXT_FILTER" | colorize
    else
        tail -f $LOG_FILES 2>/dev/null | colorize
    fi
}

# Main loop - restart tail if files change
while true; do
    find_and_tail
    sleep 10
done
