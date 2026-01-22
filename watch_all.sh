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

RESULTS_DIR="/Data/home/v-qizhengli/workspace/agent-debug/results"
LOGS_DIR="/Data/home/v-qizhengli/workspace/agent-debug/logs"

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

# Colorize function
colorize() {
    while IFS= read -r line; do
        # Extract benchmark name from path if present
        if echo "$line" | grep -qE "==> .* <==" ; then
            echo -e "${WHITE}$line${NC}"
        elif echo "$line" | grep -qiE "error|exception|failed|traceback"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -qiE "401|403|429|500|502|503|504|timeout|unauthorized"; then
            echo -e "${MAGENTA}$line${NC}"
        elif echo "$line" | grep -qiE "success|completed|finished"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -qiE "warning|warn"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qiE "starting|running|task"; then
            echo -e "${BLUE}$line${NC}"
        elif echo "$line" | grep -qiE "\[scicode\]"; then
            echo -e "${CYAN}$line${NC}"
        elif echo "$line" | grep -qiE "\[corebench\]"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -qiE "\[colbench\]"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qiE "\[scienceagentbench\]|\[sab\]"; then
            echo -e "${MAGENTA}$line${NC}"
        else
            echo "$line"
        fi
    done
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
            tail -f $LOG_FILES 2>/dev/null | grep -iE --line-buffered "$FILTER" | colorize
        else
            tail -f $LOG_FILES 2>/dev/null | colorize
        fi

        # If tail exits, wait and retry
        sleep 2
    done
}

watch_logs
