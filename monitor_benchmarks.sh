#!/bin/bash
#
# Benchmark Monitor - Watch progress of running benchmarks
#
# Usage: ./monitor_benchmarks.sh [log_dir]
# Example: ./monitor_benchmarks.sh logs/benchmark_run_20260122_120000
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the most recent log directory if not specified
if [ -z "$1" ]; then
    LOG_DIR=$(ls -td "$SCRIPT_DIR/logs/benchmark_run_"* 2>/dev/null | head -1)
    if [ -z "$LOG_DIR" ]; then
        echo "No benchmark runs found. Start a run with ./run_all_benchmarks.sh first."
        exit 1
    fi
else
    LOG_DIR="$1"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}           BENCHMARK MONITOR${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Monitoring: $LOG_DIR${NC}"
echo -e "${CYAN}Press Ctrl+C to exit monitor (benchmarks will continue)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

while true; do
    clear
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  BENCHMARK STATUS - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}  Log Dir: $LOG_DIR${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    for benchmark in "${BENCHMARKS[@]}"; do
        pid_file="$LOG_DIR/${benchmark}.pid"
        exit_code_file="$LOG_DIR/${benchmark}.exit_code"
        log_file="$LOG_DIR/${benchmark}.log"

        echo -e "${BLUE}=== $benchmark ===${NC}"

        # Status
        if [ -f "$exit_code_file" ]; then
            exit_code=$(cat "$exit_code_file")
            if [ "$exit_code" -eq 0 ]; then
                echo -e "  Status: ${GREEN}COMPLETED${NC}"
            else
                echo -e "  Status: ${RED}FAILED (exit code: $exit_code)${NC}"
            fi
        elif [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "  Status: ${YELLOW}RUNNING${NC} (PID: $pid)"
            else
                echo -e "  Status: ${RED}STOPPED${NC} (process died)"
            fi
        else
            echo -e "  Status: ${YELLOW}NOT STARTED${NC}"
        fi

        # Log stats
        if [ -f "$log_file" ]; then
            lines=$(wc -l < "$log_file")
            size=$(du -h "$log_file" | cut -f1)
            echo -e "  Log: $lines lines, $size"

            # Count successes and errors
            success_count=$(grep -ciE "success|completed" "$log_file" 2>/dev/null || echo "0")
            error_count=$(grep -ciE "error|exception|failed" "$log_file" 2>/dev/null || echo "0")
            task_count=$(grep -ciE "\[main\].*SUCCESS" "$log_file" 2>/dev/null || echo "0")

            echo -e "  Tasks completed: ${GREEN}$task_count${NC}, Errors: ${RED}$error_count${NC}"

            # Last activity
            last_line=$(grep -iE "task|\[hal\]|\[main\]|success|error" "$log_file" 2>/dev/null | tail -1)
            if [ -n "$last_line" ]; then
                echo -e "  Last: ${last_line:0:80}..."
            fi
        fi

        echo ""
    done

    # Docker stats
    echo -e "${CYAN}=== Docker Status ===${NC}"
    container_count=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -c "agentrun" || echo "0")
    echo -e "  Active containers: $container_count"

    # Show container names (first 5)
    if [ "$container_count" -gt 0 ]; then
        echo "  Recent containers:"
        docker ps --format "    {{.Names}} ({{.Status}})" 2>/dev/null | grep agentrun | head -5
    fi

    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}Refreshing in 30 seconds... (Ctrl+C to exit)${NC}"

    sleep 30
done
