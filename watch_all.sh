#!/bin/bash
#
# Watch All - Real-time benchmark progress dashboard
#
# Usage: ./watch_all.sh [mode]
# Modes:
#   (default) - Dashboard showing task progress per benchmark
#   logs      - Real-time log tailing (benchmark runs + verbose)
#   errors    - Only show errors from logs
#   api       - Only show API calls/responses from logs
#   agents    - Aggregate progress per benchmark (use --per-agent for details)
#   runs      - Benchmark run logs only (from benchmark_run_*)
#   verbose   - Per-task verbose logs only (*_verbose.log)
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-dashboard}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
BOLD='\033[1m'
BOLD_GREEN='\033[1;32m'
DIM='\033[2m'
NC='\033[0m'

RESULTS_DIR="$SCRIPT_DIR/results"
detect_run_root() {
    if [ -n "${DATA_PATH:-}" ] && [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$DATA_PATH/hal_runs/$namespace/$(basename "$SCRIPT_DIR")"
        return
    fi
    if [ -n "${HAL_DATA_ROOT:-}" ] && [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$HAL_DATA_ROOT/hal_runs/$namespace/$(basename "$SCRIPT_DIR")"
        return
    fi
    echo "$SCRIPT_DIR"
}

RUN_ROOT="$(detect_run_root)"
RESULTS_DIR="$RUN_ROOT/results"

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

LOGS_DIR="$(detect_logs_root)"

get_latest_run_dir() {
    ls -td "$LOGS_DIR"/benchmark_run_* 2>/dev/null | head -1
}

get_latest_run_id() {
    local latest_run_dir
    latest_run_dir="$(get_latest_run_dir)"
    [ -n "$latest_run_dir" ] && basename "$latest_run_dir" | sed 's/^benchmark_run_//'
}

# Benchmarks
BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")

# =============================================================================
# DASHBOARD MODE (default)
# =============================================================================

get_benchmark_status() {
    local benchmark=$1
    local log_dir="$LOGS_DIR"

    # Find the most recent benchmark run log directory
    local latest_run=$(ls -td "$log_dir"/benchmark_run_* 2>/dev/null | head -1)

    if [ -z "$latest_run" ]; then
        echo "NOT_STARTED|0|0|"
        return
    fi

    local log_file="$latest_run/${benchmark}.log"
    local pid_file="$latest_run/${benchmark}.pid"
    local exit_code_file="$latest_run/${benchmark}.exit_code"

    # Check if completed
    if [ -f "$exit_code_file" ]; then
        local exit_code=$(cat "$exit_code_file")
        if [ "$exit_code" -eq 0 ]; then
            local total=$(grep -oE '\[[0-9]+/[0-9]+\]' "$log_file" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+' | cut -d'/' -f2)
            [ -z "$total" ] && total="?"
            echo "COMPLETED|$total|$total|"
        else
            echo "FAILED|?|?|exit_code=$exit_code"
        fi
        return
    fi

    # Check if running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            local progress=$(grep -oE '\[[0-9]+/[0-9]+\]' "$log_file" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+')
            if [ -n "$progress" ]; then
                local done=$(echo "$progress" | cut -d'/' -f1)
                local total=$(echo "$progress" | cut -d'/' -f2)
                local last_activity=$(grep -iE "SUCCESS|FAILED" "$log_file" 2>/dev/null | tail -1 | head -c 70)
                echo "RUNNING|$done|$total|$last_activity"
            else
                local success_count=$(grep -c "SUCCESS" "$log_file" 2>/dev/null || echo 0)
                echo "RUNNING|$success_count|?|Initializing..."
            fi
        else
            echo "STOPPED|?|?|Process terminated"
        fi
        return
    fi

    echo "NOT_STARTED|0|0|"
}

draw_progress_bar() {
    local done=$1
    local total=$2
    local width=30

    if [ "$total" = "?" ] || [ "$total" -eq 0 ] 2>/dev/null; then
        printf "[%-${width}s]" ""
        return
    fi

    local filled=$((done * width / total))
    local empty=$((width - filled))

    printf "["
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "]"
}

print_dashboard() {
    clear

    echo -e "${CYAN}+========================================================================+${NC}"
    echo -e "${CYAN}|${NC}${BOLD}              BENCHMARK EVALUATION DASHBOARD${NC}                           ${CYAN}|${NC}"
    echo -e "${CYAN}+========================================================================+${NC}"

    # Find latest run info
    local latest_run=$(ls -td "$LOGS_DIR"/benchmark_run_* 2>/dev/null | head -1)
    local run_name=$(basename "$latest_run" 2>/dev/null || echo "No active run")
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${CYAN}|${NC} ${DIM}Run:${NC} ${run_name}                              "
    echo -e "${CYAN}|${NC} ${DIM}Updated:${NC} ${timestamp}                              "
    echo -e "${CYAN}+------------------------------------------------------------------------+${NC}"

    local total_done=0
    local total_tasks=0
    local all_completed=true

    for benchmark in "${BENCHMARKS[@]}"; do
        local status_info=$(get_benchmark_status "$benchmark")
        local status=$(echo "$status_info" | cut -d'|' -f1)
        local done=$(echo "$status_info" | cut -d'|' -f2)
        local total=$(echo "$status_info" | cut -d'|' -f3)
        local detail=$(echo "$status_info" | cut -d'|' -f4-)

        # Format benchmark name (pad to 18 chars)
        local name_padded=$(printf "%-18s" "$benchmark")

        # Status color and icon
        local status_color=""
        local status_icon=""
        case "$status" in
            COMPLETED)
                status_color="$GREEN"
                status_icon="[OK]"
                ;;
            RUNNING)
                status_color="$BLUE"
                status_icon="[..]"
                all_completed=false
                ;;
            FAILED)
                status_color="$RED"
                status_icon="[XX]"
                ;;
            STOPPED)
                status_color="$YELLOW"
                status_icon="[!!]"
                ;;
            *)
                status_color="$DIM"
                status_icon="[  ]"
                all_completed=false
                ;;
        esac

        # Progress
        local progress_str=""
        if [ "$total" != "?" ] && [ "$total" -gt 0 ] 2>/dev/null; then
            progress_str=$(printf "%3s/%-3s" "$done" "$total")
            total_done=$((total_done + done))
            total_tasks=$((total_tasks + total))
        else
            progress_str="  ?/?  "
        fi

        # Progress bar
        local bar=$(draw_progress_bar "$done" "$total")

        echo -e "${CYAN}|${NC} ${status_color}${status_icon}${NC} ${name_padded} ${progress_str}  ${bar}"

        # Show latest activity if running
        if [ -n "$detail" ] && [ "$status" = "RUNNING" ]; then
            local short_detail=$(echo "$detail" | head -c 65)
            echo -e "${CYAN}|${NC}        ${DIM}-> ${short_detail}${NC}"
        fi
    done

    echo -e "${CYAN}+------------------------------------------------------------------------+${NC}"

    # Docker status
    local agent_containers=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -c "agentrun" || echo 0)
    local total_containers=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
    total_containers=${total_containers:-0}
    local build_containers=$((total_containers - agent_containers))
    [ "$build_containers" -lt 0 ] && build_containers=0
    local images_cached=$(docker images 2>/dev/null | grep -c "agent-env" || echo 0)

    echo -e "${CYAN}|${NC} ${DIM}Docker:${NC} ${agent_containers} agents, ${build_containers} builders, ${images_cached} images cached"

    # Total progress
    if [ "$total_tasks" -gt 0 ]; then
        local pct=$((total_done * 100 / total_tasks))
        echo -e "${CYAN}|${NC} ${DIM}Total:${NC}  ${total_done}/${total_tasks} tasks (${pct}%)"
    fi

    echo -e "${CYAN}+------------------------------------------------------------------------+${NC}"
    echo -e "${CYAN}|${NC} ${DIM}Press Ctrl+C to exit | Run './watch_all.sh logs' for detailed logs${NC}"
    echo -e "${CYAN}+========================================================================+${NC}"

    if $all_completed; then
        echo ""
        echo -e "${GREEN}${BOLD}All benchmarks completed!${NC}"
        return 1  # Signal to stop
    fi

    return 0
}

run_dashboard() {
    echo -e "${CYAN}Starting dashboard mode... (Ctrl+C to exit)${NC}"
    sleep 1

    while true; do
        if ! print_dashboard; then
            break
        fi
        sleep 5
    done
}

# =============================================================================
# LOG TAILING MODE
# =============================================================================

# Format and colorize function for log tailing
format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v white="$WHITE" -v bold_green="$BOLD_GREEN" -v nc="$NC" '
    BEGIN {
        current_run_id = ""
    }
    # Match tail -f file headers: ==> /path/to/benchmark/run_id/file.log <==
    /^==> .* <==/ {
        path = $2
        n = split(path, parts, "/")
        if (n >= 3) {
            current_benchmark = parts[n-2]
            current_run_id = parts[n-1]
        } else if (n >= 2) {
            current_benchmark = ""
            current_run_id = parts[n-1]
        } else {
            current_benchmark = ""
            current_run_id = path
        }
        next
    }
    /^$/ { next }
    {
        timestamp = strftime("%H:%M:%S")
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

        line = $0
        gsub(/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - [a-zA-Z_.]+ - (DEBUG|INFO|WARNING|ERROR) - /, "", line)

        if (line ~ /Results:.*\{/ || line ~ /"accuracy"/ || line ~ /"score"/ || \
            line ~ /Evaluation completed/ || line ~ /successful_tasks/ || line ~ /failed_tasks/) {
            printf "%s%s%s%s\n", bold_green, prefix, line, nc
        }
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

collect_logs() {
    local include_runs="$1"
    local include_verbose="$2"
    local all_logs=""
    local latest_run_dir
    local run_id

    latest_run_dir="$(get_latest_run_dir)"
    run_id="$(get_latest_run_id)"

    if [ "$include_runs" = "true" ]; then
        if [ -n "$latest_run_dir" ]; then
            for log in "$latest_run_dir"/*.log; do
                [ -f "$log" ] && all_logs="$all_logs $log"
            done
        fi
    fi

    if [ "$include_verbose" = "true" ]; then
        if [ -n "$run_id" ]; then
            for benchmark_dir in "$RESULTS_DIR"/*/; do
                if [ -d "$benchmark_dir" ]; then
                    local matching_run_dir
                    matching_run_dir=$(ls -td "$benchmark_dir"*_"$run_id"/ 2>/dev/null | head -1)
                    if [ -n "$matching_run_dir" ]; then
                        for log in "$matching_run_dir"/*_verbose.log; do
                            [ -f "$log" ] && all_logs="$all_logs $log"
                        done
                    fi
                fi
            done
        fi
    fi

    echo "$all_logs"
}

watch_logs() {
    local filter="$1"
    local include_runs="$2"
    local include_verbose="$3"

    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}           LOG VIEWER MODE${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${BLUE}Filter:${NC} ${filter:-all}"
    echo -e "${BLUE}Sources:${NC} runs=${include_runs:-true}, verbose=${include_verbose:-true}"
    echo -e "${BLUE}Results:${NC} $RESULTS_DIR"
    echo -e "${BLUE}Logs:${NC} $LOGS_DIR"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    while true; do
        LOG_FILES=$(collect_logs "${include_runs:-true}" "${include_verbose:-true}")

        if [ -z "$LOG_FILES" ]; then
            echo -e "${YELLOW}No log files found. Waiting...${NC}"
            sleep 5
            continue
        fi

        LOG_COUNT=$(echo $LOG_FILES | wc -w)
        echo -e "${CYAN}Watching $LOG_COUNT log files...${NC}"

        if [ -n "$filter" ]; then
            tail -f $LOG_FILES 2>/dev/null | grep -iE --line-buffered "^==> .* <==|$filter" | format_and_colorize
        else
            tail -f $LOG_FILES 2>/dev/null | format_and_colorize
        fi

        sleep 2
    done
}

# =============================================================================
# MAIN
# =============================================================================

case "$MODE" in
    logs|all)
        watch_logs "" "true" "true"
        ;;
    errors)
        watch_logs "error|exception|failed|traceback|401|403|429|500|502|503|504|timeout|unauthorized|denied" "true" "true"
        ;;
    api)
        watch_logs "openai|azure|api|request|response|token|model|gpt|o3|o4" "true" "true"
        ;;
    progress)
        watch_logs "task|starting|running|completed|success|finished|\[hal\]|\[main\]|SUCCESS|FAILED" "true" "true"
        ;;
    runs)
        watch_logs "" "true" "false"
        ;;
    verbose)
        watch_logs "" "false" "true"
        ;;
    agents)
        python3 "$SCRIPT_DIR/scripts/track_run_progress.py" --watch --interval 2
        ;;
    dashboard|*)
        run_dashboard
        ;;
esac
