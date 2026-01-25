#!/bin/bash
#
# Watch All - benchmark progress and logs (current run only)
#
# Usage:
#   ./watch_all.sh [--batch-mode] [--prefix PREFIX] [track_run_progress args...]
#   ./watch_all.sh logs
#
# Modes:
#   (default) Aggregate progress (same as previous agents view)
#   logs      Tail logs for the latest run only
#
# Batch mode:
#   - If finished tasks do not change for >7 minutes, launch next run after 5 minutes.
#   - If finished tasks == total tasks, launch next run after 5 minutes.
#   - Next prefix is computed by incrementing the current prefix (e.g., sun12_ -> sun13_).
#   - Next run command is previewed in the watcher.
#

show_help() {
    cat <<'EOF'
Usage:
  ./watch_all.sh [--batch-mode] [--prefix PREFIX] [track_run_progress args...]
  ./watch_all.sh logs

Modes:
  (default) Aggregate progress (same as previous agents view)
  logs      Tail logs for the latest run only

Options:
  --batch-mode     Auto-start the next colbench run if progress stalls or completes.
  --prefix PREFIX  Current run prefix (e.g., sun12_) used to compute the next prefix.
EOF
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="agents"
BATCH_MODE="false"
PREFIX=""
TRACK_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        logs)
            MODE="logs"
            shift
            ;;
        --batch-mode)
            BATCH_MODE="true"
            shift
            ;;
        --prefix)
            PREFIX="${2:-}"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            TRACK_ARGS+=("$1")
            shift
            ;;
    esac
done

# Colors (log viewer only)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
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
if [ -n "${HAL_RESULTS_DIR:-}" ] && [ -d "$HAL_RESULTS_DIR" ]; then
    RESULTS_DIR="$HAL_RESULTS_DIR"
elif [ -d "$SCRIPT_DIR/.hal_data/results" ]; then
    RESULTS_DIR="$SCRIPT_DIR/.hal_data/results"
fi

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
    local runs=()
    local sorted=()
    shopt -s nullglob
    runs=("$LOGS_DIR"/benchmark_run_*)
    shopt -u nullglob
    if [ ${#runs[@]} -eq 0 ]; then
        return
    fi
    IFS=$'\n' sorted=($(printf '%s\n' "${runs[@]}" | sort -r))
    unset IFS
    printf "%s\n" "${sorted[0]}"
}

get_latest_run_id() {
    local latest_run_dir
    latest_run_dir="$(get_latest_run_dir)"
    [ -n "$latest_run_dir" ] && basename "$latest_run_dir" | sed 's/^benchmark_run_//'
}

# Format and colorize function for log tailing
format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v white="$WHITE" -v bold_green="$BOLD_GREEN" -v nc="$NC" '
    BEGIN {
        current_run_id = ""
    }
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
    local run_id="$1"
    local latest_run_dir="$2"
    local all_logs=""

    if [ -n "$latest_run_dir" ]; then
        for log in "$latest_run_dir"/*.log; do
            [ -f "$log" ] && all_logs="$all_logs $log"
        done
    fi

    if [ -n "$run_id" ]; then
        shopt -s nullglob
        for benchmark_dir in "$RESULTS_DIR"/*/; do
            [ -d "$benchmark_dir" ] || continue
            for run_dir in "$benchmark_dir"*_"$run_id"/; do
                [ -d "$run_dir" ] || continue
                for log in "$run_dir"/*_verbose.log; do
                    [ -f "$log" ] && all_logs="$all_logs $log"
                done
            done
        done
        shopt -u nullglob
    fi

    echo "$all_logs"
}

watch_logs() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}           LOG VIEWER MODE (LATEST RUN ONLY)${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${BLUE}Results:${NC} $RESULTS_DIR"
    echo -e "${BLUE}Logs:${NC} $LOGS_DIR"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    while true; do
        local latest_run_dir
        local run_id
        latest_run_dir="$(get_latest_run_dir)"
        run_id="$(get_latest_run_id)"

        echo -e "${BLUE}Run ID:${NC} ${run_id:-none}"
        LOG_FILES=$(collect_logs "$run_id" "$latest_run_dir")

        if [ -z "$LOG_FILES" ]; then
            echo -e "${YELLOW}No log files found for the latest run. Waiting...${NC}"
            sleep 5
            continue
        fi

        LOG_COUNT=$(echo $LOG_FILES | wc -w)
        echo -e "${CYAN}Watching $LOG_COUNT log files...${NC}"

        tail -f $LOG_FILES 2>/dev/null | format_and_colorize
        sleep 2
    done
}

arg_present() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

case "$MODE" in
    logs)
        watch_logs
        ;;
    *)
        if ! arg_present "--watch" "${TRACK_ARGS[@]}"; then
            TRACK_ARGS+=(--watch)
        fi
        if ! arg_present "--interval" "${TRACK_ARGS[@]}"; then
            TRACK_ARGS+=(--interval 2)
        fi
        if [ "$BATCH_MODE" = "true" ]; then
            TRACK_ARGS+=(--batch-mode)
        fi
        if [ -n "$PREFIX" ]; then
            TRACK_ARGS+=(--prefix "$PREFIX")
        fi
        python3 "$SCRIPT_DIR/scripts/track_run_progress.py" "${TRACK_ARGS[@]}"
        ;;
esac
