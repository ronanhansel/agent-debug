#!/bin/bash
# Complete wrapper to run benchmarks using /Data for temporary storage
# This prevents filling up the root partition

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HAL_REPO_ROOT:-$SCRIPT_DIR}"
WORKDIR="${HAL_WORKDIR:-$REPO_ROOT}"
DATA_ROOT="${HAL_DATA_ROOT:-/Data}"
DATA_NAMESPACE="${HAL_DATA_NAMESPACE:-$USER}"
DATA_RUN_ROOT="${HAL_DATA_RUN_ROOT:-$DATA_ROOT/hal_runs/$DATA_NAMESPACE/$(basename "$REPO_ROOT")}"

# Set working directory
cd "$WORKDIR"

# Configure all temp directories to use /Data
export TMPDIR="${HAL_TMPDIR:-$DATA_RUN_ROOT/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHON_TEMPDIR="$TMPDIR"
export DOCKER_TMPDIR="${HAL_DOCKER_TMPDIR:-$TMPDIR/docker}"
export HAL_RESULTS_DIR="${HAL_RESULTS_DIR:-$DATA_RUN_ROOT/results}"
export HAL_TRACES_DIR="${HAL_TRACES_DIR:-$DATA_RUN_ROOT/traces}"
export HAL_TMP_DIR="${HAL_TMP_DIR:-$DATA_RUN_ROOT/tmp}"

# Ensure temp directory exists
mkdir -p "$TMPDIR" "$DOCKER_TMPDIR" "$HAL_RESULTS_DIR" "$HAL_TRACES_DIR" "$HAL_TMP_DIR"

link_dir() {
    local name="$1"
    local target="$DATA_RUN_ROOT/$name"
    mkdir -p "$target"
    if [ -L "$WORKDIR/$name" ]; then
        return
    fi
    if [ -e "$WORKDIR/$name" ]; then
        echo "WARN: $WORKDIR/$name exists and is not a symlink. Move it to $target or delete it to enable linking."
        return
    fi
    ln -s "$target" "$WORKDIR/$name"
}

if [ "${HAL_LINK_DATA_DIRS:-1}" != "0" ]; then
    link_dir "results"
    link_dir "traces"
    link_dir ".tmp"
    link_dir "log"
    link_dir "logs"
    link_dir "output"
fi

# Show current disk usage
echo "=========================================="
echo "Disk Usage Status:"
echo "=========================================="
echo "Root partition:"
df -h / | tail -1
echo ""
echo "/Data partition:"
df -h /Data | tail -1
echo ""
echo "Temp directory: $TMPDIR"
echo "Available for temp files: $(df -h $TMPDIR | tail -1 | awk '{print $4}')"
echo ""
echo "Docker usage:"
docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}"
echo "=========================================="
echo ""

# Check if we have enough space
ROOT_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $ROOT_USAGE -ge 95 ]; then
    echo "⚠️  WARNING: Root partition is ${ROOT_USAGE}% full!"
    echo "⚠️  Consider running ./cleanup_docker.sh first"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please run ./cleanup_docker.sh to free space."
        exit 1
    fi
fi

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Example:"
    echo "  $0 python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 25"
    echo ""
    exit 1
fi

# Run the command
echo "Starting command with /Data temporary storage..."
echo "Command: $@"
echo ""

exec "$@"
