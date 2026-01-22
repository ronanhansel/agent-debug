#!/bin/bash
# Complete wrapper to run benchmarks using /Data for temporary storage
# This prevents filling up the root partition

# Set working directory
cd /Data/home/v-qizhengli/workspace/agent-debug

# Configure all temp directories to use /Data
export TMPDIR="/Data/home/v-qizhengli/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHON_TEMPDIR="$TMPDIR"

# Ensure temp directory exists
mkdir -p "$TMPDIR"

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
