#!/bin/bash
# Wrapper to run high-parallelism benchmarks without filling /
# Manages Docker cleanup automatically since Docker data-root can't be moved

set -e

# Ensure temp dirs are on /Data
export TMPDIR=/Data/home/v-qizhengli/tmp
export TMP=/Data/home/v-qizhengli/tmp
export TEMP=/Data/home/v-qizhengli/tmp
export XDG_CACHE_HOME=/Data/home/v-qizhengli/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# Thresholds
MIN_FREE_GB=50  # Minimum free space before cleanup
CLEANUP_INTERVAL=300  # Check every 5 minutes

cleanup_docker() {
    echo "[$(date '+%H:%M:%S')] Cleaning Docker..."
    docker container prune -f >/dev/null 2>&1 || true
    docker image prune -f >/dev/null 2>&1 || true
    local freed=$(df -h / | awk 'NR==2 {print $4}')
    echo "[$(date '+%H:%M:%S')] Free space on /: $freed"
}

monitor_and_cleanup() {
    while true; do
        sleep $CLEANUP_INTERVAL
        local free_gb=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
        if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
            echo ""
            echo "[$(date '+%H:%M:%S')] WARNING: Only ${free_gb}GB free, cleaning Docker..."
            cleanup_docker
        fi
    done
}

echo "========================================"
echo "High-Parallelism Benchmark Runner"
echo "========================================"
echo "TMPDIR: $TMPDIR"
echo "Docker: Still on / (no sudo), auto-cleanup enabled"
echo ""

# Initial cleanup
cleanup_docker

# Start background monitor
monitor_and_cleanup &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "Starting: $@"
echo "========================================"
echo ""

# Run the actual command
"$@"

# Final cleanup
echo ""
echo "========================================"
cleanup_docker
echo "Done!"
