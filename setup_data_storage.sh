#!/bin/bash

# Complete setup to move all heavy storage to /Data
# This allows running 10+ parallel Docker tasks without filling /

DATA_BASE="/Data/home/v-qizhengli"

echo "=== Setting up /Data for heavy workloads ==="

# 1. Create directories on /Data
echo "Creating directories on /Data..."
mkdir -p $DATA_BASE/tmp
mkdir -p $DATA_BASE/.cache
mkdir -p $DATA_BASE/docker_tmp
mkdir -p /Data/docker

# 2. Add to bashrc for persistent env vars
BASHRC="$DATA_BASE/.bashrc"
if ! grep -q "TMPDIR=/Data" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# Redirect temp directories to /Data (added by setup_data_storage.sh)" >> "$BASHRC"
    echo "export TMPDIR=$DATA_BASE/tmp" >> "$BASHRC"
    echo "export TEMP=$DATA_BASE/tmp" >> "$BASHRC"
    echo "export TMP=$DATA_BASE/tmp" >> "$BASHRC"
    echo "export XDG_CACHE_HOME=$DATA_BASE/.cache" >> "$BASHRC"
    echo "export DOCKER_TMPDIR=$DATA_BASE/docker_tmp" >> "$BASHRC"
    echo "Added env vars to $BASHRC"
else
    echo "Env vars already in $BASHRC"
fi

# 3. Export for current session
export TMPDIR=$DATA_BASE/tmp
export TEMP=$DATA_BASE/tmp
export TMP=$DATA_BASE/tmp
export XDG_CACHE_HOME=$DATA_BASE/.cache
export DOCKER_TMPDIR=$DATA_BASE/docker_tmp

echo ""
echo "=== Current session configured ==="
echo "TMPDIR=$TMPDIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo ""
echo "=== Next steps ==="
echo ""
echo "1. URGENT - Free space immediately (run in another terminal):"
echo "   docker stop \$(docker ps -q)"
echo "   docker container prune -f"
echo "   docker image prune -af"
echo "   docker builder prune -af"
echo "   docker system prune -af --volumes"
echo ""
echo "2. Move Docker data root (requires sudo):"
echo "   sudo bash $PWD/move_docker_to_data.sh"
echo ""
echo "3. Reload your shell:"
echo "   source ~/.bashrc"
echo ""
echo "4. Verify disk space:"
echo "   df -h / /Data"
echo ""
echo "After these steps, you can run parallel tasks like:"
echo "   ./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py \\"
echo "       --benchmark scicode --all-configs --all-tasks \\"
echo "       --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 10"
