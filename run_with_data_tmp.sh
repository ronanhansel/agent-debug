#!/bin/bash
# Wrapper script to run benchmark fixes with /Data as temp storage
# This prevents filling up the root partition

# Set all temporary directories to /Data
export TMPDIR="/Data/home/v-qizhengli/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Python-specific temp directory
export PYTHON_TEMPDIR="$TMPDIR"

# Docker build cache (if you had sudo, this would be in daemon.json)
# For now, we can only control where Python creates temp dirs
export DOCKER_TMPDIR="$TMPDIR"

# Ensure the directory exists
mkdir -p "$TMPDIR"

echo "=========================================="
echo "Temporary file configuration:"
echo "  TMPDIR: $TMPDIR"
echo "  Available space: $(df -h $TMPDIR | tail -1 | awk '{print $4}')"
echo "  Root partition: $(df -h / | tail -1 | awk '{print $5 " used"}')"
echo "=========================================="
echo

# Run the command with all arguments passed to this script
exec "$@"
