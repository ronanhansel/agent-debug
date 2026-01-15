#!/bin/bash
# Entrypoint stub for capsule-9240688
# This script provides a minimal entrypoint when run.sh is missing from the repository.

set -e

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate base 2>/dev/null || true
fi

# Set R library path if R is used
export R_LIBS_USER="${R_LIBS_USER:-./.R/library}"
mkdir -p "$R_LIBS_USER"

# Execute the main script(s) - adjust based on actual repository structure
# Look for common entrypoints
if [ -f "main.py" ]; then
    python main.py "$@"
elif [ -f "run.py" ]; then
    python run.py "$@"
elif [ -f "main.R" ]; then
    Rscript main.R "$@"
elif [ -f "run.R" ]; then
    Rscript run.R "$@"
else
    echo "No recognized entrypoint found. Please specify the command to run."
    exit 1
fi
