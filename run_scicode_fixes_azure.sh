#!/bin/bash
# Wrapper script to run SciCode fixes with Azure authentication
# Unsets proxy URLs from shell environment to ensure direct Azure access

# Clear any proxy URLs from shell environment
unset OPENAI_BASE_URL
unset OPENAI_API_BASE
unset OPENAI_API_BASE_URL
unset LITELLM_BASE_URL

echo "=================================="
echo "Running SciCode fixes with Azure"
echo "=================================="
echo ""

# Run the fix script with all arguments passed through
cd "$(dirname "$0")"
python scripts/run_scicode_fixes.py "$@"
