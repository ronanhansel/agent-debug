#!/bin/bash
# kill_all.sh - Kill all remaining processes from HAL evaluation

echo "=== Killing HAL Evaluation Processes ==="

# Kill Docker containers related to evaluation
echo "[1/5] Stopping Docker containers..."
docker ps -q --filter "name=agent-env" | xargs -r docker stop 2>/dev/null
docker ps -q --filter "name=hal" | xargs -r docker stop 2>/dev/null
docker ps -q --filter "name=benchmark" | xargs -r docker stop 2>/dev/null
docker ps -q --filter "ancestor=agent-env" | xargs -r docker stop 2>/dev/null

# Kill hal-eval processes
echo "[2/5] Killing hal-eval processes..."
pkill -f "hal-eval" 2>/dev/null
pkill -f "hal_eval" 2>/dev/null

# Kill Python processes related to evaluation scripts
echo "[3/5] Killing evaluation scripts..."
pkill -f "eval_rubric.py" 2>/dev/null
pkill -f "fixing_pipeline.py" 2>/dev/null
pkill -f "claude_fixer" 2>/dev/null
pkill -f "run_.*_fixes.py" 2>/dev/null
pkill -f "judge.py" 2>/dev/null
pkill -f "pipeline.py" 2>/dev/null

# Kill agent processes
echo "[4/5] Killing agent processes..."
pkill -f "scicode_tool_calling_agent" 2>/dev/null
pkill -f "hal_generalist_agent" 2>/dev/null
pkill -f "SWE-agent" 2>/dev/null
pkill -f "assistantbench_browser_agent" 2>/dev/null
pkill -f "colbench_example_agent" 2>/dev/null
pkill -f "smolagents" 2>/dev/null

# Kill any remaining Docker containers (aggressive)
echo "[5/5] Force removing stuck containers..."
docker ps -aq --filter "status=running" --filter "name=agent" | xargs -r docker kill 2>/dev/null
docker ps -aq --filter "status=running" --filter "name=env-" | xargs -r docker kill 2>/dev/null

# Show remaining processes (for verification)
echo ""
echo "=== Remaining Docker Containers ==="
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" 2>/dev/null || echo "None"

echo ""
echo "=== Done ==="
