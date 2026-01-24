#!/bin/bash
# kill_all.sh - Kill all remaining processes from HAL evaluation

echo "=== Killing HAL Evaluation Processes ==="

DOCKER_CMD_TIMEOUT="${DOCKER_CMD_TIMEOUT:-8}"
HAL_KILL_ALL_DOCKER="${HAL_KILL_ALL_DOCKER:-0}"

docker_cmd() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$DOCKER_CMD_TIMEOUT" docker "$@"
    else
        docker "$@"
    fi
}

resolve_docker_host() {
    if [ -z "${DOCKER_HOST:-}" ] && [ -n "${HAL_DOCKER_HOST:-}" ]; then
        export DOCKER_HOST="$HAL_DOCKER_HOST"
    fi
    if [ -z "${DOCKER_HOST:-}" ] && [ -S "/run/user/$UID/docker.sock" ]; then
        export DOCKER_HOST="unix:///run/user/$UID/docker.sock"
    fi
}

docker_ready=false
if command -v docker >/dev/null 2>&1; then
    resolve_docker_host
    if docker_cmd info >/dev/null 2>&1; then
        docker_ready=true
    else
        echo "WARN: Docker daemon not reachable; skipping Docker cleanup."
    fi
else
    echo "WARN: Docker CLI not found; skipping Docker cleanup."
fi

# Kill Docker containers related to evaluation
echo "[1/5] Stopping Docker containers..."
if $docker_ready; then
    if [ "$HAL_KILL_ALL_DOCKER" = "1" ]; then
        echo "HAL_KILL_ALL_DOCKER=1: stopping all running containers..."
        docker_cmd ps -q | xargs -r docker stop 2>/dev/null
    else
        docker_cmd ps -q --filter "name=agentrun" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "name=agentpool" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "name=agent-env" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "name=agentpreflight" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "name=hal" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "name=benchmark" | xargs -r docker stop 2>/dev/null
        docker_cmd ps -q --filter "ancestor=hal-agent-runner" | xargs -r docker stop 2>/dev/null
    fi
fi

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
if $docker_ready; then
    if [ "$HAL_KILL_ALL_DOCKER" = "1" ]; then
        docker_cmd ps -q | xargs -r docker kill 2>/dev/null
    else
        docker_cmd ps -q --filter "name=agent" | xargs -r docker kill 2>/dev/null
        docker_cmd ps -q --filter "name=env-" | xargs -r docker kill 2>/dev/null
        docker_cmd ps -q --filter "ancestor=hal-agent-runner" | xargs -r docker kill 2>/dev/null
    fi
fi

# Show remaining processes (for verification)
echo ""
echo "=== Remaining Docker Containers ==="
if $docker_ready; then
    docker_cmd ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" 2>/dev/null || echo "None"
else
    echo "Docker unavailable"
fi

echo ""
echo "=== Done ==="
