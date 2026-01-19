#!/bin/bash
# LLM API Deployment Script with Watchdog (Auto-restart on failure)
# Usage: bash deploy_llm_watchdog.sh
# Env: PORT=4001 bash deploy_llm_watchdog.sh
#
# This script deploys the LiteLLM proxy with a watchdog that monitors
# health and automatically restarts the service if it becomes unresponsive.

set -e

PORT=${PORT:-4000}
WORKSPACE=~/api
CONDA_ENV=deploy
SESSION_NAME="llm_api_$PORT"
WATCHDOG_SESSION="llm_watchdog_$PORT"

echo "=== Deploying LLM API with Watchdog on port $PORT ==="

# 1. Check and install dependencies
command -v git &>/dev/null || { sudo apt-get update && sudo apt-get install -y git; }
command -v tmux &>/dev/null || sudo apt-get install -y tmux
command -v az &>/dev/null || { curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash; }

if [ ! -f ~/miniconda3/bin/conda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p ~/miniconda3 && rm /tmp/mc.sh
fi
source ~/miniconda3/etc/profile.d/conda.sh

command -v uv &>/dev/null || pip install uv

# 2. Setup workspace and clone deploy repo
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"
[ -d "deploy" ] || git clone https://github.com/you-n-g/deploy.git

# 3. Setup conda env
conda env list | grep -q "^$CONDA_ENV " || conda create -n "$CONDA_ENV" python=3.10 pip -y
conda activate "$CONDA_ENV"

# 4. Setup venv and dependencies
if [ ! -d ".venv" ]; then
    uv venv && source .venv/bin/activate
    uv pip install 'litellm[proxy]'
    uv pip install git+https://github.com/you-n-g/litellm@support_gpt_5 --upgrade
else
    source .venv/bin/activate
fi

# 5. Kill existing sessions
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true

# 6. Start the watchdog (which will start the proxy)
tmux new-session -d -s "$WATCHDOG_SESSION" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd $WORKSPACE && source .venv/bin/activate
    export PATH=$WORKSPACE/deploy/helper_scripts/bin:\$PATH
    # Run watchdog with configurable parameters
    CHECK_INTERVAL=${CHECK_INTERVAL:-5} \
    MAX_FAILURES=${MAX_FAILURES:-3} \
    HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-5} \
    STARTUP_WAIT=${STARTUP_WAIT:-20} \
    DEFAULT_COOLDOWN=${DEFAULT_COOLDOWN:-60} \
    llm_watchdog.sh $PORT
    bash
"

echo "=== Done! ==="
echo ""
echo "Test: curl http://localhost:$PORT/v1/models -H 'Authorization: Bearer sk-1234'"
echo ""
echo "Watchdog Logs: tmux attach -t $WATCHDOG_SESSION"
echo "Proxy Logs:    tmux attach -t $SESSION_NAME"
echo ""
echo "Stop Watchdog: tmux kill-session -t $WATCHDOG_SESSION"
echo "Stop Proxy:    tmux kill-session -t $SESSION_NAME"
echo "Stop Both:     tmux kill-session -t $WATCHDOG_SESSION; tmux kill-session -t $SESSION_NAME"
echo ""
echo "Watchdog Settings (set via env vars):"
echo "  CHECK_INTERVAL=5       # Health check interval in seconds"
echo "  MAX_FAILURES=3         # Failures before action (15s total)"
echo "  HEALTH_TIMEOUT=5       # Timeout for each health check"
echo "  STARTUP_WAIT=20        # Wait time after starting proxy"
echo "  DEFAULT_COOLDOWN=60    # Fallback wait if can't parse 429 retry time"
echo ""
echo "Behavior:"
echo "  - 429 rate limit: waits exact time from error message, no restart"
echo "  - Other failures: restarts immediately, no delay"
