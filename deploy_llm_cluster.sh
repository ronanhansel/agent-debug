#!/bin/bash
# LLM API Cluster Deployment Script
# Starts 10 LiteLLM proxy instances on ports 4000-4009
#
# Usage: bash deploy_llm_cluster.sh
# Env: START_PORT=5000 NUM_INSTANCES=5 bash deploy_llm_cluster.sh

set -e

START_PORT=${START_PORT:-4000}
NUM_INSTANCES=${NUM_INSTANCES:-10}
WORKSPACE=~/api
CONDA_ENV=deploy

echo "=== Deploying LLM API Cluster: $NUM_INSTANCES instances starting at port $START_PORT ==="

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

# 5. Start instances
echo ""
echo "Starting $NUM_INSTANCES instances..."
echo ""

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    SESSION_NAME="llm_api_$PORT"
    WATCHDOG_SESSION="llm_watchdog_$PORT"

    # Kill existing sessions
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true

    # Start watchdog (which starts the proxy)
    tmux new-session -d -s "$WATCHDOG_SESSION" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        cd $WORKSPACE && source .venv/bin/activate
        export PATH=$WORKSPACE/deploy/helper_scripts/bin:\$PATH
        CHECK_INTERVAL=5 MAX_FAILURES=3 HEALTH_TIMEOUT=5 STARTUP_WAIT=20 \
        llm_watchdog.sh $PORT
        bash
    "

    echo "  Started instance $i on port $PORT (sessions: $SESSION_NAME, $WATCHDOG_SESSION)"
done

echo ""
echo "=== Done! ==="
echo ""
echo "All instances:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    echo "  Port $PORT: curl http://localhost:$PORT/v1/models -H 'Authorization: Bearer sk-1234'"
done
echo ""
echo "View logs:"
echo "  tmux attach -t llm_api_400X      # Proxy logs"
echo "  tmux attach -t llm_watchdog_400X # Watchdog logs"
echo ""
echo "Stop all:"
echo "  for i in \$(seq 0 9); do tmux kill-session -t llm_api_400\$i 2>/dev/null; tmux kill-session -t llm_watchdog_400\$i 2>/dev/null; done"
echo ""
echo "Load balance example (nginx or application-level):"
echo "  Ports: $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
