#!/bin/bash
# LLM API Deployment Script (Port 4000)
# Usage: bash deploy_api_4000.sh
# Env: PORT=4001 bash deploy_api_4000.sh

set -e

PORT=${PORT:-4000}
WORKSPACE=~/api
CONDA_ENV=deploy
SESSION_NAME="llm_api_$PORT"

echo "=== Deploying LLM API on port $PORT ==="

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

# 5. Start service in tmux
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd $WORKSPACE && source .venv/bin/activate
    export PATH=$WORKSPACE/deploy/helper_scripts/bin:\$PATH
    llm_proxy.sh $PORT
    bash
"

echo "=== Done! ==="
echo "Test: curl http://localhost:$PORT/v1/models"
echo "Logs: tmux attach -t $SESSION_NAME"
echo "Stop: tmux kill-session -t $SESSION_NAME"