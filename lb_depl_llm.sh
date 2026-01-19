#!/bin/bash
# LLM API Load Balanced Cluster Deployment
# Starts 20 LiteLLM backends + nginx load balancer (21 ports total)
#
# Usage: bash lb_depl_llm.sh              # Uses ports 5000-5020
#        LB_PORT=6000 bash lb_depl_llm.sh # Uses ports 6000-6020
#
# Default: port 5000 (LB) + ports 5001-5020 (20 backends)

set -e

LB_PORT=${LB_PORT:-5000}
START_PORT=$((LB_PORT + 1))  # Backends automatically start at LB_PORT + 1
NUM_INSTANCES=${NUM_INSTANCES:-20}
WORKSPACE=~/api
CONDA_ENV=deploy
NGINX_CONF="$WORKSPACE/deploy/configs/nginx/llm_load_balancer.conf"

echo "=== Deploying Load Balanced LLM API Cluster ==="
echo "  Load Balancer: port $LB_PORT"
echo "  Backends: ports $START_PORT - $((START_PORT + NUM_INSTANCES - 1))"
echo ""

# 1. Check and install dependencies
command -v git &>/dev/null || { sudo apt-get update && sudo apt-get install -y git; }
command -v tmux &>/dev/null || sudo apt-get install -y tmux
command -v nginx &>/dev/null || sudo apt-get install -y nginx
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

# 5. Generate nginx config dynamically (port-specific files)
NGINX_RUNTIME_CONF="/tmp/llm_lb_nginx_${LB_PORT}.conf"
NGINX_ERROR_LOG="/tmp/llm_lb_nginx_${LB_PORT}_error.log"
NGINX_ACCESS_LOG="/tmp/llm_lb_nginx_${LB_PORT}_access.log"
NGINX_PID="/tmp/llm_lb_nginx_${LB_PORT}.pid"

cat > "$NGINX_RUNTIME_CONF" << NGINX_EOF
worker_processes auto;
error_log $NGINX_ERROR_LOG warn;
pid $NGINX_PID;

events {
    worker_connections 1024;
}

http {
    access_log $NGINX_ACCESS_LOG;

    upstream llm_backends_${LB_PORT} {
        least_conn;
NGINX_EOF

# Add backend servers
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    echo "        server 127.0.0.1:$PORT weight=1 max_fails=3 fail_timeout=10s;" >> "$NGINX_RUNTIME_CONF"
done

cat >> "$NGINX_RUNTIME_CONF" << NGINX_EOF
        keepalive 32;
    }

    server {
        listen $LB_PORT;
        server_name localhost;

        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        proxy_buffering off;
        proxy_request_buffering off;
        chunked_transfer_encoding on;

        location / {
            proxy_pass http://llm_backends_${LB_PORT};
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header Connection "";
            proxy_set_header Accept-Encoding "";
            proxy_cache off;
        }

        location /health {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }
}
NGINX_EOF

echo "Generated nginx config: $NGINX_RUNTIME_CONF"

# 6. Stop existing nginx and sessions for this port
echo ""
echo "Stopping existing instances on port $LB_PORT..."
# Kill any nginx using this port's config
pkill -f "nginx.*llm_lb_nginx_${LB_PORT}" 2>/dev/null || true
# Also kill any process using the LB port directly
lsof -ti :$LB_PORT | xargs kill -9 2>/dev/null || true
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    tmux kill-session -t "llm_api_$PORT" 2>/dev/null || true
    tmux kill-session -t "llm_watchdog_$PORT" 2>/dev/null || true
done
sleep 2

# 7. Start backend instances
echo ""
echo "Starting $NUM_INSTANCES backend instances..."

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    WATCHDOG_SESSION="llm_watchdog_$PORT"

    tmux new-session -d -s "$WATCHDOG_SESSION" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        cd $WORKSPACE && source .venv/bin/activate
        export PATH=$WORKSPACE/deploy/helper_scripts/bin:\$PATH
        CHECK_INTERVAL=5 MAX_FAILURES=3 HEALTH_TIMEOUT=5 STARTUP_WAIT=20 \
        llm_watchdog.sh $PORT
        bash
    "
    echo "  Started backend on port $PORT"
done

# 8. Wait for backends to start
echo ""
echo "Waiting for backends to start (30s)..."
sleep 30

# 9. Start nginx load balancer
echo ""
echo "Starting nginx load balancer on port $LB_PORT..."
nginx -c "$NGINX_RUNTIME_CONF"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Load Balancer Entry Point:"
echo "  curl http://localhost:$LB_PORT/v1/models -H 'Authorization: Bearer sk-1234'"
echo ""
echo "Backend Instances (ports $START_PORT-$((START_PORT + NUM_INSTANCES - 1))):"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((START_PORT + i))
    echo "  Port $PORT: tmux attach -t llm_watchdog_$PORT"
done
echo ""
echo "Logs:"
echo "  Nginx access: tail -f $NGINX_ACCESS_LOG"
echo "  Nginx error:  tail -f $NGINX_ERROR_LOG"
echo ""
echo "Stop everything:"
echo "  pkill -f 'nginx.*llm_lb_nginx_${LB_PORT}'; for port in \$(seq $START_PORT $((START_PORT + NUM_INSTANCES - 1))); do tmux kill-session -t llm_api_\$port 2>/dev/null; tmux kill-session -t llm_watchdog_\$port 2>/dev/null; done"
