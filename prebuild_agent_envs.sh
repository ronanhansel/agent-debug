#!/bin/bash
# Pre-build ALL agent-env Docker images for HAL benchmarks
# Run this ONCE before starting any benchmark runs to avoid build contention

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_HARNESS="$SCRIPT_DIR/hal-harness"

echo "=========================================="
echo "Pre-building ALL Agent Environment Images"
echo "=========================================="
echo ""

# All agents used by benchmarks
AGENTS=(
    "hal_generalist_agent"      # Used by ALL benchmarks
    "scicode_tool_calling_agent" # scicode
    "sab_example_agent"          # scienceagentbench
    "core_agent"                 # corebench
    "colbench_example_agent"     # colbench
)

# Check/build base image first
echo "Step 1: Checking base image..."
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^hal-agent-runner:latest$"; then
    echo "  [OK] hal-agent-runner:latest exists"
else
    echo "  [BUILD] Building hal-agent-runner:latest..."
    docker build -t hal-agent-runner:latest "$HAL_HARNESS/hal/utils/docker/" || {
        echo "  [FAIL] Could not build base image!"
        exit 1
    }
    echo "  [DONE] Base image built"
fi
echo ""

# Build each agent-env
echo "Step 2: Building agent-env images..."
echo ""

build_agent_env() {
    local agent_dir="$1"
    local req_file="$HAL_HARNESS/agents/$agent_dir/requirements.txt"

    if [ ! -f "$req_file" ]; then
        echo "  [SKIP] $agent_dir - no requirements.txt"
        return 0
    fi

    local hash=$(md5sum "$req_file" | cut -c1-12)
    local tag="hal-agent-runner:agent-env-$hash"

    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${tag}$"; then
        echo "  [OK] $agent_dir ($tag)"
        return 0
    fi

    echo "  [BUILD] $agent_dir -> $tag"

    # Create temp Dockerfile
    local tmp_dockerfile=$(mktemp)
    cat > "$tmp_dockerfile" << 'EOF'
ARG BASE_IMAGE=hal-agent-runner:latest
FROM ${BASE_IMAGE}

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n agent_env python=3.11 && \
    conda run -n agent_env python -m pip install -U pip

COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9

WORKDIR /workspace
EOF

    local build_context="$HAL_HARNESS/agents/$agent_dir"

    # Show progress for long builds
    echo "    Installing $(wc -l < "$req_file") packages..."

    if docker build -t "$tag" -f "$tmp_dockerfile" "$build_context" 2>&1 | grep -E "^(Step|Successfully|ERROR)"; then
        echo "  [DONE] $agent_dir"
        rm -f "$tmp_dockerfile"
        return 0
    else
        echo "  [FAIL] $agent_dir - check Docker logs"
        rm -f "$tmp_dockerfile"
        return 1
    fi
}

failed=0
for agent in "${AGENTS[@]}"; do
    build_agent_env "$agent" || ((failed++))
done

echo ""
echo "=========================================="
echo "Summary:"
docker images --format "  {{.Repository}}:{{.Tag}} ({{.Size}})" | grep -E "hal-agent-runner"
echo ""

if [ $failed -eq 0 ]; then
    echo "All ${#AGENTS[@]} agent-env images ready!"
    echo ""
    echo "You can now run benchmarks with full parallelism:"
    echo "  ./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark scicode --docker --parallel-tasks 10"
else
    echo "WARNING: $failed image(s) failed to build"
fi
echo "=========================================="
