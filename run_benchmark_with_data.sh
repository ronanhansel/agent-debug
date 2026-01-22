#!/bin/bash
# Complete wrapper to run benchmarks using /Data for temporary storage
# This prevents filling up the root partition
# Also pre-builds agent-env Docker images to avoid build contention

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HAL_REPO_ROOT:-$SCRIPT_DIR}"
WORKDIR="${HAL_WORKDIR:-$REPO_ROOT}"
HAL_HARNESS="$REPO_ROOT/hal-harness"
DEFAULT_DATA_ROOT="/Data/home/${USER}"
if [ ! -d "$DEFAULT_DATA_ROOT" ]; then
    DEFAULT_DATA_ROOT="/Data"
fi
DATA_ROOT="${HAL_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
DATA_NAMESPACE="${HAL_DATA_NAMESPACE:-$USER}"
DATA_RUN_ROOT="${HAL_DATA_RUN_ROOT:-$DATA_ROOT/hal_runs/$DATA_NAMESPACE/$(basename "$REPO_ROOT")}"

# Set working directory
cd "$WORKDIR"

# Configure all temp directories to use /Data
export TMPDIR="${HAL_TMPDIR:-$DATA_RUN_ROOT/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHON_TEMPDIR="$TMPDIR"
export DOCKER_TMPDIR="${HAL_DOCKER_TMPDIR:-$TMPDIR/docker}"
export HAL_RESULTS_DIR="${HAL_RESULTS_DIR:-$DATA_RUN_ROOT/results}"
export HAL_TRACES_DIR="${HAL_TRACES_DIR:-$DATA_RUN_ROOT/traces}"
export HAL_TMP_DIR="${HAL_TMP_DIR:-$DATA_RUN_ROOT/tmp}"

# Ensure temp directory exists
mkdir -p "$TMPDIR" "$DOCKER_TMPDIR" "$HAL_RESULTS_DIR" "$HAL_TRACES_DIR" "$HAL_TMP_DIR"

link_dir() {
    local name="$1"
    local target="$DATA_RUN_ROOT/$name"
    mkdir -p "$target"
    if [ -L "$WORKDIR/$name" ]; then
        return
    fi
    if [ -e "$WORKDIR/$name" ]; then
        echo "WARN: $WORKDIR/$name exists and is not a symlink. Move it to $target or delete it to enable linking."
        return
    fi
    ln -s "$target" "$WORKDIR/$name"
}

if [ "${HAL_LINK_DATA_DIRS:-1}" != "0" ]; then
    link_dir "results"
    link_dir "traces"
    link_dir ".tmp"
    link_dir "log"
    link_dir "logs"
    link_dir "output"
fi

# =============================================================================
# Agent-env Docker image pre-building
# =============================================================================

# Map benchmarks to their required agent directories
declare -A BENCHMARK_AGENTS
BENCHMARK_AGENTS[scicode]="hal_generalist_agent scicode_tool_calling_agent"
BENCHMARK_AGENTS[scienceagentbench]="hal_generalist_agent sab_example_agent"
BENCHMARK_AGENTS[corebench]="hal_generalist_agent core_agent"
BENCHMARK_AGENTS[colbench]="hal_generalist_agent colbench_example_agent"

# Compute agent-env image tag from requirements.txt hash
get_agent_env_tag() {
    local agent_dir="$1"
    local req_file="$HAL_HARNESS/agents/$agent_dir/requirements.txt"
    if [ -f "$req_file" ]; then
        local hash=$(md5sum "$req_file" | cut -c1-12)
        echo "hal-agent-runner:agent-env-$hash"
    else
        echo ""
    fi
}

# Check if image exists
image_exists() {
    local tag="$1"
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${tag}$"
}

# Build agent-env image for a specific agent
build_agent_env() {
    local agent_dir="$1"
    local tag=$(get_agent_env_tag "$agent_dir")
    local req_file="$HAL_HARNESS/agents/$agent_dir/requirements.txt"

    if [ -z "$tag" ]; then
        echo "  [SKIP] $agent_dir - no requirements.txt"
        return 0
    fi

    if image_exists "$tag"; then
        echo "  [OK] $agent_dir - image exists ($tag)"
        return 0
    fi

    echo "  [BUILD] $agent_dir - building $tag..."

    # Create a temporary Dockerfile for agent-env
    local tmp_dockerfile=$(mktemp)
    cat > "$tmp_dockerfile" << 'DOCKERFILE'
ARG BASE_IMAGE=hal-agent-runner:latest
FROM ${BASE_IMAGE}

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create agent environment
RUN conda create -y -n agent_env python=3.11 && \
    conda run -n agent_env python -m pip install -U pip

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9

WORKDIR /workspace
DOCKERFILE

    # Build the image
    local build_context="$HAL_HARNESS/agents/$agent_dir"
    if docker build -t "$tag" -f "$tmp_dockerfile" "$build_context" > /dev/null 2>&1; then
        echo "  [DONE] $agent_dir - built successfully"
        rm -f "$tmp_dockerfile"
        return 0
    else
        echo "  [FAIL] $agent_dir - build failed, will retry at runtime"
        rm -f "$tmp_dockerfile"
        return 1
    fi
}

# Pre-build all required agent-env images for detected benchmarks
prebuild_agent_envs() {
    local benchmarks_to_build=""

    # Detect which benchmarks are in the command line
    for arg in "$@"; do
        case "$arg" in
            scicode|scienceagentbench|corebench|colbench)
                benchmarks_to_build="$benchmarks_to_build $arg"
                ;;
        esac
    done

    if [ -z "$benchmarks_to_build" ]; then
        return 0
    fi

    echo "=========================================="
    echo "Pre-building Agent Environment Images"
    echo "=========================================="

    # First ensure base image exists
    if ! image_exists "hal-agent-runner:latest"; then
        echo "Building base image hal-agent-runner:latest..."
        docker build -t hal-agent-runner:latest "$HAL_HARNESS/hal/utils/docker/" || {
            echo "ERROR: Failed to build base image"
            exit 1
        }
    else
        echo "Base image hal-agent-runner:latest exists"
    fi

    # Collect unique agents needed
    declare -A agents_needed
    for bench in $benchmarks_to_build; do
        for agent in ${BENCHMARK_AGENTS[$bench]}; do
            agents_needed[$agent]=1
        done
    done

    echo ""
    echo "Agents needed: ${!agents_needed[@]}"
    echo ""

    # Build each agent-env
    local all_ok=true
    for agent in "${!agents_needed[@]}"; do
        build_agent_env "$agent" || all_ok=false
    done

    echo ""
    if $all_ok; then
        echo "All agent-env images ready!"
    else
        echo "Some images failed to pre-build (will retry at runtime)"
    fi
    echo "=========================================="
    echo ""
}

# =============================================================================
# Main script
# =============================================================================

# Show current disk usage
echo "=========================================="
echo "Disk Usage Status:"
echo "=========================================="
echo "Root partition:"
df -h / | tail -1
echo ""
echo "/Data partition:"
df -h /Data | tail -1
echo ""
echo "Temp directory: $TMPDIR"
echo "Available for temp files: $(df -h $TMPDIR | tail -1 | awk '{print $4}')"
echo ""
echo "Docker usage:"
docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}"
echo "=========================================="
echo ""

# Check if we have enough space
ROOT_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $ROOT_USAGE -ge 95 ]; then
    echo "WARNING: Root partition is ${ROOT_USAGE}% full!"
    echo "Consider running ./cleanup_docker.sh first"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please run ./cleanup_docker.sh to free space."
        exit 1
    fi
fi

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Example:"
    echo "  $0 python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 10"
    echo ""
    echo "Supported benchmarks (with auto pre-build):"
    echo "  scicode, scienceagentbench, corebench, colbench"
    echo ""
    exit 1
fi

# Pre-build agent-env images if running with --docker
if echo "$@" | grep -q "\-\-docker"; then
    # Use Python script for correct HAL hash calculation
    if [ -f "$SCRIPT_DIR/prebuild_agent_envs.py" ]; then
        echo "Pre-building agent-env images (using HAL's hash calculation)..."
        python3 "$SCRIPT_DIR/prebuild_agent_envs.py" 2>&1 | grep -E "^\[|^Step|^All|^WARNING|^Agents"
        echo ""
    else
        # Fallback to shell-based prebuild
        prebuild_agent_envs "$@"
    fi
fi

# Run the command
echo "Starting command with /Data temporary storage..."
echo "Command: $@"
echo ""

exec "$@"
