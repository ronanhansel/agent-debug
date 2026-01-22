#!/bin/bash
#
# Pre-build Docker base images for all benchmarks
# Run this ONCE before starting parallel benchmark runs to avoid race conditions
#
# Usage: ./prebuild_images.sh
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}       PRE-BUILD DOCKER BASE IMAGES${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# SAB base image
build_sab_base() {
    local image_name="sab.base.x86_64:latest"

    if docker images -q "$image_name" 2>/dev/null | grep -q .; then
        echo -e "${GREEN}[SAB] Base image already exists: $image_name${NC}"
        return 0
    fi

    echo -e "${YELLOW}[SAB] Building base image: $image_name${NC}"
    echo -e "${YELLOW}[SAB] This may take 5-10 minutes...${NC}"

    python3 << PYTHON_SCRIPT
import docker
import sys
sys.path.insert(0, '$SCRIPT_DIR/hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/evaluation/harness')
from dockerfiles import get_dockerfile_base
from pathlib import Path
import tempfile

client = docker.from_env()
dockerfile = get_dockerfile_base('linux/x86_64')

with tempfile.TemporaryDirectory() as tmpdir:
    dockerfile_path = Path(tmpdir) / 'Dockerfile'
    dockerfile_path.write_text(dockerfile)

    print('Building sab.base.x86_64:latest...')
    image, logs = client.images.build(
        path=tmpdir,
        tag='sab.base.x86_64:latest',
        platform='linux/x86_64',
        rm=True
    )
    for log in logs:
        if 'stream' in log:
            print(log['stream'], end='')
    print(f'\nBuilt successfully: {image.tags}')
PYTHON_SCRIPT

    if docker images -q "$image_name" 2>/dev/null | grep -q .; then
        echo -e "${GREEN}[SAB] Base image built successfully${NC}"
        return 0
    else
        echo -e "${RED}[SAB] ERROR: Failed to build base image${NC}"
        return 1
    fi
}

# Check current state
echo -e "${CYAN}Current Docker images:${NC}"
docker images | grep -E "sab|sweb|agent-env" | head -10 || echo "  (none found)"
echo ""

# Build SAB
echo -e "${CYAN}Building SAB base image...${NC}"
build_sab_base
echo ""

# Summary
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}       BUILD COMPLETE${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${CYAN}Docker images:${NC}"
docker images | grep -E "sab|sweb|agent-env" | head -10 || echo "  (none)"
echo ""
echo -e "${GREEN}Ready to run benchmarks with: ./run_all_benchmarks.sh${NC}"
