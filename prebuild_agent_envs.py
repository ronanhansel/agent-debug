#!/usr/bin/env python3
"""Pre-build all agent-env Docker images using HAL's actual hash calculation."""

import sys
import hashlib
import json
from pathlib import Path

# Add hal-harness to path
HAL_HARNESS = Path(__file__).parent / "hal-harness"
sys.path.insert(0, str(HAL_HARNESS))

import docker

# Constants from HAL's docker_runner.py
DOCKER_IMAGE_NAME = "hal-agent-runner:latest"
AGENT_ENV_TEMPLATE_VERSION = 2
AGENT_ENV_PYTHON_VERSION = "3.11"

# All agents used by benchmarks
BENCHMARK_AGENTS = {
    "scicode": ["hal_generalist_agent", "scicode_tool_calling_agent"],
    "scienceagentbench": ["hal_generalist_agent", "sab_example_agent"],
    "corebench": ["hal_generalist_agent", "core_agent"],
    "colbench": ["hal_generalist_agent", "colbench_example_agent"],
}


def compute_requirements_hash(requirements_path: Path, docker_client) -> str:
    """Compute hash exactly as HAL does."""
    req_bytes = requirements_path.read_bytes()
    try:
        base_image_id = docker_client.images.get(DOCKER_IMAGE_NAME).id.encode("utf-8")
    except Exception:
        base_image_id = b"unknown-base-image"
    recipe = (
        f"template={AGENT_ENV_TEMPLATE_VERSION}\n"
        f"python={AGENT_ENV_PYTHON_VERSION}\n"
        "weave=0.51.41\n"
        "wandb=0.17.9\n"
    ).encode("utf-8")
    return hashlib.sha256(req_bytes + b"\n" + base_image_id + b"\n" + recipe).hexdigest()[:16]


def image_exists(docker_client, tag: str) -> bool:
    """Check if Docker image exists."""
    try:
        docker_client.images.get(tag)
        return True
    except docker.errors.ImageNotFound:
        return False


def build_agent_env(docker_client, agent_dir: str, agents_path: Path) -> bool:
    """Build agent-env image for a specific agent with verbose output."""
    req_file = agents_path / agent_dir / "requirements.txt"

    if not req_file.exists():
        print(f"  [SKIP] {agent_dir} - no requirements.txt")
        return True

    req_hash = compute_requirements_hash(req_file, docker_client)
    tag = f"hal-agent-runner:agent-env-{req_hash}"

    if image_exists(docker_client, tag):
        print(f"  [OK] {agent_dir} ({tag})")
        return True

    print(f"  [BUILD] {agent_dir} -> {tag}")
    num_packages = sum(1 for line in req_file.open() if line.strip() and not line.startswith('#'))
    print(f"    Installing {num_packages} packages...")
    sys.stdout.flush()

    # Create Dockerfile content
    dockerfile = f"""
FROM {DOCKER_IMAGE_NAME}

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \\
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n agent_env python={AGENT_ENV_PYTHON_VERSION} && \\
    conda run -n agent_env python -m pip install -U pip

COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \\
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9

WORKDIR /workspace
"""

    # Write temporary Dockerfile
    build_context = agents_path / agent_dir
    dockerfile_path = build_context / "Dockerfile.agent-env"
    dockerfile_path.write_text(dockerfile)

    try:
        # Use low-level API for streaming output
        api_client = docker_client.api

        print(f"    Building (this may take 10-15 minutes)...")
        sys.stdout.flush()

        # Build with streaming
        build_stream = api_client.build(
            path=str(build_context),
            dockerfile="Dockerfile.agent-env",
            tag=tag,
            rm=True,
            decode=True,
        )

        last_step = ""
        for chunk in build_stream:
            if 'stream' in chunk:
                line = chunk['stream'].strip()
                if line:
                    # Show step progress
                    if line.startswith('Step '):
                        last_step = line
                        print(f"    {line}")
                        sys.stdout.flush()
                    # Show pip install progress
                    elif 'Successfully installed' in line:
                        print(f"    {line[:80]}...")
                        sys.stdout.flush()
                    elif 'Collecting' in line and 'from' not in line.lower():
                        # Show package being collected (first 60 chars)
                        pkg = line.replace('Collecting ', '').split()[0]
                        print(f"      Installing: {pkg[:50]}", end='\r')
                        sys.stdout.flush()
            elif 'error' in chunk:
                print(f"    ERROR: {chunk['error']}")
                sys.stdout.flush()
                return False

        print(f"\n  [DONE] {agent_dir}")
        sys.stdout.flush()
        return True

    except docker.errors.BuildError as e:
        print(f"\n  [FAIL] {agent_dir}: {e}")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"\n  [FAIL] {agent_dir}: {type(e).__name__}: {e}")
        sys.stdout.flush()
        return False
    finally:
        dockerfile_path.unlink(missing_ok=True)


def main():
    print("=" * 50)
    print("Pre-building ALL Agent Environment Images")
    print("=" * 50)
    print()
    sys.stdout.flush()

    docker_client = docker.from_env()
    agents_path = HAL_HARNESS / "agents"

    # Check base image
    print("Step 1: Checking base image...")
    sys.stdout.flush()
    if not image_exists(docker_client, DOCKER_IMAGE_NAME):
        print(f"  [BUILD] Building {DOCKER_IMAGE_NAME}...")
        sys.stdout.flush()
        dockerfile_path = HAL_HARNESS / "hal" / "utils" / "docker"

        # Stream base image build too
        api_client = docker_client.api
        for chunk in api_client.build(path=str(dockerfile_path), tag=DOCKER_IMAGE_NAME, rm=True, decode=True):
            if 'stream' in chunk:
                line = chunk['stream'].strip()
                if line.startswith('Step '):
                    print(f"    {line}")
                    sys.stdout.flush()

        print(f"  [DONE] Base image built")
    else:
        print(f"  [OK] {DOCKER_IMAGE_NAME} exists")
    print()
    sys.stdout.flush()

    # Collect unique agents
    all_agents = set()
    for agents in BENCHMARK_AGENTS.values():
        all_agents.update(agents)

    print("Step 2: Building agent-env images...")
    print(f"Agents to build: {', '.join(sorted(all_agents))}")
    print()
    sys.stdout.flush()

    failed = 0
    for agent in sorted(all_agents):
        if not build_agent_env(docker_client, agent, agents_path):
            failed += 1

    print()
    print("=" * 50)
    print("Summary:")
    for img in docker_client.images.list():
        for tag in img.tags:
            if tag.startswith("hal-agent-runner:"):
                size_mb = img.attrs["Size"] / 1024 / 1024
                print(f"  {tag} ({size_mb:.0f}MB)")
    print()

    if failed == 0:
        print(f"All {len(all_agents)} agent-env images ready!")
        print()
        print("You can now run benchmarks with full parallelism:")
        print("  ./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py \\")
        print("      --benchmark scicode --docker --parallel-tasks 10")
    else:
        print(f"WARNING: {failed} image(s) failed to build")
        sys.exit(1)

    print("=" * 50)


if __name__ == "__main__":
    main()
