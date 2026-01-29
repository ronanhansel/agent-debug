# ORIGINAL.md - Reproducible HAL & Docent Runner Guide

This document provides comprehensive instructions for running **HAL** (Holistic Agent Leaderboard) and **Docent** (AI Agent Analysis Platform) using **OpenAI API keys** with the **original, unmodified runtime**.

This guide is designed for reproducibility - follow these steps exactly to set up a fresh environment.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [HAL Harness](#hal-harness)
  - [Installation](#hal-installation)
  - [Configuration](#hal-configuration)
  - [Running Evaluations](#running-evaluations)
  - [Benchmark Commands](#benchmark-commands)
  - [Result Management](#result-management)
- [Docent Platform](#docent-platform)
  - [Installation](#docent-installation)
  - [Running Docent](#running-docent)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## Overview

### What is HAL?

**HAL (Holistic Agent Leaderboard)** is a standardized evaluation harness for reproducible agent evaluations across various benchmarks. Key features:

- Unified `hal-eval` CLI across all benchmarks and agent types
- Local, Docker, or Azure VM execution modes
- Automatic logging via W&B Weave with cost tracking
- Support for custom agents without framework constraints
- Encrypted trace upload to HuggingFace Hub

### What is Docent?

**Docent** is an AI agent analysis platform for trace visualization and analysis:

- Web UI for browsing agent execution traces
- Database storage with PostgreSQL + pgvector
- Search, clustering, and LLM-powered analysis
- FastAPI backend + Next.js frontend

---

## Prerequisites

### System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Ubuntu 20.04+) or macOS |
| Python | 3.11 or 3.12 |
| Conda | Miniconda or Anaconda |
| Docker | 20.10+ (for Docker execution mode) |
| Git | 2.25+ |
| Memory | 8GB+ RAM recommended |
| Disk | 20GB+ free space |

### Required API Keys

| Key | Purpose | Get it from |
|-----|---------|-------------|
| `OPENAI_API_KEY` | LLM API calls | https://platform.openai.com/api-keys |
| `WANDB_API_KEY` | Weave logging & cost tracking | https://wandb.ai/settings |
| `HF_TOKEN` | HuggingFace Hub access | https://huggingface.co/settings/tokens |

---

## Environment Setup

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y git curl wget build-essential gpg jq

# Install Docker (if not installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for docker group to take effect

# Verify Docker
docker --version
docker run hello-world
```

### Step 2: Install Conda

```bash
# Download Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

### Step 3: Clone Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_ORG/agent-debug.git
cd agent-debug

# If already cloned without --recursive
git submodule update --init --recursive
```

### Step 4: Create Conda Environment

```bash
# Create environment with Python 3.12
conda create -n hal python=3.12 -y
conda activate hal

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Step 5: Install Python Dependencies

```bash
# Install main requirements
pip install -r requirements.txt

# Install HAL harness (editable mode)
pip install -e ./hal-harness

# Install Docent packages
pip install -e ./docent
pip install -e ./docent/docent/

# Install OpenAI SDK explicitly
pip install openai>=1.0.0
```

### Step 6: Configure Environment Variables

Create the `.env` file in `hal-harness/`:

```bash
cat > hal-harness/.env << 'EOF'
# ===========================================
# HAL Harness Configuration - OpenAI API
# ===========================================

# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Weights & Biases (for Weave logging)
WANDB_API_KEY=your-wandb-api-key-here

# HuggingFace Token (for dataset access and uploads)
HF_TOKEN=hf_your-token-here

# Optional: Anthropic API (if using Claude models)
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# ===========================================
# Azure VM Configuration (Optional)
# ===========================================
# Only needed if using --vm execution mode

# AZURE_SUBSCRIPTION_ID=
# AZURE_RESOURCE_GROUP_NAME=
# AZURE_LOCATION=
# NETWORK_SECURITY_GROUP_NAME=
# SSH_PUBLIC_KEY_PATH=
# SSH_PRIVATE_KEY_PATH=
EOF

# IMPORTANT: Edit with your actual keys
nano hal-harness/.env
```

### Step 7: Verify Installation

```bash
# Activate environment
conda activate hal

# Verify hal-eval CLI is available
hal-eval --help

# Verify OpenAI connection
python -c "from openai import OpenAI; print('OpenAI SDK OK')"

# Verify Weave
python -c "import weave; print(f'Weave version: {weave.__version__}')"
```

---

## HAL Harness

### HAL Installation

The HAL harness is installed from the `hal-harness/` submodule:

```bash
# Basic installation
pip install -e ./hal-harness

# With benchmark-specific dependencies
pip install -e "./hal-harness[swebench]"      # SWE-bench
pip install -e "./hal-harness[usaco]"         # USACO
pip install -e "./hal-harness[scicode]"       # SciCode
pip install -e "./hal-harness[corebench]"     # CORE-bench
pip install -e "./hal-harness[assistantbench]" # AssistantBench
pip install -e "./hal-harness[colbench]"      # ColBench
pip install -e "./hal-harness[taubench]"      # tau-bench
pip install -e "./hal-harness[appworld]"      # AppWorld
pip install -e "./hal-harness[azure]"         # Azure VM support
```

### HAL Configuration

The HAL CLI reads configuration from:
1. Command-line arguments
2. Environment variables in `.env`
3. Optional `config.yaml` file

**Minimal `.env` for OpenAI:**
```bash
OPENAI_API_KEY=sk-your-key-here
WANDB_API_KEY=your-wandb-key
HF_TOKEN=hf_your-token
```

### Running Evaluations

#### Basic Command Structure

```bash
hal-eval --benchmark <benchmark_name> \
    --agent_dir <path_to_agent> \
    --agent_function <module.function> \
    --agent_name "<Agent Name> (<model>)" \
    -A model_name=<model_name> \
    [OPTIONS]
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--benchmark` | Benchmark name (required) |
| `--agent_dir` | Path to agent directory |
| `--agent_function` | Entry point like `main.run` |
| `--agent_name` | Display name for leaderboard |
| `-A key=value` | Agent arguments |
| `-B key=value` | Benchmark arguments |
| `-I key=value` | Inspect AI arguments |
| `--max_concurrent N` | Parallel tasks (default: 1) |
| `--max_tasks N` | Limit tasks for testing |
| `--docker` | Run in Docker containers |
| `--vm` | Run on Azure VMs |
| `--conda_env_name` | Conda env for local execution |
| `--upload` | Upload results to HuggingFace |
| `--run_id` | Custom run identifier |
| `--continue_run` | Resume previous run |

#### Execution Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Local** | (default) | Direct execution in current/specified conda env |
| **Docker** | `--docker` | Isolated containers (4GB memory, 2 CPU cores) |
| **Azure VM** | `--vm` | Cloud execution with auto-provisioning |

### Benchmark Commands

#### SWE-bench Verified

```bash
# Mini version (50 problems)
hal-eval --benchmark swebench_verified_mini \
    --agent_dir hal-harness/agents/swebench_example_agent/ \
    --agent_function main.run \
    --agent_name "My Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 5

# Full version
hal-eval --benchmark swebench_verified \
    --agent_dir hal-harness/agents/swebench_example_agent/ \
    --agent_function main.run \
    --agent_name "My Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 10
```

**Note**: SWE-bench requires Docker. Install with:
```bash
pip install -e "./hal-harness[swebench]"
```

#### USACO

```bash
# With Docker (recommended)
hal-eval --benchmark usaco \
    --agent_dir hal-harness/agents/usaco_example_agent/ \
    --agent_function main.run \
    --agent_name "USACO Solver (gpt-4o)" \
    --docker \
    --max_concurrent 5 \
    -A model_name=gpt-4o
```

**Setup**: Download USACO data from [Google Drive](https://drive.google.com/file/d/1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi/view) and extract to `hal-harness/hal/benchmarks/USACO/data/`.

#### SciCode

```bash
# Standard version
hal-eval --benchmark scicode \
    --agent_dir hal-harness/agents/scicode_tool_calling_agent/ \
    --agent_function main.run \
    --agent_name "SciCode Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 5

# Easy version (with background info)
hal-eval --benchmark scicode_easy \
    --agent_dir hal-harness/agents/scicode_tool_calling_agent/ \
    --agent_function main.run \
    --agent_name "SciCode Agent (gpt-4o)" \
    -A model_name=gpt-4o

# Hard version (zero-shot)
hal-eval --benchmark scicode_hard \
    --agent_dir hal-harness/agents/scicode_tool_calling_agent/ \
    --agent_function main.run \
    --agent_name "SciCode Agent (gpt-4o)" \
    -A model_name=gpt-4o
```

**Setup**: Download `test_data.h5` and place in `hal-harness/hal/benchmarks/SciCode/eval/data/`.

#### CORE-bench

```bash
# Decrypt test data first
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
    --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
# Password: reproducibility

# Run evaluation
hal-eval --benchmark corebench_hard \
    --agent_dir hal-harness/agents/core_agent/ \
    --agent_function main.run \
    --agent_name "CORE Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --docker

# Difficulty levels: corebench_easy, corebench_medium, corebench_hard
```

#### ScienceAgentBench

```bash
# Download datasets first
cd hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/benchmark/
pip install gdown
gdown 1IYVRVK0TSZXRVKiSc2D0wxV1LXoXhpfA
unzip -P scienceagentbench benchmark.zip
mv benchmark/* . && rmdir benchmark
cd -

# Run with Docker
hal-eval --benchmark scienceagentbench \
    --agent_dir hal-harness/agents/hal_generalist_agent/ \
    --agent_function main.run \
    --agent_name "SAB Agent (gpt-4o)" \
    --docker \
    --max_concurrent 5 \
    -A model_name=openai/gpt-4o \
    -A max_steps=5
```

#### ColBench

```bash
# Backend programming (1000 tasks)
hal-eval --benchmark colbench_backend_programming \
    --agent_dir hal-harness/agents/colbench_example_agent/ \
    --agent_function main.run \
    --agent_name "ColBench Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 20

# Frontend design (100 tasks)
hal-eval --benchmark colbench_frontend_design \
    --agent_dir hal-harness/agents/colbench_example_agent/ \
    --agent_function main.run \
    --agent_name "ColBench Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 10
```

#### tau-bench

```bash
pip install -e "./hal-harness[taubench]"

# Retail domain
hal-eval --benchmark taubench_retail \
    --agent_dir hal-harness/agents/taubench_example_agent/ \
    --agent_function main.run \
    --agent_name "tau Agent (gpt-4o)" \
    -A model_name=gpt-4o

# Airline domain
hal-eval --benchmark taubench_airline \
    --agent_dir hal-harness/agents/taubench_example_agent/ \
    --agent_function main.run \
    --agent_name "tau Agent (gpt-4o)" \
    -A model_name=gpt-4o
```

#### AssistantBench

```bash
pip install -e "./hal-harness[assistantbench]"

hal-eval --benchmark assistantbench \
    --agent_dir hal-harness/agents/assistantbench_browser_agent/ \
    --agent_function main.run \
    --agent_name "Browser Agent (gpt-4o)" \
    -A model_name=gpt-4o
```

#### AppWorld

```bash
pip install -e "./hal-harness[appworld]"
appworld install
appworld download data --root hal-harness/agents/appworld_agent

# Normal test suite
hal-eval --benchmark appworld_test_normal \
    --agent_dir hal-harness/agents/appworld_agent/ \
    --agent_function main.run \
    --agent_name "AppWorld Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    -A method_name=simplified_react

# Challenge test suite
hal-eval --benchmark appworld_test_challenge \
    --agent_dir hal-harness/agents/appworld_agent/ \
    --agent_function main.run \
    --agent_name "AppWorld Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    -A method_name=simplified_react
```

#### Inspect AI Benchmarks

```bash
# GAIA
hal-eval --benchmark inspect_evals/gaia \
    --agent_dir hal-harness/agents/inspect/ \
    --agent_function gaia.default_agent \
    --agent_name "GAIA Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o \
    -I token_limit=4000

# Cybench (requires Docker config)
hal-eval --benchmark inspect_evals/cybench \
    --agent_dir hal-harness/agents/inspect/cybench/ \
    --agent_function cybench.default_agent \
    --agent_name "Cybench Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o

# AgentHarm (benign)
hal-eval --benchmark inspect_evals/agentharm_benign \
    --agent_dir hal-harness/agents/inspect/agentharm/ \
    --agent_function agentharm.default_agent \
    --agent_name "AgentHarm Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o \
    -A task_name=benign
```

### Result Management

#### Upload Results

```bash
# During evaluation
hal-eval ... --upload

# After evaluation - upload all for a benchmark
hal-upload -B scicode

# Upload specific file
hal-upload -F results/scicode/run_id/run_id_UPLOAD.json

# Upload directory
hal-upload -D results/scicode/
```

#### Download and Decrypt Traces

```bash
# Decrypt entire directory
hal-decrypt -D path/to/downloaded_traces/

# Decrypt single file
hal-decrypt -F trace_file_UPLOAD.zip
```

---

## Docent Platform

### Docent Installation

```bash
# Install Docent core and SDK
pip install -e ./docent
pip install -e ./docent/docent/
```

### Running Docent

Docent has three components that run separately:

#### 1. Backend Server (FastAPI)

```bash
# Start backend
docent_core server \
    --host 0.0.0.0 \
    --port 8888 \
    --workers 1

# With auto-reload for development
docent_core server \
    --host 0.0.0.0 \
    --port 8888 \
    --reload
```

#### 2. Frontend Web App (Next.js)

```bash
# Build and start frontend
docent_core web \
    --backend-url http://localhost:8888 \
    --port 3000 \
    --build \
    --install
```

#### 3. Background Worker

```bash
# Start worker for background jobs
docent_core worker --workers 4
```

#### Full Docent Stack

Run all three in separate terminals:

```bash
# Terminal 1: Backend
conda activate hal
docent_core server --host 0.0.0.0 --port 8888

# Terminal 2: Worker
conda activate hal
docent_core worker --workers 4

# Terminal 3: Frontend
conda activate hal
docent_core web --backend-url http://localhost:8888 --port 3000
```

Access the web UI at: `http://localhost:3000`

### Docent Environment Variables

Create a Docent-specific `.env`:

```bash
# Database (PostgreSQL)
DATABASE_URL=postgresql://user:pass@localhost:5432/docent

# Redis (for job queue)
REDIS_URL=redis://localhost:6379

# LLM API keys (for analysis features)
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
```

---

## Common Workflows

### Workflow 1: Run a Quick Test

```bash
# Activate environment
conda activate hal

# Run SWE-bench mini with 2 tasks
hal-eval --benchmark swebench_verified_mini \
    --agent_dir hal-harness/agents/swebench_example_agent/ \
    --agent_function main.run \
    --agent_name "Test Agent (gpt-4o-mini)" \
    -A model_name=gpt-4o-mini \
    --max_tasks 2 \
    --max_concurrent 2
```

### Workflow 2: Full Benchmark Evaluation

```bash
# Run full SciCode evaluation
hal-eval --benchmark scicode \
    --agent_dir hal-harness/agents/scicode_tool_calling_agent/ \
    --agent_function main.run \
    --agent_name "SciCode Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --max_concurrent 10 \
    --upload
```

### Workflow 3: Continue Interrupted Run

```bash
# Get run_id from results directory
ls results/scicode/

# Continue the run
hal-eval --benchmark scicode \
    --agent_dir hal-harness/agents/scicode_tool_calling_agent/ \
    --agent_function main.run \
    --agent_name "SciCode Agent (gpt-4o)" \
    -A model_name=gpt-4o \
    --run_id <previous_run_id> \
    --continue_run
```

### Workflow 4: Docker Execution

```bash
# Build HAL Docker image
cd hal-harness
docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .
cd ..

# Run with Docker
hal-eval --benchmark usaco \
    --agent_dir hal-harness/agents/usaco_example_agent/ \
    --agent_function main.run \
    --agent_name "USACO Solver (gpt-4o)" \
    --docker \
    --max_concurrent 5 \
    -A model_name=gpt-4o
```

---

## Troubleshooting

### OpenAI API Errors

**Error**: `AuthenticationError: Invalid API key`
```bash
# Verify key is set
echo $OPENAI_API_KEY
cat hal-harness/.env | grep OPENAI_API_KEY

# Test API directly
python -c "
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(response.choices[0].message.content)
"
```

### Docker Issues

**Error**: `Cannot connect to Docker daemon`
```bash
# Check Docker is running
docker ps

# Check socket permissions
ls -la /var/run/docker.sock

# Set Docker host
export HAL_DOCKER_HOST=unix:///var/run/docker.sock
```

**Error**: `Container exits immediately`
```bash
# Debug with preflight network check
export HAL_DOCKER_PREFLIGHT_NETWORK=1

# Force rebuild images
export HAL_DOCKER_FORCE_REBUILD=1
```

### Weave Logging Issues

**Error**: `Weave initialization failed`
```bash
# Verify Weave API key
python -c "
import os
import weave
print(f'WANDB_API_KEY set: {bool(os.getenv(\"WANDB_API_KEY\"))}')
"

# Login manually
wandb login
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'hal'`
```bash
# Reinstall in editable mode
pip install -e ./hal-harness

# Verify installation
python -c "import hal; print(hal.__file__)"
```

### Package Conflicts

```bash
# Recreate environment from scratch
conda deactivate
conda env remove -n hal
conda create -n hal python=3.12 -y
conda activate hal
pip install -r requirements.txt
pip install -e ./hal-harness
pip install -e ./docent
```

---

## Quick Reference

### HAL CLI Commands

| Command | Description |
|---------|-------------|
| `hal-eval` | Run agent evaluation |
| `hal-upload` | Upload results to HuggingFace |
| `hal-decrypt` | Decrypt downloaded traces |

### Docent CLI Commands

| Command | Description |
|---------|-------------|
| `docent_core server` | Start backend API server |
| `docent_core web` | Start frontend web app |
| `docent_core worker` | Start background job worker |

### Supported Benchmarks

| Benchmark | ID | Docker Required |
|-----------|-----|-----------------|
| SWE-bench Verified | `swebench_verified` | Yes |
| SWE-bench Mini | `swebench_verified_mini` | Yes |
| USACO | `usaco` | Recommended |
| SciCode | `scicode`, `scicode_easy`, `scicode_hard` | No |
| CORE-bench | `corebench_easy/medium/hard` | Yes |
| ScienceAgentBench | `scienceagentbench` | Recommended |
| ColBench | `colbench_backend_programming`, `colbench_frontend_design` | No |
| tau-bench | `taubench_retail`, `taubench_airline` | No |
| AssistantBench | `assistantbench` | No |
| AppWorld | `appworld_test_normal`, `appworld_test_challenge` | No |
| GAIA | `inspect_evals/gaia` | No |
| Cybench | `inspect_evals/cybench` | Yes |
| AgentHarm | `inspect_evals/agentharm`, `inspect_evals/agentharm_benign` | No |

### OpenAI Model Names

| Model | Use in `-A model_name=` |
|-------|-------------------------|
| GPT-4o | `gpt-4o` or `gpt-4o-2024-11-20` |
| GPT-4o Mini | `gpt-4o-mini` or `gpt-4o-mini-2024-07-18` |
| GPT-4 Turbo | `gpt-4-turbo` or `gpt-4-turbo-2024-04-09` |
| GPT-4.1 | `gpt-4.1-2025-04-14` |
| O1 | `o1` or `o1-2024-12-17` |
| O1 Mini | `o1-mini` |
| O3 | `o3-2025-04-16` |
| O3 Mini | `o3-mini` |
| O4 Mini | `o4-mini-2025-04-16` |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `WANDB_API_KEY` | Yes | Weave logging |
| `HF_TOKEN` | Yes | HuggingFace access |
| `ANTHROPIC_API_KEY` | No | For Claude models |
| `HAL_DOCKER_HOST` | No | Docker socket path |
| `HAL_DOCKER_FORCE_REBUILD` | No | Force image rebuild |

---

## Version Information

- **HAL Harness**: v0.1.0
- **Docent**: Latest from submodule
- **Weave**: 0.51.41 (pinned)
- **Python**: 3.11+ (3.12 recommended)

---

## Citations

If you use HAL in your research:

```bibtex
@Misc{hal,
  title =        {HAL: A Holistic Agent Leaderboard for Centralized and Reproducible Agent Evaluation},
  author =       {Benedikt Stroebl and Sayash Kapoor and Arvind Narayanan},
  howpublished = {\url{https://github.com/princeton-pli/hal-harness}},
  year =         {2025}
}
```

---

## Support

- HAL Documentation: https://hal.cs.princeton.edu/
- Docent Documentation: https://docs.transluce.org
- Issues: https://github.com/princeton-pli/hal-harness/issues
