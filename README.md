# Agent Debug Pipeline

Automated Item Fixing Pipeline for benchmark evaluation. The core innovation is **item-level fixing** - modifying agent configurations, prompt templates, and runtime parameters to eliminate false-positive benchmark defects without changing the benchmark source code itself.

## Supported Benchmarks

| Benchmark | Fixer Script | Runner Script | Model Config | Fixes |
|-----------|--------------|---------------|--------------|-------|
| **SciCode** | `claude_fixer_scicode.py` | `run_scicode_fixes.py` | `model_to_baseline_scicode.json` | 61 |
| **CoreBench** | `claude_fixer_corebench.py` | `run_corebench_fixes.py` | `model_to_baseline_corebench.json` | 13 |
| **ScienceAgentBench** | `claude_fixer_scienceagentbench.py` | `run_scienceagentbench_fixes.py` | `model_to_baseline_scienceagentbench.json` | 26 |
| **ColBench** | `claude_fixer_colbench.py` | `run_colbench_fixes.py` | `model_to_baseline_colbench.json` | 81 |
| **USACO** | (manual) | `run_usaco_fixes.py` | `model_to_baseline_usaco.json` | 2 |

---

## Fresh Machine Setup

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Conda**: Miniconda or Anaconda
- **Docker**: Docker Engine 20.10+ with Docker Compose
- **Git**: Git 2.25+
- **Python**: 3.11 or 3.12 (via conda)
- **GPG**: For decrypting benchmark data

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    gpg \
    jq

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

# If already cloned without --recursive, initialize submodules
git submodule update --init --recursive
```

### Step 4: Create Conda Environment

```bash
# Create environment with Python 3.12
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n hal python=3.12 -y
conda activate hal

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Step 5: Install Python Dependencies

```bash
# Install main requirements
pip install -r requirements.txt

# Install docent (rubric evaluation library)
pip install -e ./docent

# Install hal-harness with all benchmark extras
pip install -e ./hal-harness
pip install -e "./hal-harness[scicode]"
pip install -e "./hal-harness[corebench]"
pip install -e "./hal-harness[assistantbench]"

# Fix common package conflicts
pip install --upgrade --force-reinstall certifi click

# Install Azure SDK (for TRAPI access)
pip install azure-identity msal
```

### Step 6: Build Docker Image

```bash
cd hal-harness

# Build the main HAL runner image
docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .

# Verify the image has required tools
docker run --rm hal-agent-runner:latest bash -lc \
    'python --version && Rscript --version 2>&1 | head -1 && pandoc --version | head -1'

cd ..
```

### Step 7: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or vim .env
```

**Required variables in `.env`:**

```bash
# OpenAI API (or Azure OpenAI)
OPENAI_API_KEY=sk-...

# For Azure/TRAPI direct access (recommended)
# No OPENAI_BASE_URL needed - uses Azure directly

# For local proxy (optional)
# OPENAI_BASE_URL=http://localhost:4000/v1

# Anthropic (for Claude-based fixers)
ANTHROPIC_API_KEY=sk-ant-...

# Weights & Biases (for Weave logging)
WANDB_API_KEY=...

# HuggingFace (for dataset access)
HF_TOKEN=hf_...
```

### Step 8: Azure Authentication (for TRAPI)

```bash
# Login to Azure CLI
az login

# Verify authentication
az account show

# The scripts will automatically use Azure CLI credentials for TRAPI
```

### Step 9: Decrypt Benchmark Data (CoreBench)

```bash
# Decrypt CoreBench test set
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
    --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
# Passphrase: reproducibility
```

### Step 10: Download SciCode Test Data

```bash
# Download test_data.h5 for SciCode evaluation
# Place in: hal-harness/hal/benchmarks/SciCode/eval/data/test_data.h5
mkdir -p hal-harness/hal/benchmarks/SciCode/eval/data
# Download from the SciCode dataset source
```

### Step 11: Verify Installation

```bash
# Activate environment
conda activate hal

# Verify hal-eval CLI
hal-eval --help

# Verify scripts can import
python -c "import scripts.run_scicode_fixes"
python -c "import scripts.run_corebench_fixes"
python -c "import scripts.eval_rubric"

# List available fixes
python scripts/run_scicode_fixes.py --list-fixes
python scripts/run_corebench_fixes.py --list-fixes

# Test Docker connectivity
docker ps
```

---

## Directory Structure

```
agent-debug/
├── .env                         # Environment variables (create from .env.example)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── CLAUDE.md                    # Claude Code instructions
├── PIPELINE_README.md           # Detailed pipeline documentation
│
├── hal-harness/                 # HAL evaluation harness (submodule)
│   ├── agents/                  # Agent implementations
│   ├── hal/
│   │   ├── benchmarks/          # Benchmark implementations
│   │   └── utils/docker/        # Docker configuration
│   └── pyproject.toml
│
├── docent/                      # Rubric evaluation library (submodule)
│
├── scripts/                     # Pipeline scripts
│   ├── eval_rubric.py           # Rubric evaluation
│   ├── judge.py                 # Verdict aggregation
│   ├── claude_fixer_*.py        # Fix generation scripts
│   ├── run_*_fixes.py           # Fix runner scripts
│   └── merge_traces.py          # Trace merging
│
├── rubric_templates/            # Rubric prompts for LLM graders
│   ├── scicode.txt
│   ├── corebench.txt
│   ├── scienceagentbench.txt
│   ├── colbench.txt
│   └── usaco.txt
│
├── model_to_baseline_*.json     # Model configuration files
│
├── fixes/                       # Generated fix packages
│   ├── scicode/
│   ├── corebench_hard/
│   ├── scienceagentbench/
│   └── colbench/
│
├── traces/                      # Agent execution traces
├── rubrics_output/              # Rubric evaluation results
└── judge_output/                # Aggregated verdicts
```

---

## Running the Pipeline

### Full Pipeline (Any Benchmark)

```bash
# Activate environment
conda activate hal

# 1. Rubric evaluation (classify failures)
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

# 2. Judge aggregation (combine evaluations)
python scripts/judge.py \
    --pattern "<benchmark>_*" \
    --rubric-dir rubrics_output/<benchmark> \
    --model openai:gpt-5.2 -y

# 3. Generate fixes (Claude Code CLI)
python scripts/claude_fixer_<benchmark>.py \
    --rubric-dir rubrics_output/<benchmark> \
    --judge-csv judge_output/<benchmark>_verdict.csv \
    --ife-only --tasks-per-batch 5

# 4. Apply fixes and re-run
python scripts/run_<benchmark>_fixes.py --all-models --prefix iter1_ --docker
```

### Standardized CLI Interface

All fix runner scripts follow a consistent interface:

| Option | Description |
|--------|-------------|
| `--list-fixes` | List available fix directories and exit |
| `--dry-run` | Preview what would be done without running |
| `--verify-fixes` | Verify fixes are applied correctly without running HAL |
| `--all-models` | Run all models from config for each task with fixes |
| `--model MODEL` | Force a specific model for all tasks |
| `--rubric-csv PATH` | Use rubric CSV to determine which model failed each task |
| `--task-id ID` | Only run specific task(s), can be repeated |
| `--prefix PREFIX` | Prefix for run IDs and output files |
| `--docker` | Run HAL evaluation with Docker |
| `--parallel N` | Number of parallel HAL evaluations |
| `--skip-rubrics` | Skip rubric evaluation after running |

### Examples

```bash
# List what fixes are available
python scripts/run_scicode_fixes.py --list-fixes

# Dry run - see what would happen
python scripts/run_scicode_fixes.py --all-models --dry-run

# Run specific task with specific model
python scripts/run_scicode_fixes.py --task-id 11 --model openai/gpt-4.1-2025-04-14 --docker

# Run all models for all tasks with fixes
python scripts/run_scicode_fixes.py --all-models --prefix iter1_ --docker --parallel 3

# Run using rubric CSV to match failed (task, model) pairs
python scripts/run_scicode_fixes.py \
    --rubric-csv rubrics_output/scicode/scicode_combined.csv \
    --prefix iter1_ --docker
```

---

## Benchmark-Specific Workflows

### SciCode

```bash
# List IFE tasks
python scripts/claude_fixer_scicode.py --rubric-dir rubrics_output/scicode --list-ife-tasks

# Generate fixes
python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --ife-only --tasks-per-batch 5

# Apply fixes
python scripts/run_scicode_fixes.py --list-fixes
python scripts/run_scicode_fixes.py --all-models --prefix fixed_ --docker
```

### CoreBench

```bash
# List IFE tasks
python scripts/claude_fixer_corebench.py --rubric-dir rubrics_output/corebench --list-ife-tasks

# Generate fixes
python scripts/claude_fixer_corebench.py \
    --rubric-dir rubrics_output/corebench \
    --judge-csv judge_output/corebench_verdict.csv \
    --benchmark corebench_hard \
    --ife-only --tasks-per-batch 3

# Apply fixes
python scripts/run_corebench_fixes.py --list-fixes
python scripts/run_corebench_fixes.py --all-models --prefix fixed_ --docker --skip-rubrics
```

### ScienceAgentBench

```bash
# Generate fixes
python scripts/claude_fixer_scienceagentbench.py \
    --rubric-dir rubrics_output/scienceagentbench \
    --judge-csv judge_output/scienceagentbench_verdict.csv \
    --ife-only --tasks-per-batch 5

# Apply fixes
python scripts/run_scienceagentbench_fixes.py --list-fixes
python scripts/run_scienceagentbench_fixes.py --all-models --prefix fixed_ --docker
```

### ColBench

```bash
# Generate fixes
python scripts/claude_fixer_colbench.py \
    --rubric-dir rubrics_output/colbench \
    --judge-csv judge_output/colbench_verdict.csv \
    --ife-only --tasks-per-batch 5

# Apply fixes
python scripts/run_colbench_fixes.py --list-fixes
python scripts/run_colbench_fixes.py --all-models --prefix fixed_ --docker --parallel 3
```

---

## Model Configuration

Each benchmark has a `model_to_baseline_<benchmark>.json`:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3-2025-04-16",
    "short_name": "o3",
    "reasoning_effort": "medium",
    "max_steps": 5
  }
}
```

**Note**: `baseline_trace` is optional - you can run fixes without baseline traces using `--all-models` or `--model`.

---

## Fix Package Structure

Fixes are created in `fixes/<benchmark>/<task_id>/`:

```
fixes/<benchmark>/<task_id>/
├── README.md                   # Human-readable explanation
├── env_override.json           # Environment/dependencies
├── evaluation_override.json    # Criteria adjustments
├── instruction_override.json   # Clarifications
└── input_override.json         # Task input modifications
```

---

## Docker Configuration

### Network Mode

```bash
# For host-local proxy (Linux)
export HAL_DOCKER_NETWORK_MODE=host

# Or use host.docker.internal (macOS/Windows)
export OPENAI_BASE_URL=http://host.docker.internal:4000
```

### Debugging

```bash
# Debug network connectivity
export HAL_DOCKER_PREFLIGHT_NETWORK=1

# Force rebuild prepared images
export HAL_DOCKER_FORCE_REBUILD=1

# Check Docker host connection
export HAL_DOCKER_HOST=unix:///var/run/docker.sock
```

### Verify Docker Image

```bash
docker run --rm hal-agent-runner:latest bash -lc \
    'python --version && Rscript --version 2>&1 | head -1 && pandoc --version | head -1'
```

---

## Troubleshooting

### TLS/SSL Errors

```bash
pip install --upgrade --force-reinstall certifi click
```

### Docker Connection Issues

```bash
# Check Docker is running
docker ps

# Check Docker socket
ls -la /var/run/docker.sock

# Set Docker host explicitly
export HAL_DOCKER_HOST=unix:///var/run/docker.sock
```

### Azure Authentication Issues

```bash
# Re-authenticate
az login

# Check token
az account get-access-token --resource api://trapi/.default
```

### Model Not Found

```bash
# List available models in config
cat model_to_baseline_<benchmark>.json | jq 'keys'
```

### Conda Environment Issues

```bash
# Recreate environment
conda deactivate
conda env remove -n hal
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n hal python=3.12 -y
conda activate hal
pip install -r requirements.txt
pip install -e ./docent
pip install -e ./hal-harness
```

---

## Key Principle

**Make evaluation FAIR, not EASY.**

- Fix missing dependencies that ALL agents need
- Clarify ambiguous output format requirements
- Adjust overly strict numerical tolerances
- Do NOT give hints about solutions
- Do NOT simplify the problem
- Do NOT pre-compute partial results
