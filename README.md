# Agent Debug Pipeline

Automated Item Fixing Pipeline for benchmark evaluation. The core innovation is **item-level fixing** - modifying agent configurations, prompt templates, and runtime parameters to eliminate false-positive benchmark defects without changing the benchmark source code itself.

## Supported Benchmarks

| Benchmark             | Fixer Script                        | Runner Script                    | Model Config                               | Fixes |
| --------------------- | ----------------------------------- | -------------------------------- | ------------------------------------------ | ----- |
| **SciCode**           | `claude_fixer_scicode.py`           | `run_scicode_fixes.py`           | `model_to_baseline_scicode.json`           | 61    |
| **CoreBench**         | `claude_fixer_corebench.py`         | `run_corebench_fixes.py`         | `model_to_baseline_corebench.json`         | 13    |
| **ScienceAgentBench** | `claude_fixer_scienceagentbench.py` | `run_scienceagentbench_fixes.py` | `model_to_baseline_scienceagentbench.json` | 26    |
| **ColBench**          | `claude_fixer_colbench.py`          | `run_colbench_fixes.py`          | `model_to_baseline_colbench.json`          | 81    |
| **USACO**             | (manual)                            | `run_usaco_fixes.py`             | `model_to_baseline_usaco.json`             | 2     |

---

## Quick Start: Trace Extraction and Rubric Evaluation

### Standard Workflow (SciCode, CoreBench, ScienceAgentBench)

```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Step 1: Merge traces + extract from Weave
./FINAL_COMMANDS.sh scicode      # Or: corebench, sab

# Step 2: Run rubric evaluation
./RUBRIC_COMMANDS.sh scicode

# Step 3: Aggregate verdicts
./JUDGE_COMMANDS.sh scicode
```

### ColBench Special Workflow

ColBench uses a **different process** - dialogue history comes from `results/` directory, NOT Weave:

```bash
# Step 1: Merge traces + extract dialogues from results/
./FINAL_COMMANDS.sh colbench

# This automatically:
# 1. Merges individual traces
# 2. Extracts dialogue history from results/colbench_backend_programming/
# 3. Creates *_WITH_DIALOGUES.json files

# Step 2: Run rubric evaluation (uses _WITH_DIALOGUES.json files)
./RUBRIC_COMMANDS.sh colbench

# Step 3: Aggregate verdicts
./JUDGE_COMMANDS.sh colbench
```

**Why ColBench is different**: ColBench dialogue history is saved in `results/{run_id}/0/output.json`, not uploaded to Weave as LLM call traces. The `add_colbench_dialogues.py` script extracts these dialogues and embeds them in `raw_logging_results`.

### Available Flags

All scripts support running specific benchmarks:

```bash
./FINAL_COMMANDS.sh [all|scicode|corebench|sab|colbench]
./RUBRIC_COMMANDS.sh [all|scicode|corebench|sab|colbench]
./JUDGE_COMMANDS.sh [all|scicode|corebench|sab|colbench]
```

### Output Files

**After FINAL_COMMANDS.sh**:

- SciCode/CoreBench/SAB: `traces/{prefix}_openai_{model}.json` (with Weave logs)
- ColBench: `traces/{prefix}_openai_{model}_WITH_DIALOGUES.json`

**After RUBRIC_COMMANDS.sh**:

- `rubrics_output/{benchmark}/*.csv` (per-model rubric grades)

**After JUDGE_COMMANDS.sh**:

- `judge_output/{benchmark}_verdict.csv` (aggregated final verdicts)

---

## Google Sheets Upload

Some scripts (e.g., `scripts/build_response_matrix.py --upload`) push results to Google Sheets via `gspread`.

- Put a service account JSON at `~/.config/gspread/service_account.json`.
- Share the target spreadsheet with the service account email (e.g., `watcher-hal@lecole-0000.iam.gserviceaccount.com`).
- Never commit credentials to this repo.

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

```bash
# Delete the cached prepared image to force rebuild
docker images --format "{{.Repository}}:{{.Tag}}" | grep agent-env | xargs -r docker rmi -f

# Clear Docker build cache
docker builder prune -f
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

### Step 8: Setup TMPDIR and Azure Authentication

**CRITICAL**: Azure CLI requires a writable temporary directory. On shared machines, `/tmp` is often full.

#### Step 8.1: Check Disk Space and Setup TMPDIR

```bash
# Check if /tmp has space
df -h / | tail -1

# If /tmp is near 100% full, use your home directory instead
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 95 ]; then
    echo "⚠️  Root partition is >95% full, using ~/tmp instead of /tmp"
    mkdir -p ~/tmp
    chmod 700 ~/tmp
    TMPDIR_PATH=$(readlink -f ~/tmp)
else
    echo "✓ Root partition has space, using /tmp"
    TMPDIR_PATH="/tmp"
fi

# Set TMPDIR permanently
cat >> ~/.bashrc << EOF

# Fix Azure CLI tmpdir issues (required for authentication)
export TMPDIR=$TMPDIR_PATH
export TMP=$TMPDIR_PATH
export TEMP=$TMPDIR_PATH
EOF

# Apply to current shell
source ~/.bashrc

# Verify it's set
echo "TMPDIR is now: $TMPDIR"
```

#### Step 8.2: Update hal-harness/.env with TMPDIR

```bash
# Get the TMPDIR value
TMPDIR_PATH=$(readlink -f ~/tmp 2>/dev/null || echo "/tmp")

# Update .env.azure with correct TMPDIR
cat > hal-harness/.env << EOF
# Direct Azure/TRAPI Configuration (no LiteLLM proxy)
# Uses Azure CLI credentials (az login)

# Enable direct Azure mode
USE_DIRECT_AZURE=true

# Fix tmpdir issues
TMPDIR=$TMPDIR_PATH
TMP=$TMPDIR_PATH
TEMP=$TMPDIR_PATH

# TRAPI Configuration
TRAPI_ENDPOINT=https://trapi.research.microsoft.com/gcr/shared
TRAPI_API_VERSION=2024-12-01-preview
TRAPI_SCOPE=api://trapi/.default

# Azure Cognitive Services (for other Azure endpoints)
AZURE_ENDPOINT=https://msrasc-swe.cognitiveservices.azure.com/
AZURE_API_VERSION=2024-10-21
AZURE_SCOPE=https://cognitiveservices.azure.com/.default

# Dummy key - Azure AD auth doesn't need a real key
OPENAI_API_KEY=dummy

# Other settings
HAL_AUTOFIX_MODEL="azure/o3-mini"
LOG_LEVEL=DEBUG
HAL_DEBUGGER_LOG_LEVEL=DEBUG
WANDB_API_KEY=your-wandb-key
SERPAPI_API_KEY=DUMMY
HF_TOKEN=your-hf-token

# LiteLLM retry settings (used by smolagents internally)
LITELLM_NUM_RETRIES=35
LITELLM_REQUEST_TIMEOUT=600

# HAL task-level retry settings
HAL_RETRY_MAX_RETRIES=120
HAL_RETRY_MAX_DELAY=60

# Skip slow Weave trace downloads
HAL_SKIP_WEAVE_DOWNLOAD=true
EOF

# IMPORTANT: Update with your actual keys
echo "⚠️  Don't forget to update WANDB_API_KEY and HF_TOKEN in hal-harness/.env"
```

#### Step 8.3: Login to Azure

```bash
# Make sure TMPDIR is exported in current shell
export TMPDIR=$TMPDIR
export TMP=$TMPDIR
export TEMP=$TMPDIR

# Clear any stale tokens
az logout

# Login to Azure
az login

# Verify authentication
az account show

# Test token acquisition for Azure Cognitive Services
az account get-access-token --resource https://cognitiveservices.azure.com

# Verify MSAL cache is populated
ls -la ~/.azure/msal_token_cache.json
# Should show a recent timestamp

# The scripts will automatically use MSAL token cache for authentication
```

#### Step 8.4: Verify Setup

```bash
# Check all environment variables are set
echo "TMPDIR: $TMPDIR"
cat hal-harness/.env | grep TMPDIR

# Verify Azure authentication works
az account show

# Test token acquisition
az account get-access-token --resource https://cognitiveservices.azure.com | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Token acquired! Expires: {d[\"expiresOn\"]}')"

# If you see errors, check the troubleshooting section
```

**If authentication still fails**, check the [Azure troubleshooting section](#azure-authentication-no-usable-temporary-directory-found) for detailed diagnosis steps.

### Step 9: Decrypt Benchmark Data (CoreBench)

```bash
# Decrypt CoreBench test set
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
    --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
# Passphrase: reproducibility
```

### Step 10: Download Benchmark Datasets

#### SciCode Test Data

```bash
# Download test_data.h5 for SciCode evaluation
# Place in: hal-harness/hal/benchmarks/SciCode/eval/data/test_data.h5
mkdir -p hal-harness/hal/benchmarks/SciCode/eval/data
# Download from the SciCode dataset source
```

#### ScienceAgentBench Datasets

**CRITICAL**: ScienceAgentBench requires dataset files to be downloaded separately.

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download and extract ScienceAgentBench datasets (required for evaluation)
cd hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/benchmark/
gdown 1IYVRVK0TSZXRVKiSc2D0wxV1LXoXhpfA

# Extract with password: scienceagentbench
unzip -P scienceagentbench benchmark.zip

# Move files up one level (zip has nested benchmark/ directory)
mv benchmark/* .
rmdir benchmark

# Clean up zip file (optional, saves 1.7GB)
rm benchmark.zip

# Verify datasets directory exists
ls -la datasets/
# Should show 78 folders including: ocean_profiles/, temperature_statistic/, ligand_protein/, etc.

cd ../../../../..
```

**Note**: Without these datasets, ScienceAgentBench tasks will fail with "file not found" errors.

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
    --failed-only -y --max-batch-messages 1000

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

| Option              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `--list-fixes`      | List available fix directories and exit                  |
| `--dry-run`         | Preview what would be done without running               |
| `--verify-fixes`    | Verify fixes are applied correctly without running HAL   |
| `--all-models`      | Run all models from config for each task with fixes      |
| `--model MODEL`     | Force a specific model for all tasks                     |
| `--rubric-csv PATH` | Use rubric CSV to determine which model failed each task |
| `--task-id ID`      | Only run specific task(s), can be repeated               |
| `--prefix PREFIX`   | Prefix for run IDs and output files                      |
| `--docker`          | Run HAL evaluation with Docker                           |
| `--parallel N`      | Number of parallel HAL evaluations                       |
| `--skip-rubrics`    | Skip rubric evaluation after running                     |

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
python scripts/run_scicode_fixes.py --all-models --prefix scicode_mud --docker --parallel 120
```

### CoreBench

**Note**: CoreBench requires large capsule files (~2GB total). The fix runner automatically extracts them before running to prevent race conditions.

```bash
# List IFE tasks
python scripts/claude_fixer_corebench.py --rubric-dir rubrics_output/corebench --list-ife-tasks

# Generate fixes
python scripts/claude_fixer_corebench.py \
    --rubric-dir rubrics_output/corebench \
    --judge-csv judge_output/corebench_verdict.csv \
    --benchmark corebench_hard \
    --ife-only --tasks-per-batch 3

# Apply fixes (capsules auto-extracted, safe for high parallelism)
python scripts/run_corebench_fixes.py --list-fixes
python scripts/run_corebench_fixes.py --all-models --prefix qq1 --docker --skip-rubrics --max-parallel-capsules 100
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
python scripts/run_scienceagentbench_fixes.py --all-models --prefix sab_night --docker --parallel 100
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
python scripts/run_colbench_fixes.py --all-models --prefix col_bob --docker --parallel 100
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

**Quick Issue Finder**:

- [Evaluations hang after Weave initialization](#evaluations-hanging-after-weave-initialization) - No output after "Logged in as Weights & Biases user"
- [Azure "No usable temporary directory"](#azure-authentication-no-usable-temporary-directory-found) - FileNotFoundError with /tmp
- [Docker containers missing /tmp](#docker-containers-missing-tmp-directory) - Errors inside containers about temp directories
- [Expired MSAL token cache](#expired-msal-token-cache) - Azure authentication failing silently
- [High parallelism causing hangs](#high-parallelism-causing-system-instability) - System becomes unresponsive with many jobs
- [CoreBench capsule extraction race condition](#corebench-capsule-extraction-race-condition) - Already fixed in `run_corebench_fixes.py`
- [TLS/SSL errors](#tlsssl-errors) - Certificate verification issues
- [Docker connection issues](#docker-connection-issues) - Cannot connect to Docker daemon
- [Azure authentication general](#azure-authentication-issues) - General Azure login problems
- [Model not found](#model-not-found) - Model key not in config
- [Conda environment issues](#conda-environment-issues) - Package conflicts or import errors

---

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

### Evaluations Hanging After Weave Initialization

**Symptoms**: Verbose logs show Weave initialization succeeding, then the process hangs with no further output:

```
→ Initializing logging with W&B Weave...
Logged in as Weights & Biases user: <username>
View Weave data at https://wandb.ai/...
[HANGS HERE - no further output]
```

**Root Cause**: API connection issues - either:

1. No LiteLLM proxy running when `.env` is configured for proxy mode
2. Azure authentication failing when using direct Azure mode
3. Wrong environment configuration loaded

**Diagnosis**:

```bash
# Check which .env is being used
cat hal-harness/.env | head -5

# Check if using proxy mode (look for OPENAI_BASE_URL)
cat hal-harness/.env | grep OPENAI_BASE_URL

# Check if using direct Azure mode
cat hal-harness/.env | grep USE_DIRECT_AZURE

# Check if proxy is running (proxy mode only)
curl http://localhost:4000/health

# Check Azure authentication (Azure mode only)
az account show
```

**Solution 1: Fix .env Configuration**

If you see `OPENAI_BASE_URL=http://...` but no proxy is running, you have a mismatch:

```bash
# Option A: Start the proxy
./deploy_llm_cluster.sh

# Option B: Switch to Azure direct mode
cp hal-harness/.env.azure hal-harness/.env

# Option C: Use OpenAI API directly
cat > hal-harness/.env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
WANDB_API_KEY=your-wandb-key
HF_TOKEN=your-hf-token
SERPAPI_API_KEY=DUMMY
EOF

# Kill hanging processes
pkill -f "hal-eval"
pkill -f "run_.*_fixes"

# Re-run
python scripts/run_<benchmark>_fixes.py --all-models --prefix fixed_ --docker
```

**Solution 2: Fix Azure Authentication Issues**

If using direct Azure mode (`USE_DIRECT_AZURE=true`) and evaluations hang:

```bash
# Check Azure login status
az account show

# If not logged in, authenticate
az login

# Verify token acquisition works
az account get-access-token --resource api://trapi/.default

# If token acquisition fails, check MSAL cache
ls -la ~/.azure/msal_token_cache.json

# Re-login to refresh tokens
az logout
az login
```

### Azure Authentication: "No usable temporary directory found"

**Symptoms**:

```
AzureCliCredential.get_token_info failed:
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/usr/bin']
```

**Root Cause**: Python cannot detect `/tmp` directory due to missing or incorrect `TMPDIR` environment variable.

**Solution**:

```bash
# Check if your TMPDIR is set and writable
echo "Current TMPDIR: $TMPDIR"
touch $TMPDIR/test && rm $TMPDIR/test && echo "✓ TMPDIR is writable"

# Check Python's temp directory detection
python -c "import tempfile; print(tempfile.gettempdir())"

# If not set, follow Step 8.1 to set up TMPDIR dynamically
# (uses ~/tmp if disk is >95% full, otherwise /tmp)

# Verify TMPDIR is in .bashrc
grep TMPDIR ~/.bashrc

# Verify TMPDIR is in .env
grep TMPDIR hal-harness/.env

# Test Azure CLI works now
az account get-access-token --resource api://trapi/.default
```

### Docker Containers Missing /tmp Directory

**Symptoms**: Errors inside Docker containers about missing temporary directories, even though host has `/tmp`.

**Root Cause**: Docker containers don't inherit the host's `/tmp` filesystem by default.

**Solution**: The docker_runner.py has been updated to automatically mount a tmpfs for `/tmp`:

```python
# Already implemented in hal-harness/hal/utils/docker_runner.py
tmpfs = {"/tmp": "size=1G,mode=1777"}
```

If you see this issue in custom Docker runs, add the tmpfs mount:

```bash
docker run --tmpfs /tmp:size=1G,mode=1777 <image> <command>
```

### Expired MSAL Token Cache

**Symptoms**: Using Azure direct mode but authentication fails with:

```
✗ Token acquisition failed: None
```

**Diagnosis**:

```bash
# Check MSAL cache exists
ls -la ~/.azure/msal_token_cache.json

# Test token acquisition with Python
python << 'EOF'
import msal
import os

cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
cache = msal.SerializableTokenCache()
with open(cache_path, 'r') as f:
    cache.deserialize(f.read())

app = msal.PublicClientApplication(
    '04b07795-8ddb-461a-bbee-02f9e1bf7b46',
    authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
    token_cache=cache
)

accounts = app.get_accounts()
print(f'Found {len(accounts)} account(s)')
if accounts:
    result = app.acquire_token_silent(['api://trapi/.default'], account=accounts[0])
    if result and 'access_token' in result:
        print('✓ Token acquired successfully')
    else:
        print('✗ Token expired or invalid')
EOF
```

**Solution**:

```bash
# IMPORTANT: Ensure TMPDIR is set (required for az CLI)
# If not already set, follow Step 8.1 to set up TMPDIR
echo "Current TMPDIR: $TMPDIR"

# If TMPDIR is not set, set it temporarily
if [ -z "$TMPDIR" ]; then
    export TMPDIR=$(readlink -f ~/tmp 2>/dev/null || echo "/tmp")
    export TMP=$TMPDIR
    export TEMP=$TMPDIR
fi

# Re-authenticate to refresh MSAL cache
az logout
az login

# Verify it works
az account get-access-token --resource api://trapi/.default

# Ensure TMPDIR is permanent (should already be done in Step 8.1)
grep TMPDIR ~/.bashrc || echo "⚠️  TMPDIR not in .bashrc - follow Step 8.1"
```

### Complete Fix for Company Azure/TRAPI Setup

**If you're running large-scale evaluations and nothing works**, follow these steps in order:

```bash
# Step 1: Set up TMPDIR dynamically (see Step 8.1 for details)
# This automatically chooses ~/tmp if disk is >95% full, otherwise /tmp
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 95 ]; then
    mkdir -p ~/tmp
    chmod 700 ~/tmp
    TMPDIR_PATH=$(readlink -f ~/tmp)
else
    TMPDIR_PATH="/tmp"
fi

# Add to .bashrc if not already there
if ! grep -q "export TMPDIR=" ~/.bashrc; then
    cat >> ~/.bashrc << EOF
# Fix Azure CLI tmpdir issues
export TMPDIR=$TMPDIR_PATH
export TMP=$TMPDIR_PATH
export TEMP=$TMPDIR_PATH
EOF
fi
source ~/.bashrc

# Step 2: Add TMPDIR to hal-harness/.env if not already there
if ! grep -q "^TMPDIR=" hal-harness/.env; then
    cat >> hal-harness/.env << EOF

# Fix tmpdir issues
TMPDIR=$TMPDIR_PATH
TMP=$TMPDIR_PATH
TEMP=$TMPDIR_PATH
EOF
fi

# Step 3: Verify TMPDIR is writable
touch $TMPDIR/test && rm $TMPDIR/test && echo "✓ TMPDIR ($TMPDIR) works" || echo "✗ TMPDIR broken"

# Step 4: Re-authenticate Azure
az logout
az login

# Step 5: Verify authentication works
az account show
az account get-access-token --resource api://trapi/.default

# Step 6: Verify hal-harness is using Azure direct mode
cat hal-harness/.env | grep USE_DIRECT_AZURE
# Should output: USE_DIRECT_AZURE=true

# If not:
cp hal-harness/.env.azure hal-harness/.env

# Step 7: Kill any hung processes
pkill -f "hal-eval"
pkill -f "run_.*_fixes"

# Step 8: Test with a small job
python scripts/run_corebench_fixes.py \
    --task-id capsule-1624349 \
    --model openai/gpt-4.1-2025-04-14 \
    --docker \
    --skip-rubrics

# Step 9: If test works, run full evaluation
python scripts/run_corebench_fixes.py \
    --all-models \
    --prefix cb_azure \
    --docker \
    --skip-rubrics \
    --max-parallel-capsules 5
```

**Debug checklist if still failing**:

```bash
# ✓ Check 1: TMPDIR is set
echo $TMPDIR  # Should print your TMPDIR path (e.g., /tmp or ~/tmp)

# ✓ Check 2: Azure is logged in
az account show  # Should show your account

# ✓ Check 3: Can get tokens
az account get-access-token --resource api://trapi/.default  # Should return token

# ✓ Check 4: Using Azure mode
grep USE_DIRECT_AZURE hal-harness/.env  # Should be: true

# ✓ Check 5: No proxy URLs in .env
grep OPENAI_BASE_URL hal-harness/.env  # Should be empty or commented

# ✓ Check 6: Docker /tmp mount added
grep "tmpfs.*tmp" hal-harness/hal/utils/docker_runner.py  # Should find code

# ✓ Check 7: MSAL cache exists
ls -la ~/.azure/msal_token_cache.json  # Should exist with recent timestamp

# ✓ Check 8: TMPDIR matches between shell and .env
echo "Shell TMPDIR: $TMPDIR"
grep TMPDIR hal-harness/.env  # Should match
```

### High Parallelism Causing System Instability

**Symptoms**: Some evaluations complete, others hang, system becomes unresponsive.

**Root Cause**: Too many concurrent Docker containers or API calls overwhelming the system.

**Solution**:

```bash
# Check current load
docker ps | wc -l  # Number of running containers
ps aux | grep hal-eval | wc -l  # Number of HAL processes

# Kill hung processes
pkill -f "hal-eval"
pkill -f "run_.*_fixes"

# Reduce parallelism
python scripts/run_<benchmark>_fixes.py \
    --all-models \
    --prefix fixed_ \
    --docker \
    --max-parallel-capsules 3  # Lower this value (default is often too high)
```

**Recommended Parallelism Limits**:

- **Local machine**: `--max-parallel-capsules 3-5`
- **Azure VM (Standard_D4s_v3)**: `--max-parallel-capsules 5-10`
- **High-memory server**: `--max-parallel-capsules 10-20`

**Monitor Resource Usage**:

```bash
# Check CPU and memory usage
htop

# Check Docker resource usage
docker stats

# Check disk space (Docker images are large)
df -h
docker system df
```

---

## Key Principle

**Make evaluation FAIR, not EASY.**

- Fix missing dependencies that ALL agents need
- Clarify ambiguous output format requirements
- Adjust overly strict numerical tolerances
- Do NOT give hints about solutions
- Do NOT simplify the problem
- Do NOT pre-compute partial results 3. Unified Fix Runner (scripts/run_benchmark_fixes.py)

Single script works across all benchmarks:

```bash
# List configurations
python scripts/run_benchmark_fixes.py -b scicode --list-configs
# Run specific config
python scripts/run*benchmark_fixes.py -b scicode -c gpt-5_scicode_tool_calling -p test* --docker
# Run all configs with an agent
python scripts/run*benchmark_fixes.py -b scicode --agent scicode_tool_calling_agent -p test*
# Filter by model
python scripts/run*benchmark_fixes.py -b scicode --model-filter deepseek -p test*
# Run all
python scripts/run*benchmark_fixes.py -b scicode --all-configs -p iter1* --docker 4. Updated Agents
```

- colbench_example_agent - Now imports from shared module
- scicode_tool_calling_agent - Now uses shared module for Azure

How to Add New Agent Combinations

Just edit the JSON file:
"gpt-5*my_new_agent": {
"model_id": "openai/gpt-5_2025-08-07",
"short_name": "gpt-5-new",
"agent_dir": "hal-harness/agents/my_new_agent",
"agent_function": "main.run",
"max_steps": 5
}
ython scripts/run_benchmark_fixes.py -b scicode --model-filter deepseek -p test*

```bash
# Run all
python scripts/run*benchmark_fixes.py -b scicode --all-configs -p iter1* --docker
```

1. Updated Agents

- colbench_example_agent - Now imports from shared module
- scicode_tool_calling_agent - Now uses shared module for Azure

How to Add New Agent Combinations

Just edit the JSON file:

```json
"gpt-5_my_new_agent": {
"model_id": "openai/gpt-5_2025-08-07",
"short_name": "gpt-5-new",
"agent_dir": "hal-harness/agents/my_new_agent",
"agent_function": "main.run",
"max_steps": 5
}
```

```bash
# Run ALL tasks with fixes applied where available
python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs \
    --all-tasks --prefix scicode_sun1_ --docker --parallel-models 10 --parallel-tasks 5
python scripts/run_benchmark_fixes.py --benchmark scienceagentbench --all-configs \
    --all-tasks --prefix sab_sun1_ --docker --parallel-models 10 --parallel-tasks 5
python scripts/run_benchmark_fixes.py --benchmark corebench --all-configs \
    --all-tasks --prefix corebench_sun1_ --docker --parallel-models 10 --parallel-tasks 5
python scripts/run_benchmark_fixes.py --benchmark colbench --all-configs \
    --all-tasks --prefix colbench_sun1_ --docker --parallel-models 10 --parallel-tasks 5

./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs \
    --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 10
./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark scienceagentbench --all-configs \
    --all-tasks --prefix sab_moon1_ --docker --parallel-models 10 --parallel-tasks 10
./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark corebench --all-configs \
    --all-tasks --prefix corebench_moon1_ --docker --parallel-models 10 --parallel-tasks 10
./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark colbench --all-configs \
    --all-tasks --prefix colbench_moon1_ --docker --parallel-models 10 --parallel-tasks 10

# Run ALL benchmarks, ALL tasks
python scripts/run_benchmark_fixes.py --all-benchmarks --all-configs \
    --all-tasks --prefix star1_ --docker --parallel-models 10 --parallel-tasks 5

# Original behavior (only tasks with fixes)
python scripts/run_benchmark_fixes.py --benchmark scicode \
    --all-configs --prefix test_ --docker
```
