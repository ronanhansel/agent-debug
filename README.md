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

## Quick Start

### Environment Setup

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n agent python=3.12 -y
conda activate agent

# Install all requirements
pip install -r requirements.txt
pip install -e ./docent
pip install -e ./hal-harness
pip install --upgrade --force-reinstall certifi click

# Build Docker image
cd hal-harness
docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .
cd ..

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Full Pipeline (Any Benchmark)

```bash
# 1. Rubric evaluation
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

# 2. Judge aggregation
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

---

## Standardized CLI Interface

All fix runner scripts follow a consistent interface:

### Common Options

| Option | Description |
|--------|-------------|
| `--list-fixes` | List available fix directories and exit |
| `--dry-run` | Preview what would be done without running |
| `--verify-fixes` | Verify fixes are applied correctly without running HAL |
| `--all-models` | Run all models from config for each task with fixes |
| `--model MODEL` | Force a specific model for all tasks |
| `--task-id ID` | Only run specific task(s), can be repeated |
| `--prefix PREFIX` | Prefix for run IDs and output files |
| `--docker` | Run HAL evaluation with Docker |
| `--parallel N` | Number of parallel HAL evaluations |

### Usage Examples

```bash
# List what fixes are available
python scripts/run_scicode_fixes.py --list-fixes

# Dry run - see what would happen
python scripts/run_scicode_fixes.py --all-models --dry-run

# Run specific task with specific model
python scripts/run_scicode_fixes.py --task-id 11 --model openai/gpt-4.1-2025-04-14 --docker

# Run all models for all tasks with fixes
python scripts/run_scicode_fixes.py --all-models --prefix iter1_ --docker --parallel 3
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
python scripts/run_corebench_fixes.py --all-models --prefix fixed_ --docker
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

## Model Configuration Files

Each benchmark has a `model_to_baseline_<benchmark>.json`:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1_2025-04-14",
    "short_name": "gpt-4.1",
    "baseline_trace": "<benchmark>_..._UPLOAD.json",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3_2025-04-16",
    "short_name": "o3",
    "reasoning_effort": "medium",
    "max_steps": 5
  }
}
```

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

### Fix Types

**Environment Fixes** (`env_override.json`):
```json
{
  "HAL_CONDA_PACKAGES": "mne oggm",
  "HAL_PIP_PACKAGES": "biopsykit mastml",
  "HAL_APT_PACKAGES": "libfoo-dev",
  "HAL_TIMEOUT_SECONDS": 600
}
```

**Instruction Clarifications** (`instruction_override.json`):
```json
{
  "clarifications": [
    "Use scipy.integrate.simpson instead of deprecated simps",
    "Output file should be named results.csv"
  ]
}
```

---

## Legacy Commands

### Rubric Grading

```bash
for trace in traces/earth_*.json; do
    python main.py evaluate \
        --trace-file "$trace" \
        --rubrics-dir rubrics \
        --output-dir rubrics_output \
        --rubric-model gpt-5.2 \
        --output-mode csv \
        --failed-only \
        --json-mode \
        --yes
done
```

### CoreBench with Docker

```bash
python scripts/run_corebench_fixes.py \
    --fixes-root fixes/corebench_hard \
    --agent-dir hal-harness/agents/hal_generalist_agent \
    --benchmark corebench_hard \
    --all-models \
    --prefix iter1_ \
    --docker
```

### Master Rerun (CoreBench)

```bash
python scripts/master_rerun_corebench_fixes.py \
    --mapping-file model_to_baseline_corebench.json \
    --max-parallel 5 \
    --max-parallel-capsules 5 \
    --wandb-mode online \
    --docker \
    --skip-rubrics \
    --prefix moon
```

### Extract Weave Traces

```bash
python scripts/extract_weave_traces.py \
    --project <entity_id/project_id> \
    --prefix earth_openai_gpt-4_1 \
    --merge-input traces/earth_openai_gpt-4_1_MERGED_UPLOAD.json
```

---

## Docker Configuration

If Docker runs fail to connect to your model proxy:

```bash
# For host-local proxy (Linux)
export HAL_DOCKER_NETWORK_MODE=host

# Or use host.docker.internal (macOS/Windows)
export OPENAI_BASE_URL=http://host.docker.internal:4000

# Debug network connectivity
export HAL_DOCKER_PREFLIGHT_NETWORK=1

# Force rebuild prepared images
export HAL_DOCKER_FORCE_REBUILD=1
```

### Decrypting CoreBench Test Set

```bash
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
    --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
# Passphrase: reproducibility
```

---

## Troubleshooting

### TLS/SSL Errors

```bash
pip install --upgrade --force-reinstall certifi click
```

### Docker Connection Issues

```bash
# Check if Docker SDK honors contexts
export HAL_DOCKER_HOST=unix:///var/run/docker.sock

# Verify image has required tools
docker run --rm hal-agent-runner:latest bash -lc \
    'Rscript --version && pandoc --version | head -n 2'
```

### Model Not Found

```bash
# List available models in config
cat model_to_baseline_<benchmark>.json | jq 'keys'
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
