# CLAUDE.md

This file provides guidance to Claude Code when working with the HAL Agent Debugging repository.

## Project Overview

This repository implements an **Automated Item Fixing Pipeline** for benchmark evaluation. The core innovation is **item-level fixing**—modifying agent configurations, prompt templates, and runtime parameters to eliminate false-positive benchmark defects without changing the benchmark source code itself.

### Key Distinction: Item Fixing vs. Benchmark Fixing

| Approach | Target | Contribution |
|----------|--------|--------------|
| **Benchmark Fixing** | Modify benchmark source code | Contributes to benchmark repo |
| **Item Fixing** (our approach) | Modify agent/prompt/config at runtime | Novel self-contained methodology |

## Supported Benchmarks

| Benchmark | Agent Location | Rubric Template | Status |
|-----------|---------------|-----------------|--------|
| **SciCode** | `agents/scicode_tool_calling_agent/` | `rubric_templates/scicode.txt` | Active |
| **CoreBench** | `agents/hal_generalist_agent/` | `rubric_templates/corebench.txt` | Active |
| **SWE-bench** | `agents/SWE-agent/` | `rubric_templates/swebench.txt` | Ready |
| **USACO** | `agents/USACO/` | `rubric_templates/usaco.txt` | Ready |
| **AssistantBench** | `agents/assistantbench_browser_agent/` | `rubric_templates/assistantbench.txt` | Ready |
| **ScienceAgentBench** | `hal/benchmarks/scienceagentbench/` | `rubric_templates/scienceagentbench.txt` | Ready |
| **ColBench** | `agents/colbench_example_agent/` | `rubric_templates/colbench.txt` | Ready |

## Repository Structure

```
agent-debug/
├── CLAUDE.md                    # This file
├── hal-harness/                 # HAL evaluation harness (submodule)
│   ├── agents/                  # Agent implementations
│   │   ├── scicode_tool_calling_agent/  # SciCode agent
│   │   ├── SWE-agent/                   # SWE-bench agent
│   │   ├── USACO/                       # USACO agent
│   │   ├── assistantbench_browser_agent/ # AssistantBench agent
│   │   └── hal_generalist_agent/        # General purpose agent
│   └── hal/
│       ├── benchmarks/          # Benchmark implementations
│       │   ├── swebench.py
│       │   ├── usaco.py
│       │   ├── assistantbench.py
│       │   ├── scienceagentbench.py
│       │   └── scicode.py
│       └── debugger/            # Debugging pipeline components
│
├── scripts/                     # Pipeline scripts
│   ├── pipeline.py              # Unified pipeline CLI
│   ├── fixing_pipeline.py       # End-to-end fix pipeline
│   ├── eval_rubric.py           # Rubric evaluation
│   ├── judge.py                 # Verdict aggregation
│   └── claude_fixer.py          # Claude-based fix generation
│
├── rubric_templates/            # Rubric prompts for LLM graders
│   ├── scicode.txt              # SciCode benchmark rubric
│   ├── corebench.txt            # CoreBench benchmark rubric
│   ├── swebench.txt             # SWE-bench benchmark rubric
│   ├── usaco.txt                # USACO benchmark rubric
│   ├── assistantbench.txt       # AssistantBench benchmark rubric
│   ├── scienceagentbench.txt    # ScienceAgentBench benchmark rubric
│   └── colbench.txt             # ColBench benchmark rubric
│
├── rubrics_output/              # Rubric evaluation results (CSVs)
│   ├── scicode/
│   ├── corebench/
│   ├── swebench/
│   ├── usaco/
│   ├── assistantbench/
│   ├── scienceagentbench/
│   └── colbench/
│
├── traces/                      # Agent execution traces (JSON)
├── fixes/                       # Generated fix packages
├── judge_output/                # Aggregated verdicts
├── demonstrate/                 # Analysis and visualization
└── debug/                       # Per-task debug artifacts
```

## Key Concepts

### 1. Rubric Grading
LLM evaluates failed tasks against a rubric template:
- **Grade 0**: Agent capability issue (agent could have succeeded)
- **Grade 1**: Benchmark defect / Intrinsic Formation Error (no agent could succeed)

### 2. Verdict Aggregation
Multiple rubric evaluations across models are aggregated into a final verdict per task. Cross-model consensus reduces false positives.

### 3. Item-Level Fixes
Fixes are applied without modifying benchmark source:
- **Agent configuration** (`agent.py`): Compatibility shims, operator patches, import authorizations
- **Prompt templates**: Compatibility notes, deprecated API guidance
- **Rubric clarifications**: Exclusions for agent formatting errors

### 4. Run Prefixes (Naming Convention)
Runs are identified by fruit-based prefixes:

| Prefix | Status | Description |
|--------|--------|-------------|
| `*_hal_*`, `*_tool_*` | Before | Initial baseline runs |
| `*_potato_*`, `*_kiwi_*` | Before/Debug | Error investigation runs |
| `*_apple_*` | Intermediate | Partial fixes applied |
| `*_honey_*`, `*_tomato_*` | After | All fixes applied (final) |

---

## Benchmark-Specific Details

### SciCode
**Purpose**: Scientific computing code generation

**Agent**: `hal-harness/agents/scicode_tool_calling_agent/`
- Uses smolagents framework with CodeAgent
- Sandboxed Python interpreter with AUTHORIZED_IMPORTS whitelist
- Code output must be wrapped in ```python blocks

**Common Issues**:
- `scipy.integrate.simps` deprecated → use `simpson`
- `np.trapz` deprecated → use `np.trapezoid`
- `@` operator not supported in interpreter
- `heapq`, `numpy.random` not in AUTHORIZED_IMPORTS
- Quantum computing tasks need `dtype=complex`

**Fixer Script**: `scripts/claude_fixer_scicode.py`
- Diagnoses IFEs from rubric evaluations and agent traces
- Creates fixes in `fixes/scicode/<task_id>/`
- Handles deprecated API warnings, import authorizations, tolerance adjustments

**Evaluation**: Direct Python execution in sandbox

---

### SWE-bench
**Purpose**: Software engineering bug fixing

**Agent**: `hal-harness/agents/SWE-agent/`
- Custom bash-like commands for file navigation
- YAML-based configuration in `config/`
- Cost limit per instance (default: $3.00)

**Common Issues**:
- Repository checkout state mismatches
- Test suite has non-deterministic behavior
- Patch format errors (unified diff)
- Environment/conda setup failures

**Known Research-Documented Issues** (SWE-bench Verified still has problems):
- **Weak Test Coverage (~31%)**: Tests only check modified files, incorrect fixes can pass
- **Flaky Tests**: recursion depth errors, timing issues, external service dependencies (httpbin.org)
- **Log Parsing Issues**: Django "System check" output breaks test result parsing
- **Overly Specific Tests**: Valid alternative implementations may fail
- **Divergent Valid Solutions (~47%)**: Gold patch may be one of many correct approaches

**Rubric Features** (`rubric_templates/swebench.txt`):
- Cross-run validation (if one model finds files, "missing files" claims are rejected)
- Exploratory analysis section for discovering novel benchmark issues
- Known problematic task patterns (Django, Requests, SymPy, Matplotlib)
- Questions to guide LLM reasoning about potential IFEs

**Evaluation**: Docker-based, applies patch and runs test suite

**HAL Command**:
```bash
hal-eval --benchmark swebench_verified_mini \
    --agent_dir agents/SWE-agent/ \
    --agent_function main.run \
    --agent_name "SWE-agent" \
    -A model_name=gpt-4o
```

---

### USACO
**Purpose**: Competitive programming (USA Computing Olympiad)

**Agent**: `hal-harness/agents/USACO/`
- 307 problems from USACO subset
- Supports retrieval (similar problems, textbook excerpts)
- Optional reflexion (iterative improvement with test feedback)
- BM25 + semantic retrieval for context

**Common Issues**:
- Time/memory limits too strict
- Test case format differs from spec
- Retrieved examples misleading
- Judge floating-point precision

**Evaluation**: Docker-based isolated execution

**HAL Command**:
```bash
hal-eval --benchmark usaco \
    --agent_dir agents/USACO/ \
    --agent_function main.run \
    --agent_name "USACO Agent" \
    --docker \
    -A model_name=gpt-4o
```

---

### AssistantBench
**Purpose**: Web navigation and information retrieval

**Agent**: `hal-harness/agents/assistantbench_browser_agent/`
- Uses `browser_use` library for automation
- Playwright-based browser control
- Supports OpenAI, Anthropic, OpenRouter

**Common Issues**:
- Websites changed since task creation
- CAPTCHA/bot detection blocks access
- Information is outdated (prices, dates)
- Navigation failures misclassified as benchmark issues

**Evaluation**: Compares extracted answer to gold answer

**HAL Command**:
```bash
hal-eval --benchmark assistantbench \
    --agent_dir agents/assistantbench_browser_agent/ \
    --agent_function main.run \
    --agent_name "Browser Agent" \
    -A model_name=gpt-4o
```

---

### ScienceAgentBench
**Purpose**: Scientific data-driven discovery

**Agent**: `hal-harness/agents/hal_generalist_agent/`
- 102 tasks across 4 scientific domains (bioinformatics, chemistry, GIS, psychology)
- Uses smolagents CodeAgent with sandboxed PythonInterpreterTool
- Docker-based execution with prepared images

**CRITICAL: Evaluation Flow**
```
1. Agent runs in Docker with smolagents CodeAgent
2. Agent generates code wrapped in ```python blocks
3. recover_pred_from_log.py EXTRACTS code via regex: r"```python(.*?)```"
4. Extracted code runs in SEPARATE Docker container
5. Results compared to gold program output
```
**Key insight**: Sandbox "Forbidden function" errors are IRRELEVANT to final score!
The agent just needs to OUTPUT valid code, not EXECUTE it successfully in sandbox.

**Docker/File Path Setup**:
- Prepared images cached by hash of: `requirements.txt + base_image_id + template_version`
- Files copied to `environment/benchmark/datasets/` (NOT `benchmark/datasets/`)
- Working directory is `/workspace/environment/` (set by `run_agent.py` chdir)
- If files not found, check that `scienceagentbench.py` copies to correct destination

**Required Packages** (in `agents/hal_generalist_agent/requirements.txt`):
- Core: matplotlib, numpy, scipy, scikit-learn, h5py, tables
- Bioinformatics: scanpy, anndata, mudata, muon, leidenalg, biopython
- Neuroimaging: mne, neurokit2, biopsykit
- Chemistry: rdkit, deepchem, pubchempy, pymatgen
- Geospatial: scitools-iris, cartopy, geopandas, xarray, geoplot, eofs, rasterio, fiona
- Cognitive: ccobra
- ML: catboost

**AUTHORIZED_IMPORTS** (in `agents/hal_generalist_agent/main.py`):
All packages above must also be in AUTHORIZED_IMPORTS for smolagents sandbox.

**Common Issues**:
- **File not found**: Check file destination vs working directory mismatch
- **Docker cache**: Delete old images (`docker images | grep agent-env`) and clear build cache
- **API timeout**: Reduce parallelism (use `--parallel 5` instead of 20)
- **Sandbox forbidden**: These errors are IRRELEVANT - code is extracted via regex

**Known Research-Documented Issues** (from benchmark paper and trace analysis):
- **High Universal Failure Rate**: 67/102 tasks failed across ALL 4 models (GPT-4.1, O3, O4-mini)
- **Figure Evaluation Subjectivity**: GPT-4 judge penalizes color schemes, axis labels, layout differences
- **Domain Library Gaps**: Packages like `oggm`, `mastml`, `mne`, `biopsykit` needed
- **Acknowledged Evaluation Noise**: Authors note "subjective variance in color, scale, and labeling"
- **Partial Credit Issues**: "Functionally equivalent implementations may receive lower scores"

**Rubric Features** (`rubric_templates/scienceagentbench.txt`):
- Cross-run analysis with specific failure statistics
- Figure/visualization evaluation issues category
- Domain-specific library availability investigation
- Known problematic task patterns (Tasks 74, 43, 2 for environment issues)
- Exploratory questions for subjective evaluation fairness
- **CRITICAL EXCLUSION**: Sandbox "Forbidden function" errors (code is extracted, not executed)

**Fixer Script**: `scripts/claude_fixer_scienceagentbench.py`
- Diagnoses IFEs from rubric evaluations and agent traces
- Creates fixes in `fixes/scienceagentbench/<task_id>/`
- Handles missing domain libraries, figure evaluation adjustments, tolerance settings

**Fix Runner**: `scripts/run_scienceagentbench_fixes.py`
- Applies fixes and re-runs evaluation
- Uses `model_to_baseline_scienceagentbench.json` for model configs

**Metrics**: Valid Execution Rate (VER), Success Rate (SR), CodeBERTScore (CBS)

**HAL Command**:
```bash
hal-eval --benchmark scienceagentbench \
    --agent_dir agents/hal_generalist_agent/ \
    --agent_function main.run \
    --agent_name "sab_prefix_" \
    --docker \
    --max_concurrent 5 \
    -A model_name=openai/gpt-4.1-2025-04-14 \
    -A max_steps=5
```

**Debugging Docker Issues**:
```bash
# Check running containers
docker ps --format "{{.ID}}\t{{.Status}}"

# Check files in container
docker exec <container_id> bash -c "pwd && ls -la benchmark/datasets/"

# Force rebuild (delete cache)
docker images | grep agent-env | awk '{print $3}' | xargs -r docker rmi -f
docker builder prune -af
```

---

### CoreBench
**Purpose**: Scientific reproducibility

**Agent**: `hal-harness/agents/hal_generalist_agent/`
- Capsule-based evaluation (capsule-XXXXXXX task IDs)
- Requires Docker for isolated execution

**Common Issues**:
- Conda Terms of Service blocks
- Missing system headers (ft2build.h, cmake)
- Docker container crashes
- Sandbox permission denials

**Evaluation**: Docker container execution

---

### ColBench
**Purpose**: Collaborative agent-human interaction for coding tasks

**Agent**: `hal-harness/agents/colbench_example_agent/`
- Multi-turn dialogue with simulated human (GPT-4o)
- Two task variants: `colbench_backend_programming` (1000 tasks) and `colbench_frontend_design` (100 tasks)
- Agent discovers requirements through dialogue (up to 10 turns)
- Hidden information only known to simulated user

**Architecture**:
1. Agent receives problem description
2. Agent asks clarifying questions via dialogue
3. Simulated user (GPT-4o) responds based on hidden information
4. Agent provides final answer (code or HTML design)
5. Evaluation: test cases (backend) or CLIP image similarity (frontend)

**Common Issues**:
- Simulated user gives contradictory feedback
- Hidden information contains arbitrary implementation details
- Test cases verify behavior not specified in dialogue
- CLIP similarity penalizes valid alternative designs
- 10-turn limit insufficient for complex tasks

**Known Research Context** (SWEET-RL paper, Zhou et al. 2025):
- ColBench introduced to study multi-turn RL algorithms
- Designed for task diversity and complexity
- SWEET-RL achieves 6% improvement over baselines

**Rubric Features** (`rubric_templates/colbench.txt`):
- Simulated user response quality analysis
- Hidden information accessibility investigation
- Test case fairness evaluation
- CLIP evaluation bias detection
- Cross-model failure pattern identification

**Fixer Script**: `scripts/claude_fixer_colbench.py`
- Diagnoses IFEs from rubric evaluations and dialogue traces
- Creates fixes in `fixes/colbench/<task_id>/`
- Fix types: `instruction_override.json`, `evaluation_override.json`, `simulated_user_override.json`

**Fix Runner**: `scripts/run_colbench_fixes.py`
- Applies fixes and re-runs evaluation for ColBench tasks
- Uses `model_to_baseline_colbench.json` for model configs (3 models: gpt-4.1, o3-low, o4-mini-high)
- **Default benchmark**: `colbench_frontend_design` (100 tasks with CLIP similarity evaluation)
- **`--all-models` flag**: Runs each task with ALL models from config (creates jobs for every task × model combination)
- **Parallel execution**: Use `--parallel N` to run N jobs concurrently
- Model config includes separate `baseline_traces` for backend and frontend variants

**Fix Runner Usage**:
```bash
# List available fixes
python scripts/run_colbench_fixes.py --list-fixes

# Run fixes for a specific task with all models (with Docker)
python scripts/run_colbench_fixes.py --task-id 1 --all-models --prefix fixed_ --docker

# Run all fixes with all models in parallel (with Docker)
python scripts/run_colbench_fixes.py --all --all-models --parallel 3 --prefix iter1_ --docker

# Run specific task with specific model (with Docker)
python scripts/run_colbench_fixes.py --task-id 1 --model gpt-4.1-2025-04-14 --prefix test_ --docker

# Dry run to see what would execute
python scripts/run_colbench_fixes.py --all --all-models --dry-run
```

**Evaluation**:
- Backend: Test case execution (scores 0-1, partial credit)
- Frontend: CLIP similarity to ground truth image (scores 0-1)

**HAL Commands**:
```bash
# Backend programming
hal-eval --benchmark colbench_backend_programming \
    --agent_dir agents/colbench_example_agent/ \
    --agent_function main.run \
    --agent_name "ColBench Backend" \
    -A model_name=gpt-4.1-2025-04-14 \
    -A budget=1000

# Frontend design
hal-eval --benchmark colbench_frontend_design \
    --agent_dir agents/colbench_example_agent/ \
    --agent_function main.run \
    --agent_name "ColBench Frontend" \
    -A model_name=gpt-4.1-2025-04-14
```

**Model Configuration**: `model_to_baseline_colbench.json`

---

## Common Commands

### Pipeline Operations

```bash
# Full fixing pipeline (any benchmark)
python scripts/fixing_pipeline.py \
    --benchmark-name <benchmark> \
    --model claude \
    --trace-dir traces/

# Rubric evaluation only
python scripts/pipeline.py rubric \
    --prefix honey \
    --trace-file traces/<benchmark>_<prefix>_UPLOAD.json \
    --rubrics-dir rubric_templates/

# Judge (verdict aggregation)
python scripts/judge.py \
    --rubric-dir rubrics_output/<benchmark> \
    --output judge_output/<benchmark>_verdict.csv

# Cross-model rubric evaluation
python scripts/pipeline.py cross-rubric \
    --traces traces/<benchmark>_*.json \
    --model gpt-5.2
```

### HAL Evaluation Commands

```bash
# SciCode
hal-eval --benchmark scicode \
    --agent_dir agents/scicode_tool_calling_agent/ \
    --agent_function agent.run

# SWE-bench (requires Docker)
hal-eval --benchmark swebench_verified_mini \
    --agent_dir agents/SWE-agent/ \
    --agent_function main.run

# USACO (requires Docker)
hal-eval --benchmark usaco \
    --agent_dir agents/USACO/ \
    --agent_function main.run \
    --docker

# AssistantBench
hal-eval --benchmark assistantbench \
    --agent_dir agents/assistantbench_browser_agent/ \
    --agent_function main.run

# ScienceAgentBench
hal-eval --benchmark scienceagentbench \
    --agent_dir hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/ \
    --agent_function agent.run
```

### Analysis

```bash
# Generate verdict comparison figures
python demonstrate/analyze_verdicts.py

# Generate detailed rubric analysis
python demonstrate/analyze_rubrics_detailed.py
```

---

## Workflow for New Benchmarks

**See `INSTRUCTIONS_NEW_BENCHMARK.md` for detailed step-by-step instructions.**

Quick summary:

1. **Explore benchmark**: Understand agent structure, evaluation method, task format
2. **Research known issues**: Use WebSearch for papers, GitHub issues, community discussions
3. **Create rubric template**: `rubric_templates/<benchmark>.txt`
4. **Analyze traces** (if available): Identify failure patterns, cross-model correlations
5. **Create fixer script**: `scripts/claude_fixer_<benchmark>.py`
6. **Create output directories**: `mkdir -p rubrics_output/<benchmark> fixes/<benchmark>`
7. **Update CLAUDE.md**: Add benchmark details and fixer reference
8. **Run rubric evaluation**:
   ```bash
   python scripts/eval_rubric.py \
       --trace-file traces/<benchmark>_*.json \
       --rubric rubric_templates/<benchmark>.txt \
       --rubric-model openai:gpt-4o \
       --failed-only -y
   ```
9. **Aggregate verdicts**:
   ```bash
   python scripts/judge.py \
       --rubric-dir rubrics_output/<benchmark> \
       --output judge_output/<benchmark>_verdict.csv
   ```
10. **Run fixer script**:
    ```bash
    python scripts/claude_fixer_<benchmark>.py --all-ife --batch
    ```
11. **Apply fixes and re-run**: Use new prefix (e.g., `honey`)
12. **Compare before/after**: Measure improvement

---

## Rubric Template Structure

All rubric templates follow the same structure:

```
# <Benchmark> Intrinsic Formation Error Detection Rubric

## Purpose
## Scoring (0 or 1)
## Two-Question Framework
  - Question 1: Does defect exist?
  - Question 2: Did defect cause failure?

## Deficiency Categories (benchmark-specific)
## CRITICAL EXCLUSIONS: Agent Capability Issues
## Evidence Requirements
## Response Format (JSON)
## Common Failure Patterns
```

### Unified Schema

All rubrics use a unified JSON schema (`rubric_templates/rubric.schema.json`):

```json
{
  "score": 0 or 1,
  "deficiency_exists": boolean,
  "deficiency_caused_failure": boolean,
  "deficiency_type": "string",
  "existence_reasoning": "string",
  "causation_reasoning": "string",
  "evidence": "string"
}
```

The schema loading priority is:
1. Benchmark-specific schema (e.g., `swebench.schema.json`) - if exists
2. Unified schema (`rubric.schema.json`) in the rubric directory
3. Default simple schema (score + explanation only)

---

## Fixer Scripts

Fixer scripts use Claude Code CLI (`claude -p`) to automatically diagnose IFEs and generate fix packages.

### Available Fixer Scripts

| Benchmark | Fixer Script | Status |
|-----------|--------------|--------|
| **SciCode** | `scripts/claude_fixer_scicode.py` | Active |
| **ScienceAgentBench** | `scripts/claude_fixer_scienceagentbench.py` | Active |
| **ColBench** | `scripts/claude_fixer_colbench.py` | Ready |

### Fixer Script Usage

```bash
# List tasks with IFEs detected (Grade=1)
python scripts/claude_fixer_<benchmark>.py --list-ife-tasks

# Fix a specific task
python scripts/claude_fixer_<benchmark>.py --task-id <id>

# Fix all IFE tasks
python scripts/claude_fixer_<benchmark>.py --all-ife

# Batch mode (multiple tasks per Claude session - more efficient)
python scripts/claude_fixer_<benchmark>.py --all-ife --batch

# Skip tasks with existing fixes
python scripts/claude_fixer_<benchmark>.py --all-ife --skip-existing
```

### Fix Package Structure

Fixes are created in `fixes/<benchmark>/<task_id>/`:

```
fixes/<benchmark>/<task_id>/
├── README.md                   # Human-readable explanation of the fix
├── env_override.json           # Environment/dependency fixes
├── evaluation_override.json    # Evaluation criteria adjustments
├── instruction_override.json   # Task instruction clarifications
└── status.json                 # Fix application status
```

### Fix Types

1. **Environment Fixes** (`env_override.json`):
   ```json
   {
     "HAL_CONDA_PACKAGES": "oggm mne",
     "HAL_PIP_PACKAGES": "biopsykit mastml"
   }
   ```

2. **Evaluation Fixes** (`evaluation_override.json`):
   ```json
   {
     "tolerance": 1e-4,
     "allow_alternative_methods": true,
     "notes": "Justification for adjustment"
   }
   ```

3. **Instruction Clarifications** (`instruction_override.json`):
   ```json
   {
     "clarifications": [
       "Use scipy.integrate.simpson instead of deprecated simps",
       "Output file should be named results.csv"
     ]
   }
   ```

### Key Principle: Fair, Not Easy

Fixer scripts follow the principle: **Make evaluation FAIR, not EASY**.

- ✅ Fix missing dependencies that ALL agents need
- ✅ Clarify ambiguous output format requirements
- ✅ Adjust overly strict numerical tolerances
- ❌ Do NOT give hints about solutions
- ❌ Do NOT simplify the problem
- ❌ Do NOT pre-compute partial results

---

## Fix Runner Scripts

Fix runner scripts apply the generated fixes and re-run HAL evaluations.

### Available Fix Runner Scripts

| Benchmark | Fix Runner Script | Status |
|-----------|------------------|--------|
| **SciCode** | `scripts/run_scicode_fixes.py` | Active |
| **ScienceAgentBench** | `scripts/run_scienceagentbench_fixes.py` | Active |
| **CoreBench** | `scripts/run_corebench_fixes.py` | Active |
| **USACO** | `scripts/run_usaco_fixes.py` | Active |
| **ColBench** | `scripts/run_colbench_fixes.py` | Active |

### Fix Runner Usage

```bash
# List available fixes
python scripts/run_<benchmark>_fixes.py --list-fixes

# Dry run - see what would happen
python scripts/run_<benchmark>_fixes.py --task-id <id> --dry-run

# Run fixes for specific tasks
python scripts/run_<benchmark>_fixes.py --task-id 11 --task-id 12 --prefix fixed_

# Run all available fixes
python scripts/run_<benchmark>_fixes.py --all --prefix iter1_ --docker

# Run with specific model
python scripts/run_<benchmark>_fixes.py --prefix iter1_ --model gpt-4o
```

### What Fix Runners Do

1. **Load model config** from `model_to_baseline_<benchmark>.json`
2. **Load fixes** from `fixes/<benchmark>/<task_id>/`
3. **Apply environment overrides** (conda/pip packages, system libs, timeouts)
4. **Inject instruction clarifications** into task prompts
5. **Adjust evaluation parameters** (tolerances, alternative formats)
6. **Run HAL evaluation** with the original failing model
7. **Output new traces** with configurable prefix for comparison

### Model Configuration Files

Each benchmark has a `model_to_baseline_<benchmark>.json` that maps model IDs to configurations:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "baseline_trace": "scicode_agent_gpt4120250414_UPLOAD.json",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3-2025-04-16",
    "short_name": "o3",
    "baseline_trace": "scicode_agent_o320250416_UPLOAD.json",
    "reasoning_effort": "medium",
    "max_steps": 5
  }
}
```

This allows fix runners to:
- Re-run failed tasks with the **same model** that originally failed
- Apply model-specific settings (reasoning effort, max steps)
- Reference baseline traces for comparison

---

## Environment Setup

```bash
# Conda environment
conda activate hal

# Required environment variables
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export WANDB_API_KEY=...  # For Weave logging

# Install dependencies
pip install -e hal-harness/
pip install -e hal-harness/[scicode]
pip install -e hal-harness/[swebench]
pip install -e hal-harness/[usaco]
pip install -e hal-harness/[assistantbench]
```

---

## Azure/TRAPI Direct Access

This section documents the direct Azure/TRAPI access implementation, which bypasses LiteLLM proxy for faster and more reliable API calls.

### TRAPI Configuration

**Endpoint and Authentication**:
```
TRAPI_ENDPOINT=https://trapi.research.microsoft.com/gcr/shared
TRAPI_API_VERSION=2025-03-01-preview  # Required for gpt-5.2 and newer models
TRAPI_SCOPE=api://trapi/.default
```

**Azure CLI Constants**:
```python
AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'
```

### TRAPI Deployment Name Mapping

Model names must be mapped to TRAPI deployment names:

```python
# NOTE: Verify deployment availability on TRAPI before using!
# gpt-5.2 VERIFIED WORKING: gpt-5.2_2025-12-11 (requires api_version 2025-03-01-preview)
TRAPI_DEPLOYMENT_MAP = {
    # GPT-5 series (VERIFIED WORKING: gpt-5_2025-08-07, gpt-5.2_2025-12-11)
    # NOTE: GPT-5 uses max_completion_tokens like O-series
    'gpt-5': 'gpt-5_2025-08-07',
    'gpt-5-mini': 'gpt-5-mini_2025-08-07',
    'gpt-5-nano': 'gpt-5-nano_2025-08-07',
    'gpt-5-pro': 'gpt-5-pro_2025-10-06',
    'gpt-5.2': 'gpt-5.2_2025-12-11',  # VERIFIED WORKING (2026-01-19)
    'gpt-5.2-chat': 'gpt-5.2-chat_2025-12-11',

    # GPT-4 series (VERIFIED WORKING)
    'gpt-4o': 'gpt-4o_2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini_2024-07-18',
    'gpt-4.1': 'gpt-4.1_2025-04-14',
    'gpt-4.1-mini': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano_2025-04-14',

    # O-series (reasoning models) (VERIFIED WORKING)
    'o1': 'o1_2024-12-17',
    'o1-mini': 'o1-mini_2024-09-12',
    'o3': 'o3_2025-04-16',
    'o3-mini': 'o3-mini_2025-01-31',
    'o4-mini': 'o4-mini_2025-04-16',

    # GPT-5.1 series (may not be available)
    'gpt-5.1': 'gpt-5.1_2025-11-13',
    'gpt-5.1-chat': 'gpt-5.1-chat_2025-11-13',
    'gpt-5.1-codex': 'gpt-5.1-codex_2025-11-13',

    # Other models
    'deepseek-r1': 'deepseek-r1_1',
    'grok-3.1': 'grok-3_1',
    'llama-3.3': 'gcr-llama-33-70b-shared',
}
```

### Authentication Methods (Priority Order)

1. **MSAL Token Provider** (Works in Docker without az CLI):
   ```python
   import msal
   cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
   cache = msal.SerializableTokenCache()
   with open(cache_path, 'r') as f:
       cache.deserialize(f.read())
   app = msal.PublicClientApplication(
       AZURE_CLI_CLIENT_ID,
       authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
       token_cache=cache
   )
   accounts = app.get_accounts()
   result = app.acquire_token_silent([scope], account=accounts[0])
   token = result['access_token']
   ```

2. **Pre-fetched Token** (Fallback for containers):
   ```python
   token = os.environ.get('AZURE_OPENAI_AD_TOKEN')
   ```

3. **azure-identity** (Requires az CLI):
   ```python
   from azure.identity import AzureCliCredential, get_bearer_token_provider
   credential = AzureCliCredential()
   token_provider = get_bearer_token_provider(credential, scope)
   ```

### Azure OpenAI API Differences

**Parameter Differences by Model Type**:

| Model Type | Temperature | Stop | Max Tokens Parameter |
|------------|-------------|------|---------------------|
| GPT-4.x | `temperature=0.7` | Supported | `max_tokens=4096` |
| GPT-5.x | **Not supported** (only default=1) | May vary | `max_completion_tokens=4096` |
| O1, O3, O4 | Not supported | Limited | `max_completion_tokens=4096` |

**IMPORTANT**:
- GPT-5 uses `max_completion_tokens` like O-series, NOT `max_tokens`
- GPT-5 only supports `temperature=1` (default) - do NOT pass temperature parameter

```python
def _uses_max_completion_tokens(model_id: str) -> bool:
    """O-series and GPT-5 use max_completion_tokens instead of max_tokens."""
    model_lower = model_id.lower()
    # O-series models
    if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return True
    # GPT-5 also uses max_completion_tokens
    if model_lower.startswith("gpt-5"):
        return True
    return False

def _supports_temperature(model_id: str) -> bool:
    """O-series and GPT-5 models don't support temperature parameter."""
    model_lower = model_id.lower()
    if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return False
    # GPT-5 only supports temperature=1 (default)
    if model_lower.startswith("gpt-5"):
        return False
    return True
```

### smolagents Message Role Compatibility

smolagents uses non-standard role names that must be converted for Azure OpenAI:

```python
# smolagents roles -> Azure OpenAI roles
'tool-call' -> 'assistant'  # Model's tool invocation
'tool-response' -> 'user'   # Tool output (can't use 'tool' without proper tool_calls structure)
```

**CRITICAL**: Cannot use `'tool'` role directly because Azure OpenAI requires:
- A preceding assistant message with `tool_calls` array
- The tool message must have `tool_call_id` matching one of the tool calls

Since smolagents doesn't provide this structure, convert `'tool-response'` to `'user'` instead.

### Docker Integration

**Volume Mounts for Azure Credentials** (in `docker_runner.py`):
```python
azure_dir = os.path.expanduser("~/.azure")
if os.path.isdir(azure_dir) and env_vars.get("USE_DIRECT_AZURE", "").lower() == "true":
    volumes[azure_dir] = {"bind": "/root/.azure", "mode": "ro"}
    env_vars["HOME"] = "/root"  # CRITICAL: MSAL looks for cache at $HOME/.azure
```

**Environment Variables for Direct Azure**:
```python
env_vars["USE_DIRECT_AZURE"] = "true"
# Remove proxy URLs to prevent conflicts
for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "LITELLM_BASE_URL"):
    env_vars.pop(key, None)
```

**Pre-fetch Token on Host** (for containers without MSAL):
```python
from azure.identity import AzureCliCredential, get_bearer_token_provider
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, scope)
token = token_provider()
env_vars["AZURE_OPENAI_AD_TOKEN"] = token
```

### Implementation Files

| File | Purpose |
|------|---------|
| `hal-harness/agents/hal_generalist_agent/azure_direct_model.py` | Direct Azure model for smolagents |
| `hal-harness/hal/utils/docker_runner.py` | Docker integration with Azure credential mounting |
| `scripts/eval_rubric.py` | Rubric evaluation with Azure default |

### eval_rubric.py Behavior

**Default**: Uses Azure/TRAPI directly (no proxy needed)
```bash
python scripts/eval_rubric.py \
    --trace-file traces/*.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
# Output:
# [Azure] Direct TRAPI access configured
# [Azure] Model resolved: openai:gpt-5.2 -> azure_openai:gpt-5.2_2025-11-20
```

**CRITICAL**: The script automatically switches from `openai:` to `azure_openai:` provider because:
- `openai` provider uses `AsyncOpenAI()` with URL format: `{base_url}/chat/completions`
- `azure_openai` provider uses `AsyncAzureOpenAI()` with URL format: `{endpoint}/openai/deployments/{model}/chat/completions`
- TRAPI requires the Azure URL format; using the wrong format results in "The request is blocked" error

**With Proxy** (overrides Azure default, keeps original provider):
```bash
python scripts/eval_rubric.py \
    --trace-file traces/*.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-4o \
    --openai-base-url http://localhost:5000/v1 \
    --failed-only -y
# Output: [Proxy] Using custom endpoint
# Model stays as openai:gpt-4o (no provider switch)
```

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `PermissionDeniedError: The request is blocked` | Wrong URL format (openai vs azure_openai) | Use `azure_openai:` provider, not `openai:` |
| `BadRequestError: Invalid value: 'tool-call'` | smolagents non-standard roles | Convert to 'assistant'/'user' |
| `BadRequestError: 'max_tokens' not supported` | O-series model | Use `max_completion_tokens` |
| `BadRequestError: messages with role 'tool' must be response to tool_calls` | Missing tool_calls structure | Convert 'tool-response' to 'user' |
| `No accounts found in MSAL cache` | Wrong HOME in container | Set `HOME=/root` |
| `MSAL token refresh failed` | Expired refresh token | Re-authenticate with `az login` |
| `SharedTokenCacheCredential not available` | Container missing credentials | Mount `~/.azure` to `/root/.azure` |

### Testing Azure Connection

```bash
# Test token acquisition
python -c "
from azure.identity import AzureCliCredential, get_bearer_token_provider
credential = AzureCliCredential()
provider = get_bearer_token_provider(credential, 'api://trapi/.default')
token = provider()
print(f'Token acquired: {len(token)} chars')
"

# Test MSAL cache
python -c "
import msal, os
cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
cache = msal.SerializableTokenCache()
with open(cache_path) as f:
    cache.deserialize(f.read())
app = msal.PublicClientApplication(
    '04b07795-8ddb-461a-bbee-02f9e1bf7b46',
    authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
    token_cache=cache
)
accounts = app.get_accounts()
print(f'Found {len(accounts)} account(s)')
for acc in accounts:
    print(f'  - {acc.get(\"username\", \"unknown\")}')
"
```

---

## Debugging Tips

### Finding Failed Tasks
```bash
# List all Grade=1 tasks from rubrics
grep ",1," rubrics_output/<benchmark>/*.csv | cut -d',' -f1 | sort -u
```

### Checking Fix Status
```bash
# View fix status for a task
cat fixes/<benchmark>/<task_id>/status.json
```

### Viewing Traces
```bash
# Extract conversation from trace
python -c "
import json
data = json.load(open('traces/file.json'))
for entry in data.get('raw_logging_results', [])[:5]:
    print(entry.get('inputs', {}).get('messages', [])[-1])
"
```

### Comparing Before/After
```bash
# Count defects before and after
echo "Before:"
grep ",1," rubrics_output/<benchmark>/*potato*.csv | wc -l
echo "After:"
grep ",1," rubrics_output/<benchmark>/*honey*.csv | wc -l
```

---

## Important Patterns

### Trace Files
- Format: `{benchmark}_{agent}_{model}_{timestamp}_UPLOAD.json`
- Contains: config, results, raw_logging_results, raw_eval_results

### Rubric CSVs
- Format: `{prefix}_{model}.csv`
- Columns: task_id, criteria, grade, correct, explanation, model_run

### Verdict CSVs
- Format: `{benchmark}_verdict.csv` or `{benchmark}_{prefix}_verdict.csv`
- Columns: task_id, final_grade, satisfies_rubric, num_evaluations, model_runs, reasoning

---

## Results Summary (SciCode - Reference)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Verdict Defect Rate | 44.6% | 0.0% | -44.6% |
| Rubric Defect Rate | 54.2% | 26.5% | -51.2% relative |
| Tasks Fixed | - | 9/9 | 100% |

See `demonstrate/README.md` for detailed analysis and figures.

---

## Quick Reference: Which Rubric for Which Benchmark

| Benchmark | Rubric File | Key Focus |
|-----------|-------------|-----------|
| SciCode | `scicode.txt` | Code execution, API compatibility, imports |
| CoreBench | `corebench.txt` | Docker/container issues, system dependencies |
| SWE-bench | `swebench.txt` | Repository state, test suite, patch format |
| USACO | `usaco.txt` | Algorithm correctness, time limits, I/O format |
| AssistantBench | `assistantbench.txt` | Website accessibility, data freshness |
| ScienceAgentBench | `scienceagentbench.txt` | Data files, evaluation scripts, scientific validity |
| ColBench | `colbench.txt` | Simulated user quality, hidden info accessibility, test fairness |
