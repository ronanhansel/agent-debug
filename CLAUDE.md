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
│   └── scienceagentbench.txt    # ScienceAgentBench benchmark rubric
│
├── rubrics_output/              # Rubric evaluation results (CSVs)
│   ├── scicode/
│   ├── corebench/
│   ├── swebench/
│   ├── usaco/
│   ├── assistantbench/
│   └── scienceagentbench/
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

**Agent**: `hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench/agent.py`
- 102 tasks across 4 scientific domains (bioinformatics, chemistry, GIS, psychology)
- Uses ScienceAgent class with LLM engines
- Optional self-debug mechanism
- Docker-based evaluation

**Common Issues**:
- Dataset files missing or corrupted
- Evaluation scripts have bugs
- Required libraries not in container
- Output format requirements unclear

**Known Research-Documented Issues** (from benchmark paper and trace analysis):
- **High Universal Failure Rate**: 67/102 tasks failed across ALL 4 models (GPT-4.1, O3, O4-mini)
- **Figure Evaluation Subjectivity**: GPT-4 judge penalizes color schemes, axis labels, layout differences
- **Domain Library Gaps**: `oggm`, `mastml`, `mne`, `biopsykit` often missing from environment
- **Acknowledged Evaluation Noise**: Authors note "subjective variance in color, scale, and labeling"
- **Partial Credit Issues**: "Functionally equivalent implementations may receive lower scores"

**Rubric Features** (`rubric_templates/scienceagentbench.txt`):
- Cross-run analysis with specific failure statistics
- Figure/visualization evaluation issues category
- Domain-specific library availability investigation
- Known problematic task patterns (Tasks 74, 43, 2 for environment issues)
- Exploratory questions for subjective evaluation fairness

**Metrics**: Valid Execution Rate (VER), Success Rate (SR), CodeBERTScore (CBS)

**HAL Command**:
```bash
hal-eval --benchmark scienceagentbench \
    --agent_dir hal/benchmarks/scienceagentbench/ScienceAgentBench/ \
    --agent_function agent.run \
    --agent_name "Science Agent" \
    -A model_name=gpt-4o
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
    --agent_dir hal/benchmarks/scienceagentbench/ScienceAgentBench/ \
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
2. **Create rubric template**: `rubric_templates/<benchmark>.txt`
3. **Create output directory**: `mkdir -p rubrics_output/<benchmark>`
4. **Update CLAUDE.md**: Add benchmark details
5. **Run rubric evaluation**:
   ```bash
   python scripts/eval_rubric.py \
       --trace-file traces/<benchmark>_*.json \
       --rubric rubric_templates/<benchmark>.txt \
       --rubric-model openai:gpt-4o \
       --failed-only -y
   ```
6. **Aggregate verdicts**:
   ```bash
   python scripts/judge.py \
       --rubric-dir rubrics_output/<benchmark> \
       --output judge_output/<benchmark>_verdict.csv
   ```
7. **Investigate Grade=1 tasks**: Identify root causes
8. **Apply item fixes**: Agent config, prompts, rubric clarifications
9. **Re-run evaluation**: Use new prefix (e.g., `honey`)
10. **Compare before/after**: Measure improvement

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
