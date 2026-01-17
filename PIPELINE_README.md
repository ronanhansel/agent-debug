# HAL Agent Debug Pipeline

This document describes the full pipeline for debugging and improving HAL agent performance across benchmarks.

## Overview

The pipeline consists of these stages:
1. **Traces** - Agent run results from HAL evaluation
2. **Rubric Evaluation** - Classify failures using benchmark-specific rubrics
3. **Inspection** - Analyze failures and generate fix recommendations
4. **Fix Application** - Apply fixes and re-run evaluation
5. **Weave Extraction** - Pull conversation logs from W&B
6. **Merge** - Combine individual traces into a single trace

## Quick Start

```bash
# Set up environment variables
export OPENAI_API_KEY="sk-1234"
export OPENAI_BASE_URL="http://localhost:4000/v1"

# Evaluate SciCode trace with SciCode rubric
python scripts/eval_rubric.py \
    --trace-file traces/scicode_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-4o \
    --failed-only \
    -y

# Evaluate CoreBench trace with CoreBench rubric
python scripts/eval_rubric.py \
    --trace-file traces/corebench_*.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-4o \
    --failed-only \
    -y
```

## Benchmark-Specific Rubrics

Each benchmark has its own rubric template in `rubric_templates/`:

| Benchmark | Rubric File | Focus |
|-----------|-------------|-------|
| SciCode | `scicode.txt` | Intrinsic Formation Errors (syntactic corruption, contextual discontinuity, mathematical ambiguity, environmental contradiction) |
| CoreBench | `corebench.txt` | Environmental Barriers vs Capability Issues |

### Output Structure

Rubric evaluation outputs go to `rubrics_output/<rubric_name>/`:
```
rubrics_output/
├── scicode/                    # SciCode rubric results
│   └── <trace_name>.csv
├── corebench/                  # CoreBench rubric results
│   └── <trace_name>.csv
└── environmentalbarrier/       # Legacy rubric results
    └── <trace_name>.csv
```

## Rubric Evaluation (Docent-based)

The primary rubric evaluation uses **Docent** for:
- SQLite LLM response caching (no repeat API calls)
- Batch processing with retry logic
- Proper message parsing from HAL traces
- Turn-by-turn conversation deduplication

### Single Rubric Evaluation

```bash
# Evaluate with a specific rubric
python scripts/eval_rubric.py \
    --trace-file traces/scicode_hal_generalist_agent_o4mini20250416_low_1745608137_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_claudeopus41_1755801688_UPLOAD.json \
    --trace-file traces/scicode_scicode-tool_calling_agent_claudeopus4120250514_1754678715_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_claude37sonnet20250219_high_1753770104_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_claudesonnet45_high_1759429729_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_deepseekaiDeepSeekV3_1745349011_UPLOAD.json \
    --trace-file traces/scicode_hal_generalist_agent_claude37sonnet20250219_high_1748947217_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    -y
```

`--parallel`

### Batch Evaluation

```bash
# Evaluate all rubrics in a directory
python scripts/eval_rubric.py \
    --trace-file traces/corebench_*.json \
    --rubrics-dir rubric_templates \
    --rubric-model openai:gpt-4o \
    --failed-only \
    -y
```

### Available Options

| Option | Description |
|--------|-------------|
| `--trace-file` | Path to trace JSON file |
| `--rubric` | Single rubric .txt file (overrides --rubrics-dir) |
| `--rubrics-dir` | Directory of rubric .txt files |
| `--rubric-model` | Model as provider:model (e.g., `openai:gpt-4o`) |
| `--output-mode` | `csv` (default) or `stdout` |
| `--max-tasks` | Limit number of tasks to evaluate |
| `--failed-only` | Only evaluate failed tasks |
| `-y` | Skip confirmation prompt |

## Cross-Model Rubric Evaluation

For comparing failures across models (especially useful for CoreBench):

```bash
python scripts/cross_model_rubric.py \
    --traces \
        traces/corebench_hard_hal_generalist_agentgpt41_*.json \
        traces/corebench_hard_hal_generalist_agento3medium_*.json \
        traces/corebench_hard_hal_generalist_agento4minihigh_*.json \
    --rubric rubric_templates/corebench.txt \
    --prefix iter1_ \
    --failed-only
```

## Trace Files

### SciCode Traces
```bash
traces/scicode_hal_generalist_agent_gpt4120250414_*.json     # GPT-4.1
traces/scicode_hal_generalist_agent_o4mini20250416_high_*.json  # O4-mini high
traces/scicode_hal_generalist_agent_o4mini20250416_low_*.json   # O4-mini low
traces/scicode_hal_generalist_agent_o320250416_*.json         # O3
```

### CoreBench Traces
```bash
traces/corebench_hard_hal_generalist_agentgpt41_*.json      # GPT-4.1
traces/corebench_hard_hal_generalist_agento3medium_*.json   # O3 medium
traces/corebench_hard_hal_generalist_agento4minihigh_*.json # O4-mini high
traces/corebench_hard_hal_generalist_agento4minilow_*.json  # O4-mini low
```

## Prerequisites

### Environment Setup

```bash
# Install docent (one-time setup)
cd src/docent-python
pip install -e docent/ -e .
cd ../..

# Set environment variables
export OPENAI_API_KEY="sk-1234"
export OPENAI_BASE_URL="http://localhost:4000/v1"  # Local proxy
```

### Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for rubric evaluation |
| `OPENAI_BASE_URL` | API endpoint (optional, for proxy) |
| `WANDB_API_KEY` | For Weave extraction |

## Creating Custom Rubrics

Rubrics are plain text files in `rubric_templates/`. Example structure:

```
rubric_templates/
├── scicode.txt      # SciCode-specific rubric
├── corebench.txt    # CoreBench-specific rubric
└── custom.txt       # Your custom rubric
```

### Rubric Format

```text
You are evaluating an agent's performance on [BENCHMARK].

## Evaluation Criteria

[Your criteria here]

## Output Format

Respond with valid JSON:
{
    "score": <float 0.0-1.0>,
    "explanation": "<reasoning>",
    "category": "<category_name>"
}
```

## Full Pipeline Example

### SciCode Pipeline

```bash
# 1. Evaluate failed tasks with SciCode rubric
python scripts/eval_rubric.py \
    --trace-file traces/scicode_hal_generalist_agent_gpt4120250414_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-4o \
    --failed-only \
    -y

# 2. View results
cat rubrics_output/scicode/*.csv

# 3. Analyze specific error categories
grep "syntactic_corruption" rubrics_output/scicode/*.csv
grep "mathematical_ambiguity" rubrics_output/scicode/*.csv
```

### CoreBench Pipeline

```bash
# 1. Run cross-model rubric evaluation
python scripts/cross_model_rubric.py \
    --traces traces/corebench_hard_hal_generalist_agent*.json \
    --rubric rubric_templates/corebench.txt \
    --prefix iter1_ \
    --failed-only

# 2. Find environmental barriers (score=1)
RUBRIC_CSV=$(ls -t rubrics_output/corebench/iter1_*.csv | head -1)
grep ",1.0," $RUBRIC_CSV

# 3. Generate fixes for environmental barriers
python scripts/pipeline.py inspect \
    --trace-files traces/corebench_hard_hal_generalist_agent*.json \
    --rubric-csv $RUBRIC_CSV \
    --benchmark corebench_hard \
    --env-barriers-only

# 4. Apply fixes and re-run
python scripts/pipeline.py fix --prefix iter1_ --docker
```

## Output CSV Format

The rubric evaluation produces CSVs with these columns:

| Column | Description |
|--------|-------------|
| `task_id` | Task identifier (e.g., `capsule-1234567`) |
| `criteria` | Rubric name used |
| `grade` | Score (0.0-1.0) |
| `correct` | Whether task passed originally |
| `explanation` | LLM's reasoning |
| `model_run` | Model used for evaluation |

### Example CSV Row

```csv
task_id,criteria,grade,correct,explanation,model_run
11,scicode,0.0,False,"The agent encountered a Unicode minus sign (U+2212) that caused SyntaxError...",gpt-4o
```

## Troubleshooting

### "Unable to import Docent modules"

```bash
# Install docent properly
cd src/docent-python
pip install -e docent/ -e .
```

### "Connection error" with OpenAI

```bash
# Check if proxy is running
curl http://localhost:4000/v1/models

# Or use direct OpenAI (if accessible)
unset OPENAI_BASE_URL
```

### "No raw_logging_results in trace"

Run Weave extraction first:
```bash
python scripts/pipeline.py extract --prefix <prefix> --project hal-agent-debug
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/eval_rubric.py` | **Primary** Docent-based rubric evaluation |
| `scripts/cross_model_rubric.py` | Cross-model comparison rubric evaluation |
| `scripts/pipeline.py` | Unified CLI for all pipeline operations |
| `rubric_templates/scicode.txt` | SciCode Intrinsic Formation Error rubric |
| `rubric_templates/corebench.txt` | CoreBench Environmental Barrier rubric |
| `rubric_evaluator/cli.py` | Core rubric evaluation logic with Docent |

## Models

Recommended models for rubric evaluation:
- `openai:gpt-4o` - Best balance of speed/accuracy (recommended)
- `openai:gpt-4o-mini` - Fast, good for iteration
- `azure_openai:gpt-4o` - If using Azure

```bash
# Example with different models
python scripts/eval_rubric.py \
    --trace-file traces/scicode_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-4o \
    -y
```
