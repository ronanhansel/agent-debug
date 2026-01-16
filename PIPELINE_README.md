# HAL Agent Debug Pipeline

This document describes the full pipeline for debugging and improving HAL agent performance on CoreBench.

## Overview

The pipeline consists of these stages:
1. **Traces** - Agent run results from HAL evaluation
2. **Rubric Evaluation** - Classify failures as environmental barriers vs capability issues
3. **Inspection** - Analyze failures and generate fix recommendations
4. **Fix Application** - Apply fixes and re-run evaluation
5. **Weave Extraction** - Pull conversation logs from W&B
6. **Merge** - Combine individual traces into a single trace

## Quick Start

```bash
# 1. Check current status
python scripts/pipeline.py status --prefix orange

# 2. Run rubric evaluation on a trace with conversation logs
python scripts/pipeline.py rubric --prefix orange --trace-file traces/baseline.json --failed-only

# 3. Extract Weave traces (if trace lacks raw_logging_results)
python scripts/pipeline.py extract --prefix orange --project hal-agent-debug

# 4. Merge individual traces
python scripts/pipeline.py merge --prefix orange
```

## Prerequisites

### Environment Setup

```bash
# Activate conda base environment (contains docent, openai dependencies)
conda activate base

# Ensure OpenAI proxy is running (for rubric evaluation)
# Default: http://localhost:4000/v1
```

### Required Environment Variables

```bash
export WANDB_API_KEY="your-wandb-api-key"  # For Weave extraction
export OPENAI_BASE_URL="http://localhost:4000/v1"  # Optional, defaults to this
```

## Pipeline Stages

### Stage 1: Initial Traces

Traces come from HAL evaluation runs. Original baseline traces are in `traces/`:
- `corebench_hard_hal_generalist_agentgpt41_*.json` - GPT-4.1 baseline
- `corebench_hard_hal_generalist_agento3medium_*.json` - O3 baseline
- etc.

**Important**: Traces from HAL evaluation may NOT contain `raw_logging_results` (conversation logs). These need to be extracted from Weave first.

### Stage 2: Rubric Evaluation

Classify failures into:
- **Environmental Barriers (score=1)**: Infrastructure issues no agent can overcome
  - Missing R runtime with blocked apt installation
  - Missing benchmark data files
  - Permission denied on required paths
  - Conda ToS blocking in non-interactive mode

- **Capability Issues (score=0)**: Agent could have succeeded with better approach
  - Package installed to wrong Python environment
  - Config file misconfiguration
  - Tool misuse when alternatives existed
  - Path confusion

```bash
# Evaluate failed tasks only
python scripts/pipeline.py rubric \
  --prefix orange \
  --trace-file traces/corebench_hard_hal_generalist_agentgpt41_1755644685_UPLOAD.json \
  --failed-only \
  --model gpt-5.2

# Output: rubrics_output/environmental_barrier/orange_<trace_name>.csv
```

**Note**: Rubric evaluation REQUIRES `raw_logging_results` in the trace. If missing, run `extract` first.

### Stage 3: Weave Extraction

Pull conversation logs from W&B Weave for traces that lack `raw_logging_results`.

```bash
python scripts/pipeline.py extract \
  --prefix orange \
  --project hal-agent-debug

# This creates: traces/<prefix>*.json with raw_logging_results filled in
```

### Stage 4: Inspection (Fix Generation)

Analyze failures and generate fix recommendations:

```bash
python scripts/pipeline.py inspect \
  --trace-file traces/orange_MERGED_UPLOAD.json \
  --benchmark corebench_hard \
  --dry-run  # Preview only

# Without --dry-run, creates fix packages in:
# fixes/corebench_hard/<task_id>/env_override.json
```

Fix packages contain:
- `env_override.json` - Environment variables and conda packages
- `input_override.json` - Clarified task instructions (future)

### Stage 5: Fix Application

Apply fixes and re-run evaluation:

```bash
python scripts/pipeline.py fix \
  --prefix orange \
  --docker \
  --task-id capsule-2345790 \
  --task-id capsule-1394704

# Or run all tasks with fixes:
python scripts/pipeline.py fix --prefix orange --docker
```

### Stage 6: Merge Traces

Combine individual task traces into a single merged trace:

```bash
python scripts/pipeline.py merge --prefix orange

# Creates: traces/orange_MERGED_<timestamp>_UPLOAD.json
```

## Iteration Tracking with Prefixes

Use `--prefix` to track different iterations:

```bash
# First iteration
python scripts/pipeline.py full --prefix v1_ --trace-file traces/baseline.json

# After applying fixes, second iteration
python scripts/pipeline.py full --prefix v2_ --trace-file traces/v1_MERGED_UPLOAD.json

# Compare results
python scripts/pipeline.py status --prefix v1_
python scripts/pipeline.py status --prefix v2_
```

Example prefixes:
- `orange_` - First experiment batch
- `mango_` - Second experiment batch
- `v1_`, `v2_`, `v3_` - Sequential iterations
- `gpt4_`, `o3_`, `gpt5_` - Model-specific runs

## Full Pipeline Example

```bash
# 1. Start with baseline trace (already has raw_logging_results)
BASELINE="traces/corebench_hard_hal_generalist_agentgpt41_1755644685_UPLOAD.json"

# 2. Run rubric evaluation to identify failures
python scripts/pipeline.py rubric \
  --prefix iter1_ \
  --trace-file $BASELINE \
  --failed-only

# 3. Review rubric results
cat rubrics_output/environmental_barrier/iter1_*.csv

# 4. Generate fixes for capability issues (score=0)
python scripts/pipeline.py inspect \
  --trace-file $BASELINE \
  --benchmark corebench_hard

# 5. Apply fixes and re-run (creates new traces with prefix)
python scripts/pipeline.py fix \
  --prefix iter1_ \
  --docker

# 6. Extract conversation logs from Weave
python scripts/pipeline.py extract \
  --prefix iter1_ \
  --project hal-agent-debug

# 7. Merge new traces
python scripts/pipeline.py merge --prefix iter1_

# 8. Re-evaluate with rubrics
python scripts/pipeline.py rubric \
  --prefix iter1_round2_ \
  --trace-file traces/iter1_MERGED_*_UPLOAD.json \
  --failed-only

# 9. Compare iterations
python scripts/pipeline.py status --prefix iter1_
```

## Individual Scripts

For more control, use individual scripts:

### Rubric Evaluation (Lightweight)
```bash
python scripts/simple_rubric_eval.py \
  --trace-file traces/baseline.json \
  --rubrics-dir rubrics \
  --output-dir rubrics_output \
  --model gpt-5.2 \
  --failed-only
```

### Trace Merging
```bash
python scripts/merge_traces.py \
  --input 'traces/orange_*_UPLOAD.json' \
  --output traces/orange_MERGED.json \
  --force
```

### Weave Extraction
```bash
python scripts/extract_weave_traces.py \
  --project hal-agent-debug \
  --prefix orange \
  --include-costs
```

### Fix Application (Advanced)
```bash
python scripts/run_corebench_fixes.py \
  --fixes-root fixes/corebench_hard \
  --agent-dir hal-harness/agents/hal_generalist_agent \
  --agent-args agent_args.azure.json \
  --benchmark corebench_hard \
  --docker \
  --prefix orange \
  --task-id capsule-2345790
```

## Output Directories

```
agent-debug/
├── traces/                          # All trace files
│   ├── *_UPLOAD.json               # Individual task traces
│   └── *_MERGED_*_UPLOAD.json      # Merged traces
├── rubrics_output/                  # Rubric evaluation results
│   └── environmental_barrier/       # By rubric type
│       └── <prefix>_<trace>.csv    # Results CSV
├── fixes/                           # Fix packages
│   └── corebench_hard/
│       └── capsule-*/
│           └── env_override.json
└── results/                         # Re-run results
```

## Troubleshooting

### "No raw_logging_results in trace"
Run Weave extraction first:
```bash
python scripts/pipeline.py extract --prefix <prefix> --project hal-agent-debug
```

### "Trace file not found"
Check the traces directory:
```bash
ls -la traces/<prefix>*
```

### "OpenAI API error"
Ensure proxy is running:
```bash
curl http://localhost:4000/v1/models
```

### Empty merged trace
Individual traces likely lack `raw_logging_results`. Extract from Weave first.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/pipeline.py` | Unified CLI for all pipeline operations |
| `scripts/simple_rubric_eval.py` | Lightweight rubric evaluation |
| `scripts/merge_traces.py` | Merge individual traces |
| `scripts/extract_weave_traces.py` | Extract from W&B Weave |
| `scripts/item_fixer.py` | Analyze failures and generate fixes |
| `scripts/run_corebench_fixes.py` | Apply fixes and re-run |
| `rubrics/environmental_barrier.txt` | Rubric definition |

## Cross-Model Rubric Evaluation (Recommended)

For the most accurate environmental barrier detection, use cross-model evaluation:

```bash
python scripts/cross_model_rubric.py \
  --traces \
    traces/baseline_gpt41.json \
    traces/baseline_o3.json \
    traces/baseline_o4mini_high.json \
    traces/baseline_o4mini_low.json \
  --prefix iter1_ \
  --failed-only
```

### How It Works

**Key insight**: If ANY model succeeded at a task, that task has NO environmental barrier.

1. **Phase 1: Build Task Success Map**
   - Scan all model traces
   - Identify which tasks each model passed/failed
   - Extract error patterns from each model

2. **Phase 2: Context-Aware Evaluation**
   - **Quick decisions**: If any model succeeded → score=0 (capability issue)
   - **Deep analysis**: If ALL models failed → detailed rubric evaluation with cross-model context

### Benefits

- **More accurate**: Uses evidence from multiple models
- **Faster**: Quick decisions for tasks where any model succeeded
- **Better reasoning**: Rubric sees "Model A succeeded, so this is NOT an env barrier"

### Example Output

```
Task summary: 45 total tasks
  - 22 tasks solved by at least one model (NOT env barriers)
  - 23 tasks failed by ALL models (potential env barriers)

Quick decisions (other model succeeded): 22
Full evaluations needed: 23
```

### Summary Mode

Preview the cross-model summary without running evaluation:

```bash
python scripts/cross_model_rubric.py \
  --traces traces/*.json \
  --summary-only
```

## Models

Recommended models for rubric evaluation:
- `gpt-5.2` - Best accuracy (default)
- `gpt-4o` - Good balance of speed/accuracy
- `gpt-4o-mini` - Fast, lower accuracy
