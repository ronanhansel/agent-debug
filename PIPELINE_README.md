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

| Benchmark | Rubric File     | Focus                                                                                                                            |
| --------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| SciCode   | `scicode.txt`   | Intrinsic Formation Errors (syntactic corruption, contextual discontinuity, mathematical ambiguity, environmental contradiction) |
| CoreBench | `corebench.txt` | Environmental Barriers vs Capability Issues                                                                                      |

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
    --trace-file traces/scicode_scicode_tool_calling_agent_deepseekaiDeepSeekV3_1745349011_UPLOAD.json \
    --trace-file traces/scicode_hal_generalist_agent_claude37sonnet20250219_high_1748947217_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --inbetween "TMUX= ./deploy_llm.sh" \
    --sleep 5s \
    --max-batch-messages 400 \
    --inter-batch-delay 0 \
    --retries 10 \
    --sort-by-messages \
    -y
```

`--parallel` vs `--max-batch-messages`: parallel are of equal size, batch fills up to max messages.

To give out final evaluation

```bash
python scripts/judge.py \
      --pattern "scicode_*" \
      --rubric-dir rubrics_output/scicode \
      --model openai:gpt-5.2 \
      -y
```

`--max-tasks`

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

| Option           | Description                                       |
| ---------------- | ------------------------------------------------- |
| `--trace-file`   | Path to trace JSON file                           |
| `--rubric`       | Single rubric .txt file (overrides --rubrics-dir) |
| `--rubrics-dir`  | Directory of rubric .txt files                    |
| `--rubric-model` | Model as provider:model (e.g., `openai:gpt-4o`)   |
| `--output-mode`  | `csv` (default) or `stdout`                       |
| `--max-tasks`    | Limit number of tasks to evaluate                 |
| `--failed-only`  | Only evaluate failed tasks                        |
| `-y`             | Skip confirmation prompt                          |

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

| Variable          | Purpose                            |
| ----------------- | ---------------------------------- |
| `OPENAI_API_KEY`  | API key for rubric evaluation      |
| `OPENAI_BASE_URL` | API endpoint (optional, for proxy) |
| `WANDB_API_KEY`   | For Weave extraction               |

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
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --max-batch-messages 400 \
    --inter-batch-delay 0 \
    --sleep 0s \
    --retries 10 \
    --sort-by-messages \
    -y


python scripts/eval_rubric.py \
    --trace-file traces/scicode_hal_generalist_agent_deepseekaideepseekr1_1753777608_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o4mini20250416_low_1745290900_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o4mini20250416_high_1745429794_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o4mini20250416_1745274271_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o320250416_1745284451_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_gpt4120250414_1745265933_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_gemini20flash_1745437512_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_deepseekaideepseekv3_1745456160_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_deepseekaideepseekr1_1753815994_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_claude37sonnet20250219_high_1748945506_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_claude37sonnet20250219_1745345545_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_o4mini20250416_low_1745286980_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_o4mini20250416_high_1745414081_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_o320250416_1745276105_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_deepseekaideepseekr1_1753819405_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_claudehaiku45_1760968740_UPLOAD.json \
    --trace-file traces/scicode_hal_generalist_agent_gemini20flash_1745628676_UPLOAD.json \
    --trace-file traces/scicode_hal_generalist_agent_deepseekaideepseekv3_1745642054_UPLOAD.json \
    --trace-file traces/scicode_scicode_tool_calling_agent_gpt5_1754600998_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --inbetween "TMUX= ./deploy_llm.sh" \
    --sleep 15s \
    --max-batch-messages 400 \
    --inter-batch-delay 0 \
    --retries 10 \
    --sort-by-messages \
    --sort-by-file-size \
    -y

python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --rubric-dir rubrics_output/scicode \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y


python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --benchmark scicode \
    --tasks-per-batch 5 \
    --parallel 4 \
    --ife-only \
    --skip-existing

python scripts/run_scicode_fixes.py \
      --task-id 11 \
      --output-prefix fixed_scicode \
      --agent-dir hal-harness/agents/scicode_tool_calling_agent \
      --agent-args agent_args.json
      --docker
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

| Column        | Description                               |
| ------------- | ----------------------------------------- |
| `task_id`     | Task identifier (e.g., `capsule-1234567`) |
| `criteria`    | Rubric name used                          |
| `grade`       | Score (0.0-1.0)                           |
| `correct`     | Whether task passed originally            |
| `explanation` | LLM's reasoning                           |
| `model_run`   | Model used for evaluation                 |

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

### Subsequent runs Pipeline

I want to start with implementing the full pipeline for the diagnosis and fixing of USACO. How can I get started with it? Here's the code to kickstart scicode and its full pipeline, the fixes are in the fixes/ folder. Now I want to implement the exact flow, pipeline to USACO, make sure to create separate run_usaco_fixes.py that can mirror what this does.

```bash
# initial run, when there's no rubrics yet and no runs, you have to upload traces first
python scripts/eval_rubric.py \
    --trace-file traces/scicode_hal_generalist_agent_deepseekaideepseekr1_1753777608_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o4mini20250416_low_1745290900_UPLOAD.json \
    --trace-file traces/scicode_scicode_zero_shot_agent_o4mini20250416_high_1745429794_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --inbetween "TMUX= ./deploy_llm.sh" \
    --sleep 15s \
    --max-batch-messages 400 \
    --inter-batch-delay 0 \
    --retries 10 \
    --sort-by-messages \
    --sort-by-file-size \
    -y

# After initial run, you can merge the resulting traces with this and start from there

python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force

# Weave Extraction

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json

# Evaluation of rubrics

python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --max-batch-messages 400 \
    --inter-batch-delay 0 \
    --sleep 0s \
    --retries 10 \
    --sort-by-messages \
    -y

python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --rubric-dir rubrics_output/scicode \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

python scripts/run_scicode_fixes.py\
    --prefix scicode_honey \
    --parallel 20 \
    --docker
```

Full USACO Pipeline

Step 2: Rubric Evaluation

# Evaluate failed tasks against rubric

```bash
python scripts/eval_rubric.py \
    $(printf -- '--trace-file %s ' traces/scienceagentbench_*) \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --max-batch-messages 1000 \
    --max-concurrency 500 \
    --rate-limit-delay 65 \
    --retries 10 \
    --inbetween "TMUX= ./deploy_llm.sh" \
    --sleep 10s \
    --inter-batch-delay 5 \
    --sort-by-messages \
    -y
```

Step 3: Judge Aggregation

# Aggregate rubric verdicts

```bash
python scripts/judge.py \
    --pattern scienceagentbench_* \
    --rubric-dir rubrics_output/scienceagentbench \
    --model openai:gpt-5.2 \
    --parallel 100 \
    -y
```

Step 4: Create Fixes

```bash
python scripts/claude_fixer_scienceagentbench.py \
    --rubric-dir rubrics_output/scienceagentbench \
    --judge-csv judge_output/scienceagentbench_verdict.csv \
    $(printf -- '--trace-file %s ' traces/scienceagentbench_*) \
    --ife-only \
    --tasks-per-batch 5 \
    --parallel 5
```

Step 5: Verify and Apply Fixes

```bash
python scripts/run_scienceagentbench_fixes.py \
    --prefix sab_cow \
    --parallel 80 \
    --docker
```

## After initial run, you can merge the resulting traces with this and start from there

```bash
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_husky_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/sab_husky_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/sab_husky_openai_o4-mini_low_MERGED_UPLOAD.json --force
```

## Weave Extraction

```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-4_1 \
    --prefix sab_husky_openai_o3_2025 \
    --prefix sab_husky_openai_o4-mini_2025-04-16_high \
    --prefix sab_husky_openai_o4-mini_2025-04-16_low \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o4-mini_low_MERGED_UPLOAD.json
```

```bash
python scripts/eval_rubric.py \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --trace-file traces/sab_husky_openai_o3_2025.json \
    --trace-file traces/sab_husky_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/sab_husky_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --max-batch-messages 1000 \
    --sleep 0s \
    --retries 10 \
    --sort-by-messages \
    -y
```

```bash
python scripts/judge.py \
    --pattern sab_husky_* \
    --rubric-dir rubrics_output/scienceagentbench \
    --model openai:gpt-5.2 \
    --parallel 100 \
    -y
```

```bash
python scripts/run_scienceagentbench_fixes.py \
      --task-id 52 --task-id 95 --task-id 64 --task-id 12 \
      --task-id 102 --task-id 74 --task-id 97 \
      --prefix sab_husky --docker --parallel 80
```

## Key Files

| File                             | Purpose                                    |
| -------------------------------- | ------------------------------------------ |
| `scripts/eval_rubric.py`         | **Primary** Docent-based rubric evaluation |
| `scripts/cross_model_rubric.py`  | Cross-model comparison rubric evaluation   |
| `scripts/pipeline.py`            | Unified CLI for all pipeline operations    |
| `rubric_templates/scicode.txt`   | SciCode Intrinsic Formation Error rubric   |
| `rubric_templates/corebench.txt` | CoreBench Environmental Barrier rubric     |
| `rubric_evaluator/cli.py`        | Core rubric evaluation logic with Docent   |

```bash
# Example with different models
python scripts/eval_rubric.py \
    --trace-file traces/scicode_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-4o \
    -y
```

```bash
# 1. Verify fixes work correctly first
python scripts/run_scicode_fixes.py --verify-fixes

# 2. List available fixes
python scripts/run_scicode_fixes.py --list-fixes

# 3. Dry run to see what would happen
python scripts/run_scicode_fixes.py --dry-run

# 4. Run fixes with a specific model
python scripts/run_scicode_fixes.py --model openai/gpt-4.1-2025-04-14 --docker


# 5. Run fixes using model that originally failed (from rubric CSV)
python scripts/run_scicode_fixes.py \
    --rubric-csv rubrics_output/scicode/scicode_scicode_tool_calling_agent_gpt4120250414_1745260672_UPLOAD.csv \
    --docker

# 6. Run specific tasks only
python scripts/run_scicode_fixes.py \
    --model openai/gpt-4.1-2025-04-14 \
    --task-id 71 --task-id 28 \
    --docker

# Pipeline running
python scripts/run_scicode_fixes.py\
    --prefix scicode_honey \
    --parallel 20 \
    --docker
```

If you're forgetting to download the `test_data.h5` for SciCode benchmark, download it and place it in `./hal-harness/hal/benchmarks/SciCode/eval/data`:

```bash
# Dry run first (preview what will be done)
python scripts/reevaluate_scicode.py --prefix scicode_honey --dry-run

# Run the actual re-evaluation
python scripts/reevaluate_scicode.py --prefix scicode_honey

# Re-evaluate specific tasks only
python scripts/reevaluate_scicode.py --prefix scicode_honey --task-id 12 --task-id 35
```

Delete ALL cached docker images

```bash
docker images --format "{{.Repository}}:{{.Tag}}" | grep "hal-agent-runner:agent-env" | xargs -r docker rmi -f

# 2. Also delete by the specific image IDs we saw
docker rmi -f 39e3e1f3edba 9e00df765a20 2>/dev/null || true

# 3. Clear Docker build cache
docker builder prune -f

# 4. Verify the images are gone
docker images | grep hal-agent-runner
```

COLBENCH

```bash
python scripts/eval_rubric.py \
    $(printf -- '--trace-file %s ' traces/colbench_*_binary_UPLOAD.json) \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --max-batch-messages 15000 \
    --retries 10 \
    --sort-by-messages \
    --failed-only -y
```

```bash
python scripts/judge.py \
    --pattern colbench_backend* \
    --rubric-dir rubrics_output/colbench \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y
```

Step 4: Create Fixes

```bash
python scripts/claude_fixer_colbench.py \
    --rubric-dir rubrics_output/colbench \
    --judge-csv judge_output/colbench_backend_verdict.csv \
    $(printf -- '--trace-file %s ' traces/colbench_*) \
    --ife-only \
    --tasks-per-batch 10 \
    --parallel 5
```

Step 5: Verify and Apply Fixes

```bash
python scripts/run_colbench_fixes.py \
    --prefix col_zuck\
    --benchmark colbench_backend_programming \
    --parallel 80 \
    --docker
```

# Stop and remove all containers

docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# Clean up networks

docker network prune -f

# If still not working, restart Docker

sudo systemctl restart docker

```bash
python scripts/merge_traces.py --input 'traces/col_tommy__col_tommy_gpt-4_1-2025-04-14_*_UPLOAD.json' --output traces/col_tommy_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_tommy__col_tommy_o3-2025-04-16_low_*_UPLOAD.json' --output traces/col_tommy_openai_o3_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_tommy__col_tommy_o4-mini-2025-04-16_high_*_UPLOAD.json' --output traces/col_tommy_openai_o4-mini_high_MERGED_UPLOAD.json --force
```

```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_tommy_colbench_backend_programming \
    --prefix col_tommy_openai_gpt-4_1 \
    --prefix col_tommy_openai_o3 \
    --prefix col_tommy_openai_o4-mini \
    --merge-input traces/col_tommy_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_tommy_openai_o3_low_MERGED_UPLOAD.json \
    --merge-input traces/col_tommy_openai_o4-mini_high_MERGED_UPLOAD.json
```

```bash
python scripts/eval_rubric.py \
    --trace-file traces/col_tommy_openai_gpt-4_1_WITH_DIALOGUES.json \
    --trace-file traces/col_tommy_openai_o3_low_WITH_DIALOGUES.json \
    --trace-file traces/col_tommy_openai_o4-mini_high_WITH_DIALOGUES.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
```

```bash
python scripts/judge.py \
    --pattern col_cindy* \
    --rubric-dir rubrics_output/colbench \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y
```



# CoreBench

```bash
python scripts/eval_rubric.py \
    --trace-file traces/prop_corebench_hard_*_MERGED_UPLOAD.json \
                traces/iter1_corebench_hard_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y --max-batch-messages 1000
```

# SciCode
python scripts/eval_rubric.py \
    --trace-file traces/scicode_lady_*_MERGED_UPLOAD.json \
                traces/scicode_honey_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y --max-batch-messages 1000

# ScienceAgentBench
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_*_MERGED_UPLOAD.json \
                traces/sab_cow_*_MERGED_UPLOAD.json \
                traces/sab_husky_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y --max-batch-messages 1000

# ColBench
python scripts/eval_rubric.py \
    --trace-file traces/col_ivy_*_MERGED_UPLOAD.json \
                traces/col_zuck_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y --max-batch-messages 1000