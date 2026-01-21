# Complete Merge and Fetch Summary

## Overview

This document summarizes all the traces that need to be merged and fetched from Weave for the complete evaluation pipeline.

## Projects and Models

### CoreBench

**Projects:**
- `prop_corebench_hard`
- `iter1_corebench_hard`

**Models (10 total):**
1. o4-mini (high reasoning effort)
2. o4-mini (low reasoning effort)
3. gpt-4.1
4. o3 (medium reasoning effort)
5. o3 (low reasoning effort)
6. gpt-oss-120b
7. gpt-5 (medium reasoning effort)
8. deepseek-r1
9. deepseek-v3
10. gpt-4o

### SciCode

**Projects:**
- `scicode_lady_scicode`
- `scicode_honey_scicode`

**Models (10 total):**
1. o3 (medium reasoning effort)
2. o4-mini (low reasoning effort)
3. o4-mini (high reasoning effort)
4. gpt-4.1
5. gpt-5
6. deepseek-r1
7. deepseek-v3
8. gpt-5-mini
9. gpt-4o
10. o3-mini (high reasoning effort)

### ScienceAgentBench

**Projects:**
- `sab_mate_scienceagentbench` (different models)
- `sab_cow_scienceagentbench` (same models as husky)
- `sab_husky_scienceagentbench` (same models as cow)

**Models (6 total):**
1. gpt-5 (medium reasoning effort)
2. deepseek-r1
3. deepseek-v3
4. gpt-5-mini
5. gpt-4o
6. o3-mini (high reasoning effort)

### ColBench

**Projects:**
- `col_ivy_colbench_backendprogramming`
- `col_zuck_colbench_backendprogramming`

**Models (9 total):**
1. gpt-4.1
2. o4-mini (low reasoning effort)
3. o3 (medium reasoning effort)
4. o4-mini (high reasoning effort)
5. gpt-5 (medium reasoning effort)
6. gpt-4o
7. gpt-5-mini
8. o3-mini (high reasoning effort)
9. deepseek-r1

## Running the Complete Pipeline

### Step 1: Merge and Fetch All Traces

Run the automated script:

```bash
./merge_and_fetch_all.sh
```

This script will:
1. Merge all local traces for each model/prefix combination
2. Extract traces from Weave for each project
3. Create merged `*_MERGED_UPLOAD.json` files in the `traces/` directory

### Step 2: Run Rubric Evaluation

After all traces are merged and fetched, run the rubric evaluation for each benchmark:

#### CoreBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/prop_corebench_hard_*_MERGED_UPLOAD.json \
                 traces/iter1_corebench_hard_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

#### SciCode
```bash
python scripts/eval_rubric.py \
    --trace-file traces/scicode_lady_*_MERGED_UPLOAD.json \
                 traces/scicode_honey_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

#### ScienceAgentBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_*_MERGED_UPLOAD.json \
                 traces/sab_cow_*_MERGED_UPLOAD.json \
                 traces/sab_husky_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

#### ColBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/col_ivy_*_MERGED_UPLOAD.json \
                 traces/col_zuck_*_MERGED_UPLOAD.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

## Expected Output Files

After running the complete pipeline, you should have:

### CoreBench (20 merged traces)
- `prop_corebench_hard_*_MERGED_UPLOAD.json` (10 files, one per model)
- `iter1_corebench_hard_*_MERGED_UPLOAD.json` (10 files, one per model)

### SciCode (20 merged traces)
- `scicode_lady_*_MERGED_UPLOAD.json` (10 files, one per model)
- `scicode_honey_*_MERGED_UPLOAD.json` (10 files, one per model)

### ScienceAgentBench (18 merged traces)
- `sab_mate_*_MERGED_UPLOAD.json` (6 files, one per model)
- `sab_cow_*_MERGED_UPLOAD.json` (6 files, one per model)
- `sab_husky_*_MERGED_UPLOAD.json` (6 files, one per model)

### ColBench (18 merged traces)
- `col_ivy_*_MERGED_UPLOAD.json` (9 files, one per model)
- `col_zuck_*_MERGED_UPLOAD.json` (9 files, one per model)

**Total: 76 merged trace files**

## Rubric Output

Rubrics will be saved to:
- `rubrics_output/corebench/`
- `rubrics_output/scicode/`
- `rubrics_output/scienceagentbench/`
- `rubrics_output/colbench/`

## Troubleshooting

### Missing Traces
If some traces are missing from Weave, check:
1. The project name matches exactly (case-sensitive)
2. The prefix patterns are correct
3. The runs were successfully uploaded to Weave

### Merge Failures
If merge fails with "no matching files", it means:
1. Local traces haven't been copied to `traces/` directory yet
2. The naming pattern doesn't match (check actual filenames)
3. The runs haven't completed yet

### Alternative: Run Individual Commands

If the full script fails, you can run commands for individual benchmarks by extracting the relevant sections from `merge_and_fetch_all.sh`.

## Model Name Mappings

For reference, here's how model names map between config and trace filenames:

| Config Key | Model ID | Trace Pattern |
|-----------|----------|---------------|
| `openai/gpt-4.1-2025-04-14` | `openai/gpt-4.1_2025-04-14` | `*_openai_gpt-4_1_2025-04-14_*` |
| `openai/o3-2025-04-16` | `openai/o3_2025-04-16` | `*_openai_o3_2025-04-16_medium_*` |
| `o4-mini-2025-04-16-high` | `openai/o4-mini_2025-04-16` | `*_openai_o4-mini_2025-04-16_high_*` |
| `o4-mini-2025-04-16-low` | `openai/o4-mini_2025-04-16` | `*_openai_o4-mini_2025-04-16_low_*` |
| `deepseek-r1` | `DeepSeek-R1_1` | `*_DeepSeek-R1_1_*` |
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3-0324` | `*_deepseek-ai_DeepSeek-V3-0324_*` |

Note: The `/` in model IDs gets replaced with `_` in trace filenames.
