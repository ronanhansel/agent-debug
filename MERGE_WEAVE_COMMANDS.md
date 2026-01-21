# Merge and Weave Extraction Commands

## Prerequisites

Set your WANDB_API_KEY in environment:
```bash
export WANDB_API_KEY=your_key_here
```

---

## SCICODE

### SciCode Honey

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
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
```

### SciCode Lady

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
    --prefix scicode_lady_openai_gpt-4_1 \
    --prefix scicode_lady_openai_o3_2025 \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json
```

---

## COREBENCH

### CoreBench Prop

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/prop_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025 \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json
```

### CoreBench Iter1

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard \
    --prefix iter1_openai_gpt-4_1 \
    --prefix iter1_openai_o3_2025 \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json
```

---

## SCIENCEAGENTBENCH

### SAB Mate

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json
```

### SAB Cow

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json
```

### SAB Husky

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-4_1 \
    --prefix sab_husky_openai_o3-mini_2025 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json
```

---

## COLBENCH

### ColBench Ivy

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy_gpt-4_1 \
    --prefix col_ivy_o3_2025 \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json
```

### ColBench Zuck

```bash
# Step 1: Merge local traces
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_zuck_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json --force

# Step 2: Extract from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck_gpt-4_1 \
    --prefix col_zuck_o3_2025 \
    --prefix col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck_o4-mini_2025-04-16_low \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json
```

---

## RUBRIC EVALUATION

After all merges and Weave extractions complete, run:

### SciCode
```bash
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/scicode_lady_openai_gpt-4_1.json \
    --trace-file traces/scicode_lady_openai_o3_2025.json \
    --trace-file traces/scicode_lady_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_lady_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### CoreBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/prop_openai_gpt-4_1.json \
    --trace-file traces/prop_openai_o3_2025.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/iter1_openai_gpt-4_1.json \
    --trace-file traces/iter1_openai_o3_2025.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### ScienceAgentBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_openai_gpt-5_2025.json \
    --trace-file traces/sab_mate_openai_o3-mini_2025.json \
    --trace-file traces/sab_cow_openai_gpt-4_1.json \
    --trace-file traces/sab_cow_openai_o3_2025.json \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --trace-file traces/sab_husky_openai_o3-mini_2025.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### ColBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/col_ivy_gpt-4_1.json \
    --trace-file traces/col_ivy_o3_2025.json \
    --trace-file traces/col_ivy_o4-mini_2025-04-16_high.json \
    --trace-file traces/col_ivy_o4-mini_2025-04-16_low.json \
    --trace-file traces/col_zuck_gpt-4_1.json \
    --trace-file traces/col_zuck_o3_2025.json \
    --trace-file traces/col_zuck_o4-mini_2025-04-16_high.json \
    --trace-file traces/col_zuck_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

---

## JUDGE AGGREGATION

After rubric evaluations complete:

### SciCode
```bash
python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --pattern "scicode_lady_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### CoreBench
```bash
python scripts/judge.py \
    --pattern "prop_openai_*" \
    --pattern "iter1_openai_*" \
    --rubric-dir rubrics_output/corebench \
    --output judge_output/corebench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### ScienceAgentBench
```bash
python scripts/judge.py \
    --pattern "sab_mate_*" \
    --pattern "sab_cow_*" \
    --pattern "sab_husky_*" \
    --rubric-dir rubrics_output/scienceagentbench \
    --output judge_output/scienceagentbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### ColBench
```bash
python scripts/judge.py \
    --pattern "col_ivy_*" \
    --pattern "col_zuck_*" \
    --rubric-dir rubrics_output/colbench \
    --output judge_output/colbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

---

## Expected Output Files

### After Merge + Weave:
- `traces/scicode_honey_openai_gpt-4_1.json`
- `traces/scicode_honey_openai_o3_2025.json`
- `traces/scicode_honey_openai_o4-mini_2025-04-16_high.json`
- `traces/scicode_honey_openai_o4-mini_2025-04-16_low.json`
- `traces/scicode_lady_openai_*.json` (4 files)
- `traces/prop_openai_*.json` (4 files)
- `traces/iter1_openai_*.json` (4 files)
- `traces/sab_mate_openai_*.json` (2 files)
- `traces/sab_cow_openai_*.json` (2 files)
- `traces/sab_husky_openai_*.json` (2 files)
- `traces/col_ivy_*.json` (4 files)
- `traces/col_zuck_*.json` (4 files)

### After Rubric Evaluation:
- `rubrics_output/scicode/*.csv`
- `rubrics_output/corebench/*.csv`
- `rubrics_output/scienceagentbench/*.csv`
- `rubrics_output/colbench/*.csv`

### After Judge:
- `judge_output/scicode_verdict.csv`
- `judge_output/corebench_verdict.csv`
- `judge_output/scienceagentbench_verdict.csv`
- `judge_output/colbench_verdict.csv`
