# Pipeline Execution Guide

## Prerequisites

Set your WANDB_API_KEY:
```bash
export WANDB_API_KEY=your_key_here
```

---

## Step 1: Merge Local Traces + Extract from Weave

```bash
./FINAL_COMMANDS.sh
```

This will:
- Merge all local trace files for each model
- Extract conversation logs from Weave
- Create final trace files with complete data

**Output**: 36 trace files in `traces/` directory

---

## Step 2: Run Rubric Evaluation

```bash
./RUBRIC_COMMANDS.sh
```

This evaluates each failed task against rubrics to identify benchmark defects.

**Output**: CSV files in `rubrics_output/{scicode,corebench,scienceagentbench,colbench}/`

---

## Step 3: Aggregate Verdicts

```bash
./JUDGE_COMMANDS.sh
```

This aggregates rubric evaluations across models into final verdicts.

**Output**:
- `judge_output/scicode_verdict.csv`
- `judge_output/corebench_verdict.csv`
- `judge_output/scienceagentbench_verdict.csv`
- `judge_output/colbench_verdict.csv`

---

## Summary

**Total Trace Files**: 36
- SciCode: 14 (4 honey + 10 lady)
- CoreBench: 14 (10 prop + 4 iter1)
- SAB: 11 (6 mate + 4 cow + 1 husky)
- ColBench: 12 (9 ivy + 3 zuck)

**Fixed Issues**:
- ✅ Removed all hardcoded API keys from scripts
- ✅ Fixed `extract_weave_traces.py` type checking bugs
- ✅ Fixed `run_corebench_fixes.py` prefix handling for merged traces
- ✅ Identified correct model names for each project
