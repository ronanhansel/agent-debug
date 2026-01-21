# Pipeline Execution Guide

## Prerequisites

Set your WANDB_API_KEY:
```bash
export WANDB_API_KEY=your_key_here
```

---

## Usage

All scripts now support running specific benchmarks or all at once:

```bash
# Run all benchmarks
./FINAL_COMMANDS.sh
./RUBRIC_COMMANDS.sh
./JUDGE_COMMANDS.sh

# Run only ColBench
./FINAL_COMMANDS.sh colbench
./RUBRIC_COMMANDS.sh colbench
./JUDGE_COMMANDS.sh colbench

# Other options: scicode, corebench, sab
./FINAL_COMMANDS.sh scicode
./FINAL_COMMANDS.sh corebench
./FINAL_COMMANDS.sh sab
```

---

## Step-by-Step

### Step 1: Merge Local Traces + Extract from Weave

```bash
./FINAL_COMMANDS.sh scicode      # SciCode, CoreBench, SAB
./FINAL_COMMANDS.sh corebench
./FINAL_COMMANDS.sh sab
```

This will:
- Merge all local trace files for each model
- Extract conversation logs from Weave
- Create final trace files with complete data

**Output**: Trace files in `traces/` directory

### Step 1b: ColBench - Extract Dialogues from Results (SPECIAL PROCESS)

**ColBench does NOT use Weave extraction.** Instead, it extracts dialogue history from the `results/` directory:

```bash
# First, merge the traces
./FINAL_COMMANDS.sh colbench

# Then, add dialogue history from results directory
./CREATE_COLBENCH_DIALOGUES.sh
```

This creates `_WITH_DIALOGUES.json` files from `results/colbench_backend_programming/`

**Output**: `traces/*_WITH_DIALOGUES.json` files

---

### Step 2: Run Rubric Evaluation

```bash
./RUBRIC_COMMANDS.sh              # All benchmarks
# OR
./RUBRIC_COMMANDS.sh colbench     # Just ColBench
```

This evaluates each failed task against rubrics to identify benchmark defects.

**Output**: CSV files in `rubrics_output/{scicode,corebench,scienceagentbench,colbench}/`

---

### Step 3: Aggregate Verdicts

```bash
./JUDGE_COMMANDS.sh              # All benchmarks
# OR
./JUDGE_COMMANDS.sh colbench     # Just ColBench
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
- ✅ Fixed `extract_weave_traces.py` type checking for raw_eval_results (handles both dict and list)
- ✅ Fixed `run_corebench_fixes.py` prefix handling for merged traces
- ✅ Added benchmark selection flags to all scripts
- ✅ Identified correct model names for each project
