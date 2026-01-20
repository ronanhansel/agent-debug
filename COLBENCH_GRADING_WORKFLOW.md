# ColBench Grading Workflow

Complete step-by-step guide for evaluating ColBench traces and identifying benchmark defects.

## Prerequisites

```bash
# Set WANDB_API_KEY (from .env)
export WANDB_API_KEY="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"

# Verify you have traces from the fix runner
ls traces/<prefix>__<prefix>_*_UPLOAD.json | wc -l  # Should show ~56 files per model
```

## Quick Reference: Replace Variables

For any new run, replace these variables:
- `<prefix>`: Your run prefix (e.g., `col_tommy`, `col_cindy`, `col_jimmy`)
- `<output_name>`: Desired output name (e.g., `col_tommy`, `col_cindy`)

---

## Step 1: Merge Individual Task Traces

**Purpose:** Combine 56 single-task traces into one file per model

```bash
# Set your prefix
PREFIX=col_cindy  # CHANGE THIS

# Merge GPT-4.1 traces
python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_gpt-4_1-2025-04-14_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" \
    --force

# Merge O3-low traces
python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_o3-2025-04-16_low_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" \
    --force

# Merge O4-mini-high traces
python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_o4-mini-2025-04-16_high_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" \
    --force
```

**Output:** Creates 3 merged files with:
- `successful_tasks` / `failed_tasks` lists
- `average_correctness` and `accuracy` metrics
- `raw_eval_results` dict (task_id → score)

---

## Step 2: Add Dialogue History to Traces

**Purpose:** Extract dialogue history from result directories for rubric evaluation

```bash
# Set your prefix
PREFIX=col_cindy  # CHANGE THIS
TIMESTAMP=20260120_080618  # Find this with: ls results/colbench_backend_programming/${PREFIX}_* | head -1

# GPT-4.1
python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_gpt-4_1-2025-04-14_*" \
    --output "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json"

# O3-low
python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_o3-2025-04-16_low_*" \
    --output "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json"

# O4-mini-high
python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_o4-mini-2025-04-16_high_*" \
    --output "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json"
```

**Output:** Creates 3 files with `raw_logging_results` containing:
- `task_id`: Task identifier
- `score`: Evaluation score (0.0-1.0)
- `dialogue_history`: 10-turn agent-user conversation
- `answer`: Agent's final code submission
- `task`: Test cases and ground truth

**Verification:**
```bash
python3 -c "
import json
with open('traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json') as f:
    data = json.load(f)
print(f'Successful: {len(data[\"results\"][\"successful_tasks\"])}')
print(f'Failed: {len(data[\"results\"][\"failed_tasks\"])}')
print(f'Dialogues added: {len(data[\"raw_logging_results\"])}')
"
```

---

## Step 3: Run Rubric Evaluation

**Purpose:** Identify Intrinsic Formation Errors (IFEs) using LLM judge

```bash
# Set your prefix
PREFIX=col_cindy  # CHANGE THIS

python scripts/eval_rubric.py \
    --trace-file "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json" \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
```

**What this does:**
- Evaluates only failed tasks from all 3 models
- Uses ColBench rubric to check for:
  - Simulated user issues (contradictory feedback, unclear responses)
  - Hidden information problems (undiscoverable through dialogue)
  - Test case fairness (arbitrary requirements, subjective grading)
  - Infrastructure issues (parser errors, evaluation bugs)
- Outputs CSVs to: `rubrics_output/colbench/<trace_name>.csv`

**Expected output:**
```
Processing: ${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json
🧵 Found 56 unique tasks
🎯 Filtering to 21 failed task(s) out of 56 (--failed-only).
🧪 Running rubric 'colbench' on 21 agent runs...
```

---

## Step 4: Aggregate Verdicts

**Purpose:** Cross-model consensus to confirm IFEs

```bash
# Set your prefix
PREFIX=col_cindy  # CHANGE THIS

python scripts/judge.py \
    --rubric-dir rubrics_output/colbench \
    --output "judge_output/${PREFIX}_verdict.csv"
```

**What this does:**
- Aggregates rubric evaluations from all models
- Applies cross-model consensus (if multiple models agree → stronger evidence)
- Outputs final verdict per task:
  - **Grade 0**: Agent capability issue (agent could have succeeded)
  - **Grade 1**: Benchmark defect / IFE (no agent could succeed)

**Expected output:**
```
============================================================
SUMMARY (Binary Grades)
============================================================
IFE Confirmed (grade=1): X/Y
No IFE (grade=0):        Z/Y

Tasks with confirmed IFEs:
  - 49
  - ...
```

---

## Step 5: Review IFE Tasks

**Purpose:** Analyze each IFE and create fixes

```bash
# View verdict details
PREFIX=col_cindy  # CHANGE THIS

python3 << PYEOF
import pandas as pd
df = pd.read_csv('judge_output/${PREFIX}_verdict.csv')

print('=' * 70)
print('VERDICT SUMMARY')
print('=' * 70)
print(f'Total evaluated: {len(df)}')
print(f'Grade 0 (agent): {(df["final_grade"] == 0).sum()}')
print(f'Grade 1 (IFE):   {(df["final_grade"] == 1).sum()}')

ife_tasks = df[df['final_grade'] == 1]
if not ife_tasks.empty:
    print('\n' + '=' * 70)
    print('TASKS WITH CONFIRMED IFEs')
    print('=' * 70)
    for _, row in ife_tasks.iterrows():
        print(f'\nTask {row["task_id"]}:')
        print(f'  Models evaluated: {row["num_evaluations"]}')
        print(f'  Consensus: {row["reasoning"][:200]}...')
PYEOF
```

**For each IFE task, examine:**
1. Dialogue history (what user said vs what tests expect)
2. Ground truth implementation (arbitrary constants?)
3. Test case requirements (fair or hidden?)

---

## Step 6: Create Fixes (Example for Task 49)

**Purpose:** Document and fix benchmark defects

```bash
# Create fix directory
mkdir -p fixes/colbench_backend_programming/49

# Create README documenting the IFE
cat > fixes/colbench_backend_programming/49/README.md << 'READMEEOF'
# Task 49: [Brief Description]

## IFE Analysis
[Document what the IFE is, evidence from rubrics, dialogue quotes, etc.]

## Fix Strategy
[Explain what the fix does and why it's fair]
READMEEOF

# Create instruction override
cat > fixes/colbench_backend_programming/49/instruction_override.json << 'JSONEOF'
{
  "clarifications": [
    "Provide the missing information that was hidden/undiscoverable"
  ],
  "notes": "Explanation of why this fix is needed"
}
JSONEOF

# Create status file
cat > fixes/colbench_backend_programming/49/status.json << 'STATUSEOF'
{
  "fix_created": "2026-01-20",
  "verified": true,
  "fix_type": "instruction_override",
  "ife_category": "hidden_information_not_discoverable",
  "cross_model_agreement": "3/3 models graded 1.0"
}
STATUSEOF
```

---

## Step 7: Re-run with Fixes

**Purpose:** Verify fixes improve scores

```bash
# List available fixes
python scripts/run_colbench_fixes.py --list-fixes

# Run specific fixed task
python scripts/run_colbench_fixes.py \
    --task-id 49 \
    --prefix col_verified_ \
    --benchmark colbench_backend_programming \
    --docker

# Run all fixes
python scripts/run_colbench_fixes.py \
    --all \
    --prefix col_fixed_ \
    --benchmark colbench_backend_programming \
    --parallel 20 \
    --docker
```

---

## Complete One-Liner Workflow

```bash
# CONFIGURE THESE
PREFIX=col_cindy
export WANDB_API_KEY="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"

# STEP 1: Merge traces
python scripts/merge_traces.py --input "traces/${PREFIX}__${PREFIX}_gpt-4_1-2025-04-14_*_UPLOAD.json" --output "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" --force && \
python scripts/merge_traces.py --input "traces/${PREFIX}__${PREFIX}_o3-2025-04-16_low_*_UPLOAD.json" --output "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" --force && \
python scripts/merge_traces.py --input "traces/${PREFIX}__${PREFIX}_o4-mini-2025-04-16_high_*_UPLOAD.json" --output "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" --force && \
\
# STEP 2: Add dialogues
python scripts/add_colbench_dialogues.py "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" --results-dir results/colbench_backend_programming --run-pattern "${PREFIX}_gpt-4_1-2025-04-14_*" --output "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json" && \
python scripts/add_colbench_dialogues.py "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" --results-dir results/colbench_backend_programming --run-pattern "${PREFIX}_o3-2025-04-16_low_*" --output "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json" && \
python scripts/add_colbench_dialogues.py "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" --results-dir results/colbench_backend_programming --run-pattern "${PREFIX}_o4-mini-2025-04-16_high_*" --output "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json" && \
\
# STEP 3: Run rubric evaluation
python scripts/eval_rubric.py \
    --trace-file "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json" \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y && \
\
# STEP 4: Aggregate verdicts
python scripts/judge.py \
    --rubric-dir rubrics_output/colbench \
    --output "judge_output/${PREFIX}_verdict.csv" && \
\
echo "✅ Complete! Check judge_output/${PREFIX}_verdict.csv for IFEs"
```

---

## Detailed Step-by-Step Instructions

### Step 1: Merge Traces (3 models × 56 tasks → 3 files)

```bash
PREFIX=col_cindy  # CHANGE THIS

python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_gpt-4_1-2025-04-14_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" \
    --force

python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_o3-2025-04-16_low_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" \
    --force

python scripts/merge_traces.py \
    --input "traces/${PREFIX}__${PREFIX}_o4-mini-2025-04-16_high_*_UPLOAD.json" \
    --output "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" \
    --force
```

**Verify:**
```bash
ls -lh traces/${PREFIX}_openai_*_MERGED_UPLOAD.json
# Should show 3 files, each a few KB in size
```

---

### Step 2: Add Dialogue History

```bash
PREFIX=col_cindy  # CHANGE THIS

python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_gpt-4_1_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_gpt-4_1-2025-04-14_*" \
    --output "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json"

python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_o3_low_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_o3-2025-04-16_low_*" \
    --output "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json"

python scripts/add_colbench_dialogues.py \
    "traces/${PREFIX}_openai_o4-mini_high_MERGED_UPLOAD.json" \
    --results-dir results/colbench_backend_programming \
    --run-pattern "${PREFIX}_o4-mini-2025-04-16_high_*" \
    --output "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json"
```

**Expected output for each:**
```
Processing 56 tasks...
Found dialogue history for 56/56 tasks (0 missing)
Wrote traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json
```

**Verify:**
```bash
python3 -c "
import json
with open('traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json') as f:
    data = json.load(f)
print(f'Dialogues: {len(data[\"raw_logging_results\"])}')
print(f'Failed tasks: {len(data[\"results\"][\"failed_tasks\"])}')
"
```

---

### Step 3: Run Rubric Evaluation

```bash
PREFIX=col_cindy  # CHANGE THIS

python scripts/eval_rubric.py \
    --trace-file "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json" \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
```

**Expected output:**
```
[Azure] Direct TRAPI access configured
[Azure] Model resolved: openai:gpt-5.2 -> azure_openai:gpt-5.2_2025-12-11

Processing: ${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json
🧵 Found 56 unique tasks
🎯 Filtering to 21 failed task(s) out of 56 (--failed-only).
🧪 Running rubric 'colbench' on 21 agent runs with azure_openai:gpt-5.2_2025-12-11...
  Processing batch 1/1: 21 tasks...
🗂️  Rubric CSV written to rubrics_output/colbench/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.csv
```

**Output files:**
- `rubrics_output/colbench/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.csv`
- `rubrics_output/colbench/${PREFIX}_openai_o3_low_WITH_DIALOGUES.csv`
- `rubrics_output/colbench/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.csv`

**Verify:**
```bash
wc -l rubrics_output/colbench/${PREFIX}_*.csv
# Should show line counts matching number of failed tasks
```

---

### Step 4: Aggregate Verdicts

```bash
PREFIX=col_cindy  # CHANGE THIS

python scripts/judge.py \
    --rubric-dir rubrics_output/colbench \
    --output "judge_output/${PREFIX}_verdict.csv"
```

**Expected output:**
```
============================================================
SUMMARY (Binary Grades)
============================================================
IFE Confirmed (grade=1): X/Y
No IFE (grade=0):        Z/Y

Tasks with confirmed IFEs:
  - 49
  - ...

Output: judge_output/${PREFIX}_verdict.csv
```

**Verify:**
```bash
cat judge_output/${PREFIX}_verdict.csv
# Columns: task_id, final_grade, satisfies_rubric, num_evaluations, model_runs, reasoning
```

---

### Step 5: Review IFE Details

```bash
PREFIX=col_cindy  # CHANGE THIS

# View IFE summary
python3 << 'PYEOF'
import pandas as pd

df = pd.read_csv('judge_output/${PREFIX}_verdict.csv')
ife_tasks = df[df['final_grade'] == 1]

print('=' * 70)
print(f'IFE TASKS: {len(ife_tasks)} found')
print('=' * 70)

for _, row in ife_tasks.iterrows():
    task_id = row['task_id']
    print(f'\nTask {task_id}:')
    print(f'  Models evaluated: {row["num_evaluations"]}/{row["num_evaluations"]} agreed')
    print(f'  Reasoning: {row["reasoning"][:150]}...')
PYEOF

# View specific task dialogue
TASK_ID=49  # CHANGE THIS

python3 << PYEOF
import json
with open('traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json') as f:
    data = json.load(f)

for entry in data['raw_logging_results']:
    if entry['task_id'] == '${TASK_ID}':
        print(f'Task {TASK_ID} - Score: {entry["score"]}')
        print('\nDialogue:')
        for i, msg in enumerate(entry['dialogue_history'], 1):
            content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
            print(f'{i}. [{msg["role"]:10s}] {content}')
        print(f'\nAnswer (preview): {entry["answer"][:200]}...')
        print(f'\nGround truth (preview): {entry["task"]["ground_truth"][:200]}...')
        break
PYEOF
```

---

## Step 6: Create Fixes for IFE Tasks

For each task with `final_grade=1`, create a fix in `fixes/colbench_backend_programming/<task_id>/`

**Fix structure:**
```
fixes/colbench_backend_programming/<task_id>/
├── README.md                    # IFE analysis and evidence
├── instruction_override.json    # Clarifications to add to task
└── status.json                  # Fix metadata
```

**Example (Task 49):**

```json
// instruction_override.json
{
  "clarifications": [
    "For Saab model categorization, use the following horsepower ranges:",
    "  - 'Linear': horsepower < 200",
    "  - 'Aero': 200 <= horsepower < 240",
    "  - For horsepower >= 240, return 'Unknown'"
  ],
  "notes": "The simulated user couldn't provide ranges (said 'I don't know'), so test expectations were hidden/arbitrary."
}
```

**Principle:** Make evaluation FAIR (agent knows requirements), not EASY (agent still must implement correctly).

---

## Step 7: Re-run with Fixes Applied

```bash
# Re-run single task to verify fix
python scripts/run_colbench_fixes.py \
    --task-id 49 \
    --prefix col_verified_ \
    --benchmark colbench_backend_programming \
    --docker

# Re-run all fixed tasks
python scripts/run_colbench_fixes.py \
    --all \
    --prefix col_fixed_ \
    --benchmark colbench_backend_programming \
    --parallel 20 \
    --docker
```

**Then repeat Steps 1-4 with the new prefix** to compare before/after.

---

## Quick Reference Table

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Merge | `merge_traces.py` | 56 single-task UPLOAD.json | 3 MERGED files |
| 2. Add Dialogues | `add_colbench_dialogues.py` | MERGED + results/ dirs | 3 WITH_DIALOGUES files |
| 3. Rubric Eval | `eval_rubric.py` | WITH_DIALOGUES files | CSVs in rubrics_output/ |
| 4. Judge | `judge.py` | Rubric CSVs | Verdict CSV in judge_output/ |

---

## Common Issues

### Issue: "No failed tasks available for rubric evaluation"

**Cause:** Traces don't have dialogue history

**Fix:** Run Step 2 (`add_colbench_dialogues.py`) first

---

### Issue: "Found 0/56 tasks (56 missing)"

**Cause:** Wrong run pattern (doesn't match actual directory names)

**Fix:** Check actual directory names:
```bash
ls results/colbench_backend_programming/${PREFIX}_gpt-4_1* | head -3
# Use the pattern you see (usually ${PREFIX}_gpt-4_1-2025-04-14_*)
```

---

### Issue: Rubric evaluation runs but finds "1 unique tasks"

**Cause:** `raw_logging_results` format not recognized by rubric evaluator

**Fix:** This was fixed in `rubric_evaluator/cli.py` - make sure you have the latest version

---

## Files Generated

### Traces Directory
```
traces/
├── col_cindy__col_cindy_gpt-4_1-2025-04-14_1_UPLOAD.json     # Single-task (56 files per model)
├── col_cindy_openai_gpt-4_1_MERGED_UPLOAD.json               # Step 1: Merged
├── col_cindy_openai_gpt-4_1_WITH_DIALOGUES.json              # Step 2: With dialogues
└── (same for o3_low and o4-mini_high)
```

### Rubric Outputs
```
rubrics_output/colbench/
├── col_cindy_openai_gpt-4_1_WITH_DIALOGUES.csv
├── col_cindy_openai_o3_low_WITH_DIALOGUES.csv
└── col_cindy_openai_o4-mini_high_WITH_DIALOGUES.csv
```

### Judge Output
```
judge_output/
└── col_cindy_verdict.csv
```

### Fixes
```
fixes/colbench_backend_programming/
├── 49/
│   ├── README.md
│   ├── instruction_override.json
│   └── status.json
└── (other task IDs)
```

---

## Expected Timings

| Step | Tasks | Time Estimate |
|------|-------|---------------|
| 1. Merge | 168 files | ~1 second |
| 2. Add Dialogues | 168 tasks | ~10 seconds |
| 3. Rubric Eval | ~57 failed tasks | ~15-30 minutes (GPT-5.2) |
| 4. Judge | ~57 tasks | ~1 second |

**Total:** ~20-35 minutes for complete workflow

---

## Next Steps After Grading

1. **Review verdict.csv** - Identify all Grade 1 (IFE) tasks
2. **Create fixes** - For each IFE task in `fixes/colbench_backend_programming/<task_id>/`
3. **Re-run with fixes** - Use `run_colbench_fixes.py` with new prefix
4. **Compare metrics** - Before vs after fix application
5. **Document results** - Update this file or create summary

---

**Last updated:** 2026-01-20 (for col_tommy, col_cindy runs)
