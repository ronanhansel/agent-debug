# ColBench Traces Ready for Rubric Evaluation

## ✅ Completed Steps

### 1. Merged Individual Task Traces (56 tasks per model)
- `traces/col_tommy_openai_gpt-4_1_MERGED_UPLOAD.json`
- `traces/col_tommy_openai_o3_low_MERGED_UPLOAD.json`
- `traces/col_tommy_openai_o4-mini_high_MERGED_UPLOAD.json`

### 2. Added Dialogue History for Rubric Evaluation
- `traces/col_tommy_openai_gpt-4_1_WITH_DIALOGUES.json` ✅ 56/56 tasks
- `traces/col_tommy_openai_o3_low_WITH_DIALOGUES.json` ✅ 56/56 tasks
- `traces/col_tommy_openai_o4-mini_high_WITH_DIALOGUES.json` ✅ 56/56 tasks

## 📊 Current Results Summary

| Model | Total Tasks | Successful | Failed | Avg Score | Accuracy |
|-------|-------------|------------|--------|-----------|----------|
| GPT-4.1 | 56 | 35 | 21 | 0.755 | 62.5% |
| O3-low | 56 | 39 | 17 | 0.730 | 69.6% |
| O4-mini-high | 56 | 37 | 19 | 0.741 | 66.1% |

## 🔍 Trace Format (Ready for Rubrics)

Each `*_WITH_DIALOGUES.json` file contains:

```json
{
  "config": {
    "run_id": "col_tommy_openai_gpt-4_1_MERGED_UPLOAD",
    "benchmark_name": "colbench_backend_programming"
  },
  "results": {
    "successful_tasks": ["31", "44", ...],  // 35 tasks with score == 1.0
    "failed_tasks": ["1", "4", "8", ...],   // 21 tasks with score < 1.0
    "average_correctness": 0.755,
    "accuracy": 0.625
  },
  "raw_logging_results": [
    {
      "task_id": "1",
      "score": 0.9,
      "answer": "...generated code...",
      "dialogue_history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...  // 10 turns total
      ],
      "task": {
        "test_cases": {"test1": "...", "test2": "...", ...},
        "ground_truth": "...reference implementation..."
      }
    }
  ]
}
```

## ▶️ Next Steps (For You to Run)

### Step 1: Run Rubric Evaluation

```bash
export WANDB_API_KEY="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"

python scripts/eval_rubric.py \
    --trace-file traces/col_tommy_openai_gpt-4_1_WITH_DIALOGUES.json \
    --trace-file traces/col_tommy_openai_o3_low_WITH_DIALOGUES.json \
    --trace-file traces/col_tommy_openai_o4-mini_high_WITH_DIALOGUES.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
```

**What this does:**
- Evaluates only the **failed tasks** (21 + 17 + 19 = 57 total failed across all models)
- Uses ColBench rubric to check for:
  - Simulated user issues (contradictory feedback, unclear responses)
  - Hidden information problems (not discoverable through dialogue)
  - Test case fairness (arbitrary requirements, subjective grading)
- Outputs CSVs to `rubrics_output/colbench_backend_programming/`

### Step 2: Aggregate Verdicts

```bash
python scripts/judge.py \
    --rubric-dir rubrics_output/colbench_backend_programming \
    --output judge_output/colbench_backend_programming_verdict.csv
```

**What this does:**
- Aggregates rubric evaluations across all 3 models
- Applies cross-model consensus
- Outputs final verdict:
  - **Grade 0**: Agent capability issue
  - **Grade 1**: Benchmark defect (Intrinsic Formation Error)

### Step 3: Review Results

```bash
# View verdict summary
python -c "
import pandas as pd
df = pd.read_csv('judge_output/colbench_backend_programming_verdict.csv')
print('Total tasks evaluated:', len(df))
print('Grade 0 (agent issue):', (df['final_grade'] == 0).sum())
print('Grade 1 (benchmark defect):', (df['final_grade'] == 1).sum())
print('\nTasks with benchmark defects:')
print(df[df['final_grade'] == 1][['task_id', 'num_evaluations', 'reasoning']].to_string())
"
```

## 📁 File Locations

- **Traces with dialogues**: `traces/col_tommy_openai_*_WITH_DIALOGUES.json`
- **Rubric template**: `rubric_templates/colbench.txt`
- **Rubric outputs**: `rubrics_output/colbench_backend_programming/`
- **Final verdict**: `judge_output/colbench_backend_programming_verdict.csv`

## 🔧 Troubleshooting

If rubric evaluation fails:
1. Check WANDB_API_KEY is exported
2. Verify rubric template exists: `cat rubric_templates/colbench.txt | head`
3. Check trace format: `python -c "import json; print(json.load(open('traces/col_tommy_openai_gpt-4_1_WITH_DIALOGUES.json')).keys())"`

---

**Status**: ✅ All traces ready for rubric evaluation. Run Step 1 above to start grading.
