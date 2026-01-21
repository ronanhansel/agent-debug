#!/bin/bash
# RUBRIC EVALUATION AND JUDGE AGGREGATION COMMANDS
# Run these AFTER merge_and_weave_only.sh completes

# ============================================================
# STEP 1: RUBRIC EVALUATION
# ============================================================

echo "============================================================"
echo "STEP 1: RUBRIC EVALUATION"
echo "============================================================"
echo ""

# --- SCICODE ---
echo "=== SciCode Rubric Evaluation ==="
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

echo ""
echo "=== CoreBench Rubric Evaluation ==="
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

echo ""
echo "=== ScienceAgentBench Rubric Evaluation ==="
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

echo ""
echo "=== ColBench Rubric Evaluation ==="
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

echo ""
echo "============================================================"
echo "STEP 2: JUDGE AGGREGATION"
echo "============================================================"
echo ""

# --- SCICODE ---
echo "=== SciCode Judge Aggregation ==="
python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --pattern "scicode_lady_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

echo ""
echo "=== CoreBench Judge Aggregation ==="
python scripts/judge.py \
    --pattern "prop_openai_*" \
    --pattern "iter1_openai_*" \
    --rubric-dir rubrics_output/corebench \
    --output judge_output/corebench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

echo ""
echo "=== ScienceAgentBench Judge Aggregation ==="
python scripts/judge.py \
    --pattern "sab_mate_*" \
    --pattern "sab_cow_*" \
    --pattern "sab_husky_*" \
    --rubric-dir rubrics_output/scienceagentbench \
    --output judge_output/scienceagentbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

echo ""
echo "=== ColBench Judge Aggregation ==="
python scripts/judge.py \
    --pattern "col_ivy_*" \
    --pattern "col_zuck_*" \
    --rubric-dir rubrics_output/colbench \
    --output judge_output/colbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

echo ""
echo "============================================================"
echo "ALL DONE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Rubrics: rubrics_output/*/*.csv"
echo "  - Verdicts: judge_output/*_verdict.csv"
echo ""
