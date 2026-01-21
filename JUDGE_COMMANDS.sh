#!/bin/bash
# JUDGE AGGREGATION - Run after RUBRIC_COMMANDS.sh completes

# ============================================================
# JUDGE AGGREGATION
# ============================================================

# SCICODE
python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --pattern "scicode_lady_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

# COREBENCH
python scripts/judge.py \
    --pattern "prop_openai_*" \
    --pattern "iter1_openai_*" \
    --rubric-dir rubrics_output/corebench \
    --output judge_output/corebench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

# SAB
python scripts/judge.py \
    --pattern "sab_mate_*" \
    --pattern "sab_cow_*" \
    --pattern "sab_husky_*" \
    --rubric-dir rubrics_output/scienceagentbench \
    --output judge_output/scienceagentbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

# COLBENCH
python scripts/judge.py \
    --pattern "col_ivy_*" \
    --pattern "col_zuck_*" \
    --rubric-dir rubrics_output/colbench \
    --output judge_output/colbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y

echo "============================================================"
echo "ALL VERDICTS CREATED!"
echo "============================================================"
echo ""
ls -lh judge_output/
