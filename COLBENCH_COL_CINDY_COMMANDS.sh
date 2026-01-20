#!/bin/bash
# ColBench Grading Commands for col_cindy run

PREFIX=col_cindy
export WANDB_API_KEY="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"

echo "=========================================="
echo "STEP 1: Merging traces"
echo "=========================================="

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

echo ""
echo "=========================================="
echo "STEP 2: Adding dialogue history"
echo "=========================================="

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

echo ""
echo "=========================================="
echo "STEP 3: Running rubric evaluation"
echo "=========================================="

python scripts/eval_rubric.py \
    --trace-file "traces/${PREFIX}_openai_gpt-4_1_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o3_low_WITH_DIALOGUES.json" \
    --trace-file "traces/${PREFIX}_openai_o4-mini_high_WITH_DIALOGUES.json" \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

echo ""
echo "=========================================="
echo "STEP 4: Aggregating verdicts"
echo "=========================================="

python scripts/judge.py \
    --rubric-dir rubrics_output/colbench \
    --output "judge_output/${PREFIX}_verdict.csv"

echo ""
echo "=========================================="
echo "✅ COMPLETE!"
echo "=========================================="
echo "Results: judge_output/${PREFIX}_verdict.csv"
