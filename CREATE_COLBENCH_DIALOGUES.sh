#!/bin/bash
# Create _WITH_DIALOGUES.json files for ColBench from results directory
# This is the CORRECT way to get conversation logs for ColBench (not Weave extraction)

set -e

echo "============================================================"
echo "Creating ColBench _WITH_DIALOGUES.json files"
echo "============================================================"
echo ""

RESULTS_DIR="results/colbench_backend_programming"

# COL_IVY (9 models)
echo "=== COL_IVY ==="

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_gpt-4_1_2025-04-14_*" \
    --output traces/col_ivy_openai_gpt-4_1_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_o3_2025-04-16_medium_*" \
    --output traces/col_ivy_openai_o3_medium_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_o4-mini_2025-04-16_high_*" \
    --output traces/col_ivy_openai_o4-mini_high_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_o4-mini_2025-04-16_low_*" \
    --output traces/col_ivy_openai_o4-mini_low_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_gpt-5_2025-08-07_medium_*" \
    --output traces/col_ivy_openai_gpt-5_medium_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_gpt-5-mini_2025-08-07_*" \
    --output traces/col_ivy_openai_gpt-5-mini_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_gpt-4o_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_gpt-4o_2024-11-20_*" \
    --output traces/col_ivy_openai_gpt-4o_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_o3-mini_2025-01-31_high_*" \
    --output traces/col_ivy_openai_o3-mini_high_WITH_DIALOGUES.json

python scripts/add_colbench_dialogues.py \
    traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json \
    --results-dir "$RESULTS_DIR" \
    --run-pattern "col_ivy_DeepSeek-R1_1_*" \
    --output traces/col_ivy_DeepSeek-R1_WITH_DIALOGUES.json

echo ""
echo "✓ COL_IVY dialogues created!"
echo ""

# COL_ZUCK already has _WITH_DIALOGUES.json files
echo "COL_ZUCK: Using existing _WITH_DIALOGUES.json files"
ls -1 traces/col_zuck_*_WITH_DIALOGUES.json

echo ""
echo "============================================================"
echo "COMPLETE! ColBench traces with dialogues ready"
echo "============================================================"
echo ""
echo "Next step: Run rubric evaluation"
echo "  ./RUBRIC_COMMANDS.sh colbench"
