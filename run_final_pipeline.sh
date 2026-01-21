#!/bin/bash
# Final corrected pipeline - handles missing traces gracefully
set -e

# Note: WANDB_API_KEY must be set in your environment before running
# export WANDB_API_KEY=your_key_here

LOG_DIR="logs/final_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "FINAL PIPELINE: MERGE + WEAVE + RUBRIC"
echo "============================================================"
echo ""

# Helper function: merge if pattern exists
merge_if_exists() {
    local pattern=$1
    local output=$2
    if compgen -G "$pattern" > /dev/null; then
        python scripts/merge_traces.py --input "$pattern" --output "$output" --force 2>&1 | tail -1
    else
        echo "  (no traces matching: $pattern)"
    fi
}

# Helper function: extract from weave
weave_extract() {
    local project=$1
    shift
    python scripts/extract_weave_traces.py --project "$project" "$@" 2>&1 | grep -E "(Wrote|Remote calls matched|WARNING)" | tail -20
}

# ============================================================
# SCICODE
# ============================================================
echo "=== [1/4] SCICODE ==="

echo "Merging scicode_honey..."
merge_if_exists 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' 'traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' 'traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' 'traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json'

echo "Merging scicode_lady..."
merge_if_exists 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' 'traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' 'traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' 'traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json'

echo "Extracting from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode" \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json

weave_extract "ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode" \
    --prefix scicode_lady_openai_gpt-4_1 \
    --prefix scicode_lady_openai_o3_2025 \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json

echo "Running rubric evaluation..."
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
    --failed-only -y --max-batch-messages 1000 2>&1 | tail -20

echo "✓ SciCode DONE"
echo ""

# ============================================================
# COREBENCH
# ============================================================
echo "=== [2/4] COREBENCH ==="

echo "Merging prop..."
merge_if_exists 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' 'traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' 'traces/prop_openai_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' 'traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' 'traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json'

echo "Merging iter1..."
merge_if_exists 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' 'traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' 'traces/iter1_openai_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' 'traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' 'traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json'

echo "Extracting from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard" \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025 \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json

weave_extract "ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard" \
    --prefix iter1_openai_gpt-4_1 \
    --prefix iter1_openai_o3_2025 \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json

echo "Running rubric evaluation..."
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
    --failed-only -y --max-batch-messages 1000 2>&1 | tail -20

echo "✓ CoreBench DONE"
echo ""

# ============================================================
# SAB
# ============================================================
echo "=== [3/4] SCIENCEAGENTBENCH ==="

echo "Merging sab_mate..."
merge_if_exists 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' 'traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' 'traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json'

echo "Merging sab_cow..."
merge_if_exists 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' 'traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json'

echo "Merging sab_husky..."
merge_if_exists 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/sab_husky__sab_husky_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' 'traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json'

echo "Extracting from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench" \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json

weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench" \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json

weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench" \
    --prefix sab_husky_openai_gpt-4_1 \
    --prefix sab_husky_openai_o3-mini_2025 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json

echo "Running rubric evaluation..."
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_openai_gpt-5_2025.json \
    --trace-file traces/sab_mate_openai_o3-mini_2025.json \
    --trace-file traces/sab_cow_openai_gpt-4_1.json \
    --trace-file traces/sab_cow_openai_o3_2025.json \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --trace-file traces/sab_husky_openai_o3-mini_2025.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y --max-batch-messages 1000 2>&1 | tail -20

echo "✓ SAB DONE"
echo ""

# ============================================================
# COLBENCH
# ============================================================
echo "=== [4/4] COLBENCH ==="

echo "Merging col_ivy..."
merge_if_exists 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' 'traces/col_ivy_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' 'traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' 'traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json'

echo "Merging col_zuck..."
merge_if_exists 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_UPLOAD.json' 'traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_UPLOAD.json' 'traces/col_zuck_o3_medium_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_UPLOAD.json' 'traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json'
merge_if_exists 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_UPLOAD.json' 'traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json'

echo "Extracting from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming" \
    --prefix col_ivy_gpt-4_1 \
    --prefix col_ivy_o3_2025 \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json

weave_extract "ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming" \
    --prefix col_zuck_gpt-4_1 \
    --prefix col_zuck_o3_2025 \
    --prefix col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck_o4-mini_2025-04-16_low \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json

echo "Running rubric evaluation..."
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
    --failed-only -y --max-batch-messages 1000 2>&1 | tail -20

echo "✓ ColBench DONE"
echo ""

echo "============================================================"
echo "COMPLETE! Rubric results saved to rubrics_output/"
echo "============================================================"
ls -lh rubrics_output/*/*.csv 2>/dev/null | wc -l | xargs echo "Total CSV files:"
