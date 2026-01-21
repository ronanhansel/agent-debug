#!/bin/bash
# MERGE + WEAVE ONLY (No rubric evaluation)
set -e

# Note: WANDB_API_KEY must be set in your environment before running
# export WANDB_API_KEY=your_key_here

LOG_DIR="logs/merge_weave_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "MERGE + WEAVE EXTRACTION ONLY"
echo "============================================================"
echo "Log directory: $LOG_DIR"
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

# Helper function: extract from weave (silent, log to file)
weave_extract() {
    local project=$1
    local log_file=$2
    shift 2
    python scripts/extract_weave_traces.py --project "$project" "$@" > "$log_file" 2>&1
    grep -E "(Wrote|Remote calls matched)" "$log_file" | tail -10
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

echo "Extracting scicode_honey from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode" "$LOG_DIR/weave_scicode_honey.log" \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json

echo "Extracting scicode_lady from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode" "$LOG_DIR/weave_scicode_lady.log" \
    --prefix scicode_lady_openai_gpt-4_1 \
    --prefix scicode_lady_openai_o3_2025 \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ SciCode completed"
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

echo "Extracting prop from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard" "$LOG_DIR/weave_prop.log" \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025 \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json

echo "Extracting iter1 from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard" "$LOG_DIR/weave_iter1.log" \
    --prefix iter1_openai_gpt-4_1 \
    --prefix iter1_openai_o3_2025 \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ CoreBench completed"
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

echo "Extracting sab_mate from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench" "$LOG_DIR/weave_sab_mate.log" \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json

echo "Extracting sab_cow from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench" "$LOG_DIR/weave_sab_cow.log" \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json

echo "Extracting sab_husky from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench" "$LOG_DIR/weave_sab_husky.log" \
    --prefix sab_husky_openai_gpt-4_1 \
    --prefix sab_husky_openai_o3-mini_2025 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json

echo "✓ SAB completed"
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

echo "Extracting col_ivy from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming" "$LOG_DIR/weave_col_ivy.log" \
    --prefix col_ivy_gpt-4_1 \
    --prefix col_ivy_o3_2025 \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json

echo "Extracting col_zuck from Weave..."
weave_extract "ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming" "$LOG_DIR/weave_col_zuck.log" \
    --prefix col_zuck_gpt-4_1 \
    --prefix col_zuck_o3_2025 \
    --prefix col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck_o4-mini_2025-04-16_low \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json

echo "✓ ColBench completed"
echo ""

echo "============================================================"
echo "MERGE + WEAVE COMPLETED!"
echo "============================================================"
echo ""
echo "Final trace files created:"
ls -1 traces/{scicode_honey,scicode_lady,prop,iter1,sab_mate,sab_cow,sab_husky,col_ivy,col_zuck}_*.json 2>/dev/null | grep -v "_MERGED_UPLOAD.json$" | wc -l | xargs echo "  Total trace files:"
echo ""
echo "Logs saved in: $LOG_DIR/"
echo "============================================================"
