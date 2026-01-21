#!/bin/bash
# Complete pipeline: Merge local + Weave fetch + Rubric eval for ALL models
set -e

# Note: WANDB_API_KEY must be set in your environment before running
# export WANDB_API_KEY=your_key_here

LOG_DIR="logs/full_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "FULL PIPELINE: ALL BENCHMARKS + ALL MODELS"
echo "============================================================"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================
# STEP 1: MERGE ALL LOCAL TRACES + WEAVE FETCH
# ============================================================

echo "STEP 1/2: Merging local traces and fetching from Weave..."
echo ""

# --- SCICODE HONEY ---
echo "[1/9] SciCode Honey..."
{
    python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
        --prefix scicode_honey_openai_gpt-4_1 \
        --prefix scicode_honey_openai_o3_2025 \
        --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
        --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
        --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/scicode_honey.log"
echo "   ✓ Done (log: $LOG_DIR/scicode_honey.log)"

# --- SCICODE LADY ---
echo "[2/9] SciCode Lady..."
{
    python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
        --prefix scicode_lady_openai_gpt-4_1 \
        --prefix scicode_lady_openai_o3_2025 \
        --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
        --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
        --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/scicode_lady.log"
echo "   ✓ Done (log: $LOG_DIR/scicode_lady.log)"

# --- COREBENCH PROP ---
echo "[3/9] CoreBench Prop..."
{
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
        --prefix prop_openai_gpt-4_1 \
        --prefix prop_openai_o3_2025 \
        --prefix prop_openai_o4-mini_2025-04-16_high \
        --prefix prop_openai_o4-mini_2025-04-16_low \
        --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/prop_corebench.log"
echo "   ✓ Done (log: $LOG_DIR/prop_corebench.log)"

# --- COREBENCH ITER1 ---
echo "[4/9] CoreBench Iter1..."
{
    python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard \
        --prefix iter1_openai_gpt-4_1 \
        --prefix iter1_openai_o3_2025 \
        --prefix iter1_openai_o4-mini_2025-04-16_high \
        --prefix iter1_openai_o4-mini_2025-04-16_low \
        --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/iter1_corebench.log"
echo "   ✓ Done (log: $LOG_DIR/iter1_corebench.log)"

# --- SAB MATE ---
echo "[5/9] ScienceAgentBench Mate..."
{
    python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
        --prefix sab_mate_openai_gpt-5_2025 \
        --prefix sab_mate_openai_o3-mini_2025 \
        --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
        --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/sab_mate.log"
echo "   ✓ Done (log: $LOG_DIR/sab_mate.log)"

# --- SAB COW ---
echo "[6/9] ScienceAgentBench Cow..."
{
    python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
        --prefix sab_cow_openai_gpt-4_1 \
        --prefix sab_cow_openai_o3_2025 \
        --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/sab_cow.log"
echo "   ✓ Done (log: $LOG_DIR/sab_cow.log)"

# --- SAB HUSKY ---
echo "[7/9] ScienceAgentBench Husky..."
{
    python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_husky_openai_o3_medium_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
        --prefix sab_husky_openai_gpt-4_1 \
        --prefix sab_husky_openai_o3_2025 \
        --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/sab_husky_openai_o3_medium_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/sab_husky.log"
echo "   ✓ Done (log: $LOG_DIR/sab_husky.log)"

# --- COLBENCH IVY ---
echo "[8/9] ColBench Ivy..."
{
    python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
        --prefix col_ivy_gpt-4_1 \
        --prefix col_ivy_o3_2025 \
        --prefix col_ivy_o4-mini_2025-04-16_high \
        --prefix col_ivy_o4-mini_2025-04-16_low \
        --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/col_ivy.log"
echo "   ✓ Done (log: $LOG_DIR/col_ivy.log)"

# --- COLBENCH ZUCK ---
echo "[9/9] ColBench Zuck..."
{
    python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_zuck_o3_medium_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --force 2>&1
    python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json --force 2>&1

    python scripts/extract_weave_traces.py \
        --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
        --prefix col_zuck_gpt-4_1 \
        --prefix col_zuck_o3_2025 \
        --prefix col_zuck_o4-mini_2025-04-16_high \
        --prefix col_zuck_o4-mini_2025-04-16_low \
        --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
        --merge-input traces/col_zuck_o3_medium_MERGED_UPLOAD.json \
        --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json \
        --merge-input traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json 2>&1
} > "$LOG_DIR/col_zuck.log"
echo "   ✓ Done (log: $LOG_DIR/col_zuck.log)"

echo ""
echo "STEP 1 COMPLETED: All traces merged and fetched!"
echo ""

# ============================================================
# STEP 2: RUN RUBRIC EVALUATIONS
# ============================================================

echo "STEP 2/2: Running rubric evaluations..."
echo ""

# --- SCICODE ---
echo "[1/4] SciCode rubric evaluation..."
{
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
        --failed-only -y --max-batch-messages 1000 2>&1
} > "$LOG_DIR/rubric_scicode.log"
echo "   ✓ Done (log: $LOG_DIR/rubric_scicode.log)"

# --- COREBENCH ---
echo "[2/4] CoreBench rubric evaluation..."
{
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
        --failed-only -y --max-batch-messages 1000 2>&1
} > "$LOG_DIR/rubric_corebench.log"
echo "   ✓ Done (log: $LOG_DIR/rubric_corebench.log)"

# --- SAB ---
echo "[3/4] ScienceAgentBench rubric evaluation..."
{
    python scripts/eval_rubric.py \
        --trace-file traces/sab_mate_openai_gpt-5_2025.json \
        --trace-file traces/sab_mate_openai_o3-mini_2025.json \
        --trace-file traces/sab_cow_openai_gpt-4_1.json \
        --trace-file traces/sab_cow_openai_o3_2025.json \
        --trace-file traces/sab_husky_openai_gpt-4_1.json \
        --trace-file traces/sab_husky_openai_o3_2025.json \
        --rubric rubric_templates/scienceagentbench.txt \
        --rubric-model openai:gpt-5.2 \
        --failed-only -y --max-batch-messages 1000 2>&1
} > "$LOG_DIR/rubric_sab.log"
echo "   ✓ Done (log: $LOG_DIR/rubric_sab.log)"

# --- COLBENCH ---
echo "[4/4] ColBench rubric evaluation..."
{
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
        --failed-only -y --max-batch-messages 1000 2>&1
} > "$LOG_DIR/rubric_colbench.log"
echo "   ✓ Done (log: $LOG_DIR/rubric_colbench.log)"

echo ""
echo "============================================================"
echo "FULL PIPELINE COMPLETED!"
echo "============================================================"
echo ""
echo "Results summary:"
echo ""
ls -lh rubrics_output/*/*.csv 2>/dev/null | wc -l | xargs echo "  Total rubric CSV files created:"
echo ""
echo "To view results:"
echo "  ls rubrics_output/scicode/"
echo "  ls rubrics_output/corebench/"
echo "  ls rubrics_output/scienceagentbench/"
echo "  ls rubrics_output/colbench/"
echo ""
echo "Logs saved in: $LOG_DIR/"
echo "============================================================"
