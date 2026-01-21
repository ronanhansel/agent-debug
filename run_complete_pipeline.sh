#!/bin/bash
# Complete two-step pipeline: Merge local traces, then fetch from Weave
set -e

# Note: WANDB_API_KEY must be set in your environment before running
# export WANDB_API_KEY=your_key_here

LOG_DIR="logs/complete_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "COMPLETE PIPELINE: Merge + Weave Extraction"
echo "============================================================"
echo ""

# ============================================================
# SCICODE HONEY
# ============================================================
echo "=== SciCode Honey ==="
echo "[Step 1/2] Merging local traces..."

python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force

echo "[Step 2/2] Fetching from Weave and merging..."

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ SciCode Honey completed"
echo ""

# ============================================================
# SCICODE LADY
# ============================================================
echo "=== SciCode Lady ==="
echo "[Step 1/2] Merging local traces..."

python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json --force

echo "[Step 2/2] Fetching from Weave and merging..."

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
    --prefix scicode_lady_openai_gpt-4_1 \
    --prefix scicode_lady_openai_o3_2025 \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ SciCode Lady completed"
echo ""

# ============================================================
# COREBENCH PROP
# ============================================================
echo "=== CoreBench Prop ==="
echo "[Step 1/2] Merging local traces..."

python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json --force

echo "[Step 2/2] Fetching from Weave and merging..."

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025 \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ CoreBench Prop completed"
echo ""

# ============================================================
# COREBENCH ITER1
# ============================================================
echo "=== CoreBench Iter1 ==="
echo "[Step 1/2] Merging local traces..."

python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force

echo "[Step 2/2] Fetching from Weave and merging..."

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard \
    --prefix iter1_openai_gpt-4_1 \
    --prefix iter1_openai_o3_2025 \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ CoreBench Iter1 completed"
echo ""

# ============================================================
# COLBENCH IVY
# ============================================================
echo "=== ColBench Ivy ==="
echo "[Step 1/2] Merging local traces..."

python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force

echo "[Step 2/2] Fetching from Weave and merging..."

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy_gpt-4_1 \
    --prefix col_ivy_o3_2025 \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json

echo "✓ ColBench Ivy completed"
echo ""

echo "============================================================"
echo "PIPELINE COMPLETED!"
echo "============================================================"
echo ""
echo "Now run rubric evaluation:"
echo ""
echo "python scripts/eval_rubric.py \\"
echo "    --trace-file traces/scicode_honey_openai_gpt-4_1.json \\"
echo "    --trace-file traces/scicode_honey_openai_o3_2025.json \\"
echo "    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \\"
echo "    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \\"
echo "    --rubric rubric_templates/scicode.txt \\"
echo "    --rubric-model openai:gpt-5.2 \\"
echo "    --failed-only -y --max-batch-messages 1000"
echo ""
