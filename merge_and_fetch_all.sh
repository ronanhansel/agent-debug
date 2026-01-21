#!/bin/bash
# Complete merge and fetch script for all benchmarks
# Generated for: prop_corebench_hard, iter1_corebench_hard, scicode_lady_scicode, scicode_honey_scicode,
#                sab_mate_scienceagentbench, sab_cow_scienceagentbench, sab_husky_scienceagentbench,
#                col_ivy_colbench_backendprogramming, col_zuck_colbench_backendprogramming

set -e  # Exit on error

echo "=========================================="
echo "COREBENCH - prop_corebench_hard"
echo "=========================================="

# Merge prop_corebench_hard traces
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force

# Extract prop_corebench_hard from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --prefix prop_openai_gpt-4_1_2025-04-14 \
    --prefix prop_openai_o3_2025-04-16_medium \
    --prefix prop_openai_o3_2025-04-16_low \
    --prefix prop_openai_gpt-oss-120b_1 \
    --prefix prop_openai_gpt-5_2025-08-07_medium \
    --prefix prop_DeepSeek-R1_1 \
    --prefix prop_deepseek-ai_DeepSeek-V3-0324 \
    --prefix prop_openai_gpt-4o_2024-11-20 \
    --merge-input traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json

echo "=========================================="
echo "COREBENCH - iter1_corebench_hard"
echo "=========================================="

# Merge iter1_corebench_hard traces
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force

# Extract iter1_corebench_hard from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --prefix iter1_openai_gpt-4_1_2025-04-14 \
    --prefix iter1_openai_o3_2025-04-16_medium \
    --prefix iter1_openai_o3_2025-04-16_low \
    --prefix iter1_openai_gpt-oss-120b_1 \
    --prefix iter1_openai_gpt-5_2025-08-07_medium \
    --prefix iter1_DeepSeek-R1_1 \
    --prefix iter1_deepseek-ai_DeepSeek-V3-0324 \
    --prefix iter1_openai_gpt-4o_2024-11-20 \
    --merge-input traces/iter1_corebench_hard_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_o3-low_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/iter1_corebench_hard_gpt-4o_MERGED_UPLOAD.json

echo "=========================================="
echo "SCICODE - scicode_lady_scicode"
echo "=========================================="

# Merge scicode_lady traces
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_lady_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_lady_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_gpt-5_2025-08-07_*_UPLOAD.json' --output traces/scicode_lady_gpt-5_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_DeepSeek-R1_1_*_UPLOAD.json' --output traces/scicode_lady_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/scicode_lady_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/scicode_lady_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_lady_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/scicode_lady_o3-mini-high_MERGED_UPLOAD.json --force

# Extract scicode_lady from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
    --prefix scicode_lady_openai_o3_2025-04-16_medium \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_gpt-4_1_2025-04-14 \
    --prefix scicode_lady_openai_gpt-5_2025-08-07 \
    --prefix scicode_lady_DeepSeek-R1_1 \
    --prefix scicode_lady_deepseek-ai_DeepSeek-V3-0324 \
    --prefix scicode_lady_openai_gpt-5-mini_2025-08-07 \
    --prefix scicode_lady_openai_gpt-4o_2024-11-20 \
    --prefix scicode_lady_openai_o3-mini_2025-01-31_high \
    --merge-input traces/scicode_lady_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_gpt-5_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCICODE - scicode_honey_scicode"
echo "=========================================="

# Merge scicode_honey traces
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_gpt-5_2025-08-07_*_UPLOAD.json' --output traces/scicode_honey_gpt-5_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_DeepSeek-R1_1_*_UPLOAD.json' --output traces/scicode_honey_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/scicode_honey_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/scicode_honey_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/scicode_honey_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/scicode_honey_o3-mini-high_MERGED_UPLOAD.json --force

# Extract scicode_honey from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey_openai_o3_2025-04-16_medium \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_gpt-4_1_2025-04-14 \
    --prefix scicode_honey_openai_gpt-5_2025-08-07 \
    --prefix scicode_honey_DeepSeek-R1_1 \
    --prefix scicode_honey_deepseek-ai_DeepSeek-V3-0324 \
    --prefix scicode_honey_openai_gpt-5-mini_2025-08-07 \
    --prefix scicode_honey_openai_gpt-4o_2024-11-20 \
    --prefix scicode_honey_openai_o3-mini_2025-01-31_high \
    --merge-input traces/scicode_honey_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_gpt-5_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_mate_scienceagentbench"
echo "=========================================="

# Merge sab_mate traces
python scripts/merge_traces.py --input 'traces/sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate_DeepSeek-R1_1_*_UPLOAD.json' --output traces/sab_mate_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/sab_mate_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/sab_mate_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_o3-mini-high_MERGED_UPLOAD.json --force

# Extract sab_mate from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate_openai_gpt-5_2025-08-07_medium \
    --prefix sab_mate_DeepSeek-R1_1 \
    --prefix sab_mate_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_mate_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_mate_openai_gpt-4o_2024-11-20 \
    --prefix sab_mate_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_mate_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_cow_scienceagentbench"
echo "=========================================="

# Merge sab_cow traces
python scripts/merge_traces.py --input 'traces/sab_cow_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_cow_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow_DeepSeek-R1_1_*_UPLOAD.json' --output traces/sab_cow_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/sab_cow_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/sab_cow_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/sab_cow_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_cow_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_cow_o3-mini-high_MERGED_UPLOAD.json --force

# Extract sab_cow from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow_openai_gpt-5_2025-08-07_medium \
    --prefix sab_cow_DeepSeek-R1_1 \
    --prefix sab_cow_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_cow_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_cow_openai_gpt-4o_2024-11-20 \
    --prefix sab_cow_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_cow_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_husky_scienceagentbench"
echo "=========================================="

# Merge sab_husky traces
python scripts/merge_traces.py --input 'traces/sab_husky_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_husky_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky_DeepSeek-R1_1_*_UPLOAD.json' --output traces/sab_husky_deepseek-r1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/sab_husky_deepseek-v3_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/sab_husky_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/sab_husky_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/sab_husky_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_husky_o3-mini-high_MERGED_UPLOAD.json --force

# Extract sab_husky from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-5_2025-08-07_medium \
    --prefix sab_husky_DeepSeek-R1_1 \
    --prefix sab_husky_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_husky_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_husky_openai_gpt-4o_2024-11-20 \
    --prefix sab_husky_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_husky_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "COLBENCH - col_ivy_colbench_backendprogramming"
echo "=========================================="

# Merge col_ivy traces
python scripts/merge_traces.py --input 'traces/col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_ivy_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_ivy_o3-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_ivy_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_ivy_deepseek-r1_MERGED_UPLOAD.json --force

# Extract col_ivy from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy_gpt-4_1_2025-04-14 \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --prefix col_ivy_o3_2025-04-16_medium \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_gpt-5_2025-08-07_medium \
    --prefix col_ivy_gpt-4o_2024-11-20 \
    --prefix col_ivy_gpt-5-mini_2025-08-07 \
    --prefix col_ivy_o3-mini_2025-01-31_high \
    --prefix col_ivy_DeepSeek-R1_1 \
    --merge-input traces/col_ivy_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_deepseek-r1_MERGED_UPLOAD.json

echo "=========================================="
echo "COLBENCH - col_zuck_colbench_backendprogramming"
echo "=========================================="

# Merge col_zuck traces
python scripts/merge_traces.py --input 'traces/col_zuck_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4.1_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o4-mini-low_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_zuck_o3-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_zuck_gpt-5-medium_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_zuck_gpt-4o_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_zuck_o3-mini-high_MERGED_UPLOAD.json --force
python scripts/merge_traces.py --input 'traces/col_zuck_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_zuck_deepseek-r1_MERGED_UPLOAD.json --force

# Extract col_zuck from Weave
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck_gpt-4_1_2025-04-14 \
    --prefix col_zuck_o4-mini_2025-04-16_low \
    --prefix col_zuck_o3_2025-04-16_medium \
    --prefix col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck_gpt-5_2025-08-07_medium \
    --prefix col_zuck_gpt-4o_2024-11-20 \
    --prefix col_zuck_gpt-5-mini_2025-08-07 \
    --prefix col_zuck_o3-mini_2025-01-31_high \
    --prefix col_zuck_DeepSeek-R1_1 \
    --merge-input traces/col_zuck_gpt-4.1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini-low_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3-medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3-mini-high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_deepseek-r1_MERGED_UPLOAD.json

echo ""
echo "=========================================="
echo "ALL DONE! Now run rubric evaluation:"
echo "=========================================="
echo ""
echo "# CoreBench:"
echo "python scripts/eval_rubric.py --trace-file traces/prop_corebench_hard_*_MERGED_UPLOAD.json traces/iter1_corebench_hard_*_MERGED_UPLOAD.json --rubric rubric_templates/corebench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# SciCode:"
echo "python scripts/eval_rubric.py --trace-file traces/scicode_lady_*_MERGED_UPLOAD.json traces/scicode_honey_*_MERGED_UPLOAD.json --rubric rubric_templates/scicode.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# ScienceAgentBench:"
echo "python scripts/eval_rubric.py --trace-file traces/sab_mate_*_MERGED_UPLOAD.json traces/sab_cow_*_MERGED_UPLOAD.json traces/sab_husky_*_MERGED_UPLOAD.json --rubric rubric_templates/scienceagentbench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
echo ""
echo "# ColBench:"
echo "python scripts/eval_rubric.py --trace-file traces/col_ivy_*_MERGED_UPLOAD.json traces/col_zuck_*_MERGED_UPLOAD.json --rubric rubric_templates/colbench.txt --rubric-model openai:gpt-5.2 --failed-only -y --max-batch-messages 1000"
