#!/bin/bash
# CORRECTED merge and fetch script based on actual trace naming patterns
# Generated for: prop_corebench_hard, iter1_corebench_hard, scicode_lady_scicode, scicode_honey_scicode,
#                sab_mate_scienceagentbench, sab_cow_scienceagentbench, sab_husky_scienceagentbench,
#                col_ivy_colbench_backendprogramming, col_zuck_colbench_backendprogramming

set -e  # Exit on error

# WANDB_API_KEY should be set in your environment
# export WANDB_API_KEY=your_key_here

echo "=========================================="
echo "COREBENCH - prop_corebench_hard"
echo "=========================================="

# Note: Some CoreBench traces may already be merged - skip if MERGED file exists
if [ ! -f traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: prop o4-mini-high"
fi

if [ ! -f traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: prop o4-mini-low"
fi

if [ ! -f traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: prop gpt-4.1"
fi

if [ ! -f traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: prop o3-medium"
fi

if [ ! -f traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json --force || echo "Pattern not found: prop o3-low"
fi

if [ ! -f traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force || echo "Pattern not found: prop gpt-oss-120b"
fi

if [ ! -f traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: prop gpt-5-medium"
fi

if [ ! -f traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: prop deepseek-r1"
fi

if [ ! -f traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: prop deepseek-v3"
fi

if [ ! -f traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: prop gpt-4o"
fi

# Extract prop_corebench_hard from Weave
echo "Extracting prop_corebench_hard from Weave..."
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
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 o4-mini-high"
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 o4-mini-low"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 gpt-4.1"
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 o3-medium"
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-low_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 o3-low"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 gpt-oss-120b"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 gpt-5-medium"
python scripts/merge_traces.py --input 'traces/iter1_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 deepseek-r1"
python scripts/merge_traces.py --input 'traces/iter1_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 deepseek-v3"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: iter1 gpt-4o"

# Extract iter1_corebench_hard from Weave
echo "Extracting iter1_corebench_hard from Weave..."
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

# Note: Double prefix pattern scicode_lady__scicode_lady_
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady o3-medium"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady o4-mini-low"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady o4-mini-high"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady gpt-4.1"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-5_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady gpt-5"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_DeepSeek-R1_1_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady deepseek-r1"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_deepseek-ai_DeepSeek-V3-0324_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady deepseek-v3"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5-mini_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady gpt-5-mini"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4o_2024-11-20_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady gpt-4o"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3-mini_2025-01-31_high_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_lady o3-mini-high"

# Extract scicode_lady from Weave
echo "Extracting scicode_lady_scicode from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
    --prefix scicode_lady__scicode_lady_openai_o3_2025-04-16_medium \
    --prefix scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low \
    --prefix scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14 \
    --prefix scicode_lady__scicode_lady_openai_gpt-5_2025-08-07 \
    --prefix scicode_lady__scicode_lady_DeepSeek-R1_1 \
    --prefix scicode_lady__scicode_lady_deepseek-ai_DeepSeek-V3-0324 \
    --prefix scicode_lady__scicode_lady_openai_gpt-5-mini_2025-08-07 \
    --prefix scicode_lady__scicode_lady_openai_gpt-4o_2024-11-20 \
    --prefix scicode_lady__scicode_lady_openai_o3-mini_2025-01-31_high \
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

# Note: Double prefix pattern scicode_honey__scicode_honey_
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey o3-medium"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey o4-mini-low"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey o4-mini-high"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey gpt-4.1"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-5_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey gpt-5"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_DeepSeek-R1_1_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey deepseek-r1"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_deepseek-ai_DeepSeek-V3-0324_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey deepseek-v3"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5-mini_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey gpt-5-mini"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4o_2024-11-20_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey gpt-4o"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3-mini_2025-01-31_high_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: scicode_honey o3-mini-high"

# Extract scicode_honey from Weave
echo "Extracting scicode_honey_scicode from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey__scicode_honey_openai_o3_2025-04-16_medium \
    --prefix scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low \
    --prefix scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14 \
    --prefix scicode_honey__scicode_honey_openai_gpt-5_2025-08-07 \
    --prefix scicode_honey__scicode_honey_DeepSeek-R1_1 \
    --prefix scicode_honey__scicode_honey_deepseek-ai_DeepSeek-V3-0324 \
    --prefix scicode_honey__scicode_honey_openai_gpt-5-mini_2025-08-07 \
    --prefix scicode_honey__scicode_honey_openai_gpt-4o_2024-11-20 \
    --prefix scicode_honey__scicode_honey_openai_o3-mini_2025-01-31_high \
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

# Note: Double prefix pattern sab_mate__sab_mate_ and gpt-4_1 (underscore not period)
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_mate o3-mini-high"

# Extract sab_mate from Weave
echo "Extracting sab_mate_scienceagentbench from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium \
    --prefix sab_mate__sab_mate_DeepSeek-R1_1 \
    --prefix sab_mate__sab_mate_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_mate__sab_mate_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_mate__sab_mate_openai_gpt-4o_2024-11-20 \
    --prefix sab_mate__sab_mate_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_mate_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_cow_scienceagentbench"
echo "=========================================="

python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_cow o3-mini-high"

# Extract sab_cow from Weave
echo "Extracting sab_cow_scienceagentbench from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow__sab_cow_openai_gpt-5_2025-08-07_medium \
    --prefix sab_cow__sab_cow_DeepSeek-R1_1 \
    --prefix sab_cow__sab_cow_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_cow__sab_cow_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_cow__sab_cow_openai_gpt-4o_2024-11-20 \
    --prefix sab_cow__sab_cow_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_cow_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_husky_scienceagentbench"
echo "=========================================="

python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_deepseek-v3_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: sab_husky o3-mini-high"

# Extract sab_husky from Weave
echo "Extracting sab_husky_scienceagentbench from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky__sab_husky_openai_gpt-5_2025-08-07_medium \
    --prefix sab_husky__sab_husky_DeepSeek-R1_1 \
    --prefix sab_husky__sab_husky_deepseek-ai_DeepSeek-V3-0324 \
    --prefix sab_husky__sab_husky_openai_gpt-5-mini_2025-08-07 \
    --prefix sab_husky__sab_husky_openai_gpt-4o_2024-11-20 \
    --prefix sab_husky__sab_husky_openai_o3-mini_2025-01-31_high \
    --merge-input traces/sab_husky_gpt-5-medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_deepseek-r1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_deepseek-v3_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_o3-mini-high_MERGED_UPLOAD.json

echo "=========================================="
echo "COLBENCH - col_ivy_colbench_backendprogramming"
echo "=========================================="

# Note: Double prefix and NO openai prefix: col_ivy__col_ivy_gpt-4_1_2025-04-14
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy gpt-4.1"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy o4-mini-low"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy o3-medium"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy o4-mini-high"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5_2025-08-07_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy gpt-5-medium"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4o_2024-11-20_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy gpt-4o"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5-mini_2025-08-07_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy gpt-5-mini"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3-mini_2025-01-31_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy o3-mini-high"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_DeepSeek-R1_1_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: col_ivy deepseek-r1"

# Extract col_ivy from Weave
echo "Extracting col_ivy_colbench_backendprogramming from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy__col_ivy_gpt-4_1_2025-04-14 \
    --prefix col_ivy__col_ivy_o4-mini_2025-04-16_low \
    --prefix col_ivy__col_ivy_o3_2025-04-16_medium \
    --prefix col_ivy__col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy__col_ivy_gpt-5_2025-08-07_medium \
    --prefix col_ivy__col_ivy_gpt-4o_2024-11-20 \
    --prefix col_ivy__col_ivy_gpt-5-mini_2025-08-07 \
    --prefix col_ivy__col_ivy_o3-mini_2025-01-31_high \
    --prefix col_ivy__col_ivy_DeepSeek-R1_1 \
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

python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-4.1_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck gpt-4.1"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o4-mini-low_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck o4-mini-low"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o3-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck o3-medium"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o4-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck o4-mini-high"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5_2025-08-07_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-5-medium_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck gpt-5-medium"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4o_2024-11-20_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-4o_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck gpt-4o"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5-mini_2025-08-07_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck gpt-5-mini"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3-mini_2025-01-31_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o3-mini-high_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck o3-mini-high"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_DeepSeek-R1_1_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_deepseek-r1_MERGED_UPLOAD.json --force || echo "Pattern not found: col_zuck deepseek-r1"

# Extract col_zuck from Weave
echo "Extracting col_zuck_colbench_backendprogramming from Weave..."
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck__col_zuck_gpt-4_1_2025-04-14 \
    --prefix col_zuck__col_zuck_o4-mini_2025-04-16_low \
    --prefix col_zuck__col_zuck_o3_2025-04-16_medium \
    --prefix col_zuck__col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck__col_zuck_gpt-5_2025-08-07_medium \
    --prefix col_zuck__col_zuck_gpt-4o_2024-11-20 \
    --prefix col_zuck__col_zuck_gpt-5-mini_2025-08-07 \
    --prefix col_zuck__col_zuck_o3-mini_2025-01-31_high \
    --prefix col_zuck__col_zuck_DeepSeek-R1_1 \
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
