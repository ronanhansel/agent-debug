#!/bin/bash
# Merge and fetch CoreBench traces (prop + iter1)
set -e

echo "=========================================="
echo "COREBENCH - prop_corebench_hard"
echo "=========================================="

# Skip merge if already exists
if [ ! -f traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop o4-mini-high"
fi

if [ ! -f traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop o4-mini-low"
fi

if [ ! -f traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop gpt-4.1"
fi

if [ ! -f traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop o3-medium"
fi

if [ ! -f traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_o3-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop o3-low"
fi

if [ ! -f traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop gpt-oss-120b"
fi

if [ ! -f traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop gpt-5-medium"
fi

if [ ! -f traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop deepseek-r1"
fi

if [ ! -f traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop deepseek-v3"
fi

if [ ! -f traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json ]; then
    python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/prop_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: prop gpt-4o"
fi

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

python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 o4-mini-high"
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 o4-mini-low"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 gpt-4.1"
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 o3-medium"
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_low_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_o3-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 o3-low"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-oss-120b_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-oss-120b_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 gpt-oss-120b"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-5_2025-08-07_medium_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 gpt-5-medium"
python scripts/merge_traces.py --input 'traces/iter1_DeepSeek-R1_1_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 deepseek-r1"
python scripts/merge_traces.py --input 'traces/iter1_deepseek-ai_DeepSeek-V3-0324_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 deepseek-v3"
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4o_2024-11-20_capsule-*_corebench_hard_*_UPLOAD.json' --output traces/iter1_corebench_hard_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: iter1 gpt-4o"

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

echo "CoreBench completed!"
