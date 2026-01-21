#!/bin/bash
# Merge and fetch SciCode traces (lady + honey)
set -e

echo "=========================================="
echo "SCICODE - scicode_lady_scicode"
echo "=========================================="

python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady o3-medium"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady o4-mini-low"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady o4-mini-high"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady gpt-4.1"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-5_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady gpt-5"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_DeepSeek-R1_1_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady deepseek-r1"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_deepseek-ai_DeepSeek-V3-0324_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady deepseek-v3"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5-mini_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady gpt-5-mini"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4o_2024-11-20_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady gpt-4o"
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3-mini_2025-01-31_high_*_scicode_*_UPLOAD.json' --output traces/scicode_lady_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_lady o3-mini-high"

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

python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey o3-medium"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey o4-mini-low"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey o4-mini-high"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey gpt-4.1"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-5_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey gpt-5"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_DeepSeek-R1_1_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey deepseek-r1"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_deepseek-ai_DeepSeek-V3-0324_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey deepseek-v3"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5-mini_2025-08-07_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey gpt-5-mini"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4o_2024-11-20_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey gpt-4o"
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3-mini_2025-01-31_high_*_scicode_*_UPLOAD.json' --output traces/scicode_honey_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: scicode_honey o3-mini-high"

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

echo "SciCode completed!"
