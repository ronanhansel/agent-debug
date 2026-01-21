#!/bin/bash
# Merge and fetch ScienceAgentBench traces (mate + cow + husky)
set -e

echo "=========================================="
echo "SCIENCEAGENTBENCH - sab_mate_scienceagentbench"
echo "=========================================="

python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_mate_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_mate o3-mini-high"

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

python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_cow_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_cow o3-mini-high"

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

python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-5_2025-08-07_medium_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky gpt-5-medium"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_DeepSeek-R1_1_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky deepseek-r1"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_deepseek-ai_DeepSeek-V3-0324_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_deepseek-v3_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky deepseek-v3"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-5-mini_2025-08-07_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky gpt-5-mini"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4o_2024-11-20_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky gpt-4o"
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3-mini_2025-01-31_high_*_scienceagentbench_*_UPLOAD.json' --output traces/sab_husky_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: sab_husky o3-mini-high"

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

echo "ScienceAgentBench completed!"
