#!/bin/bash
# Merge and fetch ColBench traces (ivy + zuck)
set -e

echo "=========================================="
echo "COLBENCH - col_ivy_colbench_backendprogramming"
echo "=========================================="

python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy gpt-4.1"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy o4-mini-low"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy o3-medium"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy o4-mini-high"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5_2025-08-07_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy gpt-5-medium"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4o_2024-11-20_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy gpt-4o"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5-mini_2025-08-07_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy gpt-5-mini"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3-mini_2025-01-31_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy o3-mini-high"
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_DeepSeek-R1_1_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_ivy_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_ivy deepseek-r1"

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

python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-4.1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck gpt-4.1"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o4-mini-low_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck o4-mini-low"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o3-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck o3-medium"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o4-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck o4-mini-high"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5_2025-08-07_medium_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-5-medium_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck gpt-5-medium"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4o_2024-11-20_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-4o_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck gpt-4o"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5-mini_2025-08-07_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck gpt-5-mini"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3-mini_2025-01-31_high_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_o3-mini-high_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck o3-mini-high"
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_DeepSeek-R1_1_*_colbench_backend_programming_*_UPLOAD.json' --output traces/col_zuck_deepseek-r1_MERGED_UPLOAD.json --force 2>/dev/null || echo "Pattern not found: col_zuck deepseek-r1"

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

echo "ColBench completed!"
