#!/bin/bash
# FINAL MERGE + WEAVE + RUBRIC COMMANDS
# Based on actual trace files that exist

# ============================================================
# SCICODE
# ============================================================

# SCICODE_HONEY (4 models)
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force

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

# SCICODE_LADY (10 models)
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5_2025-08-07_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-5_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/scicode_lady_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/scicode_lady_openai_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_DeepSeek-R1_1_*_UPLOAD.json' --output traces/scicode_lady_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_lady__scicode_lady_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_lady_scicode \
    --prefix scicode_lady_openai_gpt-4_1 \
    --prefix scicode_lady_openai_o3_2025 \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_high \
    --prefix scicode_lady_openai_o4-mini_2025-04-16_low \
    --prefix scicode_lady_openai_gpt-5_2025 \
    --prefix scicode_lady_openai_gpt-5-mini_2025 \
    --prefix scicode_lady_openai_gpt-4o_2024 \
    --prefix scicode_lady_openai_o3-mini_2025 \
    --prefix scicode_lady_DeepSeek-R1 \
    --prefix scicode_lady_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/scicode_lady_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_gpt-5_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_openai_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_lady_deepseek-v3_MERGED_UPLOAD.json

# ============================================================
# COREBENCH
# ============================================================

# PROP (10 models)
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/prop_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/prop_openai_o3_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-5_2025-08-07_medium_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4o_2024-11-20_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-oss-120b_1_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-oss-120b_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_DeepSeek-R1_1_capsule-*_UPLOAD.json' --output traces/prop_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_deepseek-ai_DeepSeek-V3-0324_capsule-*_UPLOAD.json' --output traces/prop_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025-04-16_medium \
    --prefix prop_openai_o3_2025-04-16_low \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --prefix prop_openai_gpt-5_2025 \
    --prefix prop_openai_gpt-4o_2024 \
    --prefix prop_openai_gpt-oss-120b \
    --prefix prop_DeepSeek-R1 \
    --prefix prop_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_low_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-oss-120b_MERGED_UPLOAD.json \
    --merge-input traces/prop_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/prop_deepseek-v3_MERGED_UPLOAD.json

# ITER1 (4 models)
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force

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

# ============================================================
# SAB
# ============================================================

# SAB_MATE (6 models)
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_DeepSeek-R1_1_*_UPLOAD.json' --output traces/sab_mate_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_gpt-5-mini_2025 \
    --prefix sab_mate_openai_gpt-4o_2024 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --prefix sab_mate_DeepSeek-R1 \
    --prefix sab_mate_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json

# SAB_COW (4 models)
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/sab_cow_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/sab_cow_openai_o4-mini_low_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --prefix sab_cow_openai_o4-mini_2025-04-16_high \
    --prefix sab_cow_openai_o4-mini_2025-04-16_low \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o4-mini_low_MERGED_UPLOAD.json

# SAB_HUSKY (1 model)
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-4_1 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json

# ============================================================
# COLBENCH
# ============================================================

# COL_IVY (9 models)
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy_gpt-4_1 \
    --prefix col_ivy_o3_2025 \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --prefix col_ivy_gpt-5_2025 \
    --prefix col_ivy_gpt-5-mini_2025 \
    --prefix col_ivy_gpt-4o_2024 \
    --prefix col_ivy_o3-mini_2025 \
    --prefix col_ivy_DeepSeek-R1 \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json

# COL_ZUCK (3 models)
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1-2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3-2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o3_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini-2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck_gpt-4_1 \
    --prefix col_zuck_o3-2025 \
    --prefix col_zuck_o4-mini-2025 \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_low_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json

echo "============================================================"
echo "MERGE + WEAVE COMPLETE!"
echo "============================================================"
echo ""
echo "Now run RUBRIC EVALUATION - see RUBRIC_COMMANDS.sh"
