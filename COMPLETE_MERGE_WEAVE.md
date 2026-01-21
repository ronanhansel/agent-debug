# Complete Merge and Weave Commands - ALL MODELS

## SCICODE (10 models each)

### SciCode Honey - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5_2025-08-07_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-5_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_DeepSeek-R1_1_*_UPLOAD.json' --output traces/scicode_honey_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json --force
```

### SciCode Honey - Step 2: Extract from Weave
```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --prefix scicode_honey_openai_gpt-5_2025 \
    --prefix scicode_honey_openai_gpt-5-mini_2025 \
    --prefix scicode_honey_openai_gpt-4o_2024 \
    --prefix scicode_honey_openai_o3-mini_2025 \
    --prefix scicode_honey_DeepSeek-R1 \
    --prefix scicode_honey_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_gpt-5_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_deepseek-v3_MERGED_UPLOAD.json
```

### SciCode Lady - Step 1: Merge Local
```bash
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
```

### SciCode Lady - Step 2: Extract from Weave
```bash
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
```

---

## COREBENCH (10 models each)

### CoreBench Prop - Step 1: Merge Local
```bash
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
```

### CoreBench Prop - Step 2: Extract from Weave
```bash
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
```

### CoreBench Iter1 - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force
```

### CoreBench Iter1 - Step 2: Extract from Weave
```bash
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
```

---

## SCIENCEAGENTBENCH

### SAB Mate - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json --force
```

### SAB Mate - Step 2: Extract from Weave
```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json
```

### SAB Cow - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json --force
```

### SAB Cow - Step 2: Extract from Weave
```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json
```

### SAB Husky - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json --force
```

### SAB Husky - Step 2: Extract from Weave
```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-4_1 \
    --prefix sab_husky_openai_o3-mini_2025 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_husky_openai_o3-mini_high_MERGED_UPLOAD.json
```

---

## COLBENCH (9 models each)

### ColBench Ivy - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json --force
```

### ColBench Ivy - Step 2: Extract from Weave
```bash
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
```

### ColBench Zuck - Step 1: Merge Local
```bash
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_zuck_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_zuck_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_zuck_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_zuck_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_zuck_DeepSeek-R1_MERGED_UPLOAD.json --force
```

### ColBench Zuck - Step 2: Extract from Weave
```bash
python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck_gpt-4_1 \
    --prefix col_zuck_o3_2025 \
    --prefix col_zuck_o4-mini_2025-04-16_high \
    --prefix col_zuck_o4-mini_2025-04-16_low \
    --prefix col_zuck_gpt-5_2025 \
    --prefix col_zuck_gpt-5-mini_2025 \
    --prefix col_zuck_gpt-4o_2024 \
    --prefix col_zuck_o3-mini_2025 \
    --prefix col_zuck_DeepSeek-R1 \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_DeepSeek-R1_MERGED_UPLOAD.json
```

---

## RUBRIC EVALUATION COMMANDS

### SciCode (8 trace files)
```bash
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/scicode_honey_openai_gpt-5_2025.json \
    --trace-file traces/scicode_honey_openai_gpt-5-mini_2025.json \
    --trace-file traces/scicode_honey_openai_gpt-4o_2024.json \
    --trace-file traces/scicode_honey_openai_o3-mini_2025.json \
    --trace-file traces/scicode_honey_DeepSeek-R1.json \
    --trace-file traces/scicode_honey_deepseek-ai_DeepSeek-V3.json \
    --trace-file traces/scicode_lady_openai_gpt-4_1.json \
    --trace-file traces/scicode_lady_openai_o3_2025.json \
    --trace-file traces/scicode_lady_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_lady_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/scicode_lady_openai_gpt-5_2025.json \
    --trace-file traces/scicode_lady_openai_gpt-5-mini_2025.json \
    --trace-file traces/scicode_lady_openai_gpt-4o_2024.json \
    --trace-file traces/scicode_lady_openai_o3-mini_2025.json \
    --trace-file traces/scicode_lady_DeepSeek-R1.json \
    --trace-file traces/scicode_lady_deepseek-ai_DeepSeek-V3.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### CoreBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/prop_openai_gpt-4_1.json \
    --trace-file traces/prop_openai_o3_2025-04-16_medium.json \
    --trace-file traces/prop_openai_o3_2025-04-16_low.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/prop_openai_gpt-5_2025.json \
    --trace-file traces/prop_openai_gpt-4o_2024.json \
    --trace-file traces/prop_openai_gpt-oss-120b.json \
    --trace-file traces/prop_DeepSeek-R1.json \
    --trace-file traces/prop_deepseek-ai_DeepSeek-V3.json \
    --trace-file traces/iter1_openai_gpt-4_1.json \
    --trace-file traces/iter1_openai_o3_2025.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### ScienceAgentBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_openai_gpt-5_2025.json \
    --trace-file traces/sab_mate_openai_o3-mini_2025.json \
    --trace-file traces/sab_cow_openai_gpt-4_1.json \
    --trace-file traces/sab_cow_openai_o3_2025.json \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --trace-file traces/sab_husky_openai_o3-mini_2025.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

### ColBench
```bash
python scripts/eval_rubric.py \
    --trace-file traces/col_ivy_gpt-4_1.json \
    --trace-file traces/col_ivy_o3_2025.json \
    --trace-file traces/col_ivy_o4-mini_2025-04-16_high.json \
    --trace-file traces/col_ivy_o4-mini_2025-04-16_low.json \
    --trace-file traces/col_ivy_gpt-5_2025.json \
    --trace-file traces/col_ivy_gpt-5-mini_2025.json \
    --trace-file traces/col_ivy_gpt-4o_2024.json \
    --trace-file traces/col_ivy_o3-mini_2025.json \
    --trace-file traces/col_ivy_DeepSeek-R1.json \
    --trace-file traces/col_zuck_gpt-4_1.json \
    --trace-file traces/col_zuck_o3_2025.json \
    --trace-file traces/col_zuck_o4-mini_2025-04-16_high.json \
    --trace-file traces/col_zuck_o4-mini_2025-04-16_low.json \
    --trace-file traces/col_zuck_gpt-5_2025.json \
    --trace-file traces/col_zuck_gpt-5-mini_2025.json \
    --trace-file traces/col_zuck_gpt-4o_2024.json \
    --trace-file traces/col_zuck_o3-mini_2025.json \
    --trace-file traces/col_zuck_DeepSeek-R1.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
```

---

## JUDGE AGGREGATION COMMANDS

### SciCode
```bash
python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --pattern "scicode_lady_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### CoreBench
```bash
python scripts/judge.py \
    --pattern "prop_openai_*" \
    --pattern "iter1_openai_*" \
    --rubric-dir rubrics_output/corebench \
    --output judge_output/corebench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### ScienceAgentBench
```bash
python scripts/judge.py \
    --pattern "sab_mate_*" \
    --pattern "sab_cow_*" \
    --pattern "sab_husky_*" \
    --rubric-dir rubrics_output/scienceagentbench \
    --output judge_output/scienceagentbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### ColBench
```bash
python scripts/judge.py \
    --pattern "col_ivy_*" \
    --pattern "col_zuck_*" \
    --rubric-dir rubrics_output/colbench \
    --output judge_output/colbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

---

## Summary

**Total Models:**
- SciCode: 10 models × 2 runs (honey + lady) = 20 trace files
- CoreBench: 10 models (prop) + 4 models (iter1) = 14 trace files
- SAB: 2 models (mate) + 2 models (cow) + 2 models (husky) = 6 trace files
- ColBench: 9 models × 2 runs (ivy + zuck) = 18 trace files

**Grand Total: 58 final trace files**
