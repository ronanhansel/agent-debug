#!/bin/bash
# RUBRIC EVALUATION - Run after FINAL_COMMANDS.sh completes

# ============================================================
# RUBRIC EVALUATION
# ============================================================

# SCICODE (14 traces total: 4 from honey + 10 from lady)
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
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

# COREBENCH (14 traces total: 10 from prop + 4 from iter1)
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

# SAB (11 traces total: 6 from mate + 4 from cow + 1 from husky)
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_openai_gpt-5_2025.json \
    --trace-file traces/sab_mate_openai_gpt-5-mini_2025.json \
    --trace-file traces/sab_mate_openai_gpt-4o_2024.json \
    --trace-file traces/sab_mate_openai_o3-mini_2025.json \
    --trace-file traces/sab_mate_DeepSeek-R1.json \
    --trace-file traces/sab_mate_deepseek-ai_DeepSeek-V3.json \
    --trace-file traces/sab_cow_openai_gpt-4_1.json \
    --trace-file traces/sab_cow_openai_o3_2025.json \
    --trace-file traces/sab_cow_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/sab_cow_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000

# COLBENCH (12 traces total: 9 from ivy + 3 from zuck)
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
    --trace-file traces/col_zuck_o3-2025.json \
    --trace-file traces/col_zuck_o4-mini-2025.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000
