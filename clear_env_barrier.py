#!/usr/bin/env python3
"""
Clear environmental barrier flags for cells that were replaced in the new trace run.

Logic:
- Load the list of cells that were changed in merge_traces_to_matrix.py
- For each of those cells in rubrics_matrix_environmentalbarrier.csv:
  - If the value is 1 (environmental barrier flagged), set it to 0
- Save the updated matrix as rubrics_matrix_environmentalbarrier_newrun.csv
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# Re-use the mapping logic from merge_traces_to_matrix.py
# ============================================================
# CONFIGURATION: Mapping from trace prefixes to CSV row prefixes
# ============================================================

TRACE_PREFIX_TO_AGENT = {
    'scicode_honey_': ('scicode_tool_calling_agent', 'scicode'),
    'scicode_sea_': ('scicode_zero_shot_agent', 'scicode'),
    'prop_': ('hal_generalist_agent', 'corebench_hard'),
    'iter1_': ('core_agent', 'corebench_hard'),
    'sab_mate_': ('sab_self_debug', 'scienceagentbench'),
    'sab_cow_': ('sab_self_debug', 'scienceagentbench'),
    'sab_husky_': ('sab_self_debug', 'scienceagentbench'),
    'col_ivy_': ('colbench_example_agent', 'colbench_backend_programming'),
    'col_zuck_': ('colbench_text', 'colbench_backend_programming'),
}


def normalize_model_name(trace_model_part: str) -> str:
    """Convert trace file model name to CSV row model name suffix."""
    model_mappings = {
        'openai_gpt-4_1': 'gpt_4_1_20250414',
        'openai_gpt-4o_2024': 'gpt_4o_20241120',
        'openai_gpt-4o': 'gpt_4o_20241120',
        'openai_gpt-5_2025': 'gpt_5_20250807',
        'openai_gpt-5_medium': 'gpt_5_20250807_medium',
        'openai_gpt-5-mini_2025': 'gpt_5_mini_20250807',
        'openai_gpt-5-mini': 'gpt_5_mini_20250807',
        'openai_o3_2025': 'o3_20250416',
        'openai_o3_medium': 'o3_20250416_medium',
        'openai_o3_low': 'o3_20250416_low',
        'openai_o3-mini_2025': 'o3_mini_20250131_high',
        'openai_o3-mini_high': 'o3_mini_20250131_high',
        'openai_o3_2025-04-16_low': 'o3_20250416_low',
        'openai_o3_2025-04-16_medium': 'o3_20250416_medium',
        'openai_o4-mini_2025-04-16_high': 'o4_mini_20250416_high',
        'openai_o4-mini_2025-04-16_low': 'o4_mini_20250416_low',
        'openai_o4-mini_high': 'o4_mini_20250416_high',
        'openai_o4-mini_low': 'o4_mini_20250416_low',
        'openai_gpt-oss-120b': 'oss_120b',
        'DeepSeek-R1': 'deepseek_r1',
        'deepseek-ai_DeepSeek-V3': 'deepseek_v3',
    }

    if trace_model_part in model_mappings:
        return model_mappings[trace_model_part]
    return trace_model_part.replace('-', '_').replace('.', '_').lower()


def extract_trace_info(trace_filename: str) -> Tuple[str, str, str]:
    """Extract agent prefix, model name from trace filename."""
    name = trace_filename.replace('.json', '').replace('_WITH_DIALOGUES', '')

    for prefix in sorted(TRACE_PREFIX_TO_AGENT.keys(), key=len, reverse=True):
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            return prefix, remainder, ''

    return '', name, ''


def get_csv_row_id(trace_filename: str) -> str:
    """Convert trace filename to CSV row ID."""
    prefix, model_part, _ = extract_trace_info(trace_filename)

    if prefix not in TRACE_PREFIX_TO_AGENT:
        return None

    agent_name, _ = TRACE_PREFIX_TO_AGENT[prefix]
    model_suffix = normalize_model_name(model_part)

    return f"{agent_name}:{model_suffix}"


def get_column_prefix(trace_filename: str) -> str:
    """Get the column prefix for the benchmark from trace filename."""
    prefix, _, _ = extract_trace_info(trace_filename)

    if prefix not in TRACE_PREFIX_TO_AGENT:
        return None

    _, col_prefix = TRACE_PREFIX_TO_AGENT[prefix]
    return col_prefix


def task_id_to_column(task_id: str, col_prefix: str) -> str:
    """Convert task ID from trace to CSV column name."""
    return f"{col_prefix}.{task_id}"


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load and parse a trace JSON file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def get_all_replaced_cells(traces_dir: Path) -> List[Tuple[str, str]]:
    """
    Get list of all (row_id, column) pairs that were replaced in the new trace run.

    Returns list of (row_id, column) tuples for cells that have trace data.
    """
    replaced_cells = []

    trace_files = sorted([f.name for f in traces_dir.glob('*.json')])

    for trace_file in trace_files:
        trace_path = traces_dir / trace_file

        # Get CSV row ID
        row_id = get_csv_row_id(trace_file)
        if row_id is None:
            continue

        # Get column prefix
        col_prefix = get_column_prefix(trace_file)
        if col_prefix is None:
            continue

        # Load trace data
        try:
            trace_data = load_trace(str(trace_path))
        except Exception as e:
            continue

        # Get all tasks from trace
        results = trace_data.get('results', {})
        all_tasks = set(results.get('successful_tasks', [])) | set(results.get('failed_tasks', []))

        if not all_tasks:
            continue

        # Add all (row_id, column) pairs
        for task_id in all_tasks:
            col_name = task_id_to_column(task_id, col_prefix)
            replaced_cells.append((row_id, col_name))

    return replaced_cells


def main():
    # Paths
    traces_dir = Path('/home/v-tatruong/hal/agent-debug/new_traces')
    env_barrier_csv = Path('/home/v-tatruong/hal/reeval-multi/hal/data-hal/rubrics_matrix_environmentalbarrier.csv')
    output_csv = Path('/home/v-tatruong/hal/agent-debug/rubrics_matrix_environmentalbarrier_newrun.csv')

    # Load environmental barrier matrix
    print(f"Loading environmental barrier matrix from {env_barrier_csv}...")
    df = pd.read_csv(env_barrier_csv, index_col=0)
    print(f"Matrix shape: {df.shape}")
    print(f"Rows (models): {len(df)}")
    print(f"Columns (questions): {len(df.columns)}")

    # Get all replaced cells from traces
    print(f"\nAnalyzing traces in {traces_dir}...")
    replaced_cells = get_all_replaced_cells(traces_dir)
    print(f"Found {len(replaced_cells)} cells that were replaced in new trace run")

    # Track changes
    cleared_cells = []
    not_found_cells = []
    already_zero_cells = []

    # Process each replaced cell
    for row_id, col_name in replaced_cells:
        # Check if row exists
        if row_id not in df.index:
            not_found_cells.append((row_id, col_name, "row not found"))
            continue

        # Check if column exists
        if col_name not in df.columns:
            not_found_cells.append((row_id, col_name, "column not found"))
            continue

        # Get current value
        current_value = df.loc[row_id, col_name]

        # Check if it's a 1 (environmental barrier flagged)
        try:
            val = float(current_value)
            if val == 1.0:
                # Clear the barrier
                df.loc[row_id, col_name] = 0.0
                cleared_cells.append((row_id, col_name))
            elif val == 0.0:
                already_zero_cells.append((row_id, col_name))
            # else: NaN or other value, skip
        except (ValueError, TypeError):
            # NaN or non-numeric, skip
            pass

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal cells from new trace run: {len(replaced_cells)}")
    print(f"Cells with row/column not found: {len(not_found_cells)}")
    print(f"Cells already 0 (no barrier): {len(already_zero_cells)}")
    print(f"Cells cleared (1 → 0): {len(cleared_cells)}")

    if cleared_cells:
        print(f"\n{'='*60}")
        print("ENVIRONMENTAL BARRIERS CLEARED")
        print("="*60)

        # Group by row
        cleared_by_row = {}
        for row_id, col_name in cleared_cells:
            if row_id not in cleared_by_row:
                cleared_by_row[row_id] = []
            cleared_by_row[row_id].append(col_name)

        for row_id, columns in sorted(cleared_by_row.items()):
            print(f"\n{row_id}:")
            for col in columns[:10]:
                print(f"  - {col}: 1 → 0")
            if len(columns) > 10:
                print(f"  ... and {len(columns) - 10} more")

    # Save the updated matrix
    print(f"\nSaving updated matrix to {output_csv}...")
    df.to_csv(output_csv)
    print("Done!")

    return {
        'total_replaced_cells': len(replaced_cells),
        'cleared_cells': len(cleared_cells),
        'already_zero': len(already_zero_cells),
        'not_found': len(not_found_cells),
        'cleared_details': cleared_cells
    }


if __name__ == '__main__':
    result = main()
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Environmental barriers cleared: {result['cleared_cells']}")
