#!/usr/bin/env python3
"""
Merge trace results into result_matrix_merged.csv

This script reads trace files from the new_traces folder and updates the
result_matrix_merged.csv with the new evaluation results.

The mapping is carefully defined to ensure alignment between:
- Trace file names → CSV row IDs (model identifiers)
- Task IDs in traces → CSV column IDs (question identifiers)
"""

import json
import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import sys

# ============================================================
# CONFIGURATION: Mapping from trace prefixes to CSV row prefixes
# ============================================================

# Maps trace file prefix to (agent_name, benchmark_name_column_prefix)
TRACE_PREFIX_TO_AGENT = {
    # SciCode traces
    'scicode_honey_': ('scicode_tool_calling_agent', 'scicode'),
    'scicode_sea_': ('scicode_zero_shot_agent', 'scicode'),

    # CoreBench traces
    'prop_': ('hal_generalist_agent', 'corebench_hard'),
    'iter1_': ('core_agent', 'corebench_hard'),

    # ScienceAgentBench traces
    'sab_mate_': ('sab_self_debug', 'scienceagentbench'),
    'sab_cow_': ('sab_self_debug', 'scienceagentbench'),
    'sab_husky_': ('sab_self_debug', 'scienceagentbench'),

    # ColBench traces
    'col_ivy_': ('colbench_example_agent', 'colbench_backend_programming'),
    'col_zuck_': ('colbench_text', 'colbench_backend_programming'),
}

# ============================================================
# CONFIGURATION: Mapping from trace model names to CSV model suffixes
# ============================================================

def normalize_model_name(trace_model_part: str, trace_filename: str = None, trace_data: dict = None) -> str:
    """
    Convert trace file model name to CSV row model name suffix.

    Uses trace_data config to get reasoning_effort when needed for proper mapping.

    Examples:
    - openai_gpt-4_1 → gpt_4_1_20250414
    - openai_gpt-4o_2024 → gpt_4o_20241120
    - openai_gpt-5_2025 → gpt_5_20250807
    - openai_gpt-5_medium → gpt_5_20250807_medium
    - openai_gpt-5-mini_2025 → gpt_5_mini_20250807
    - openai_o3_2025 → o3_20250416
    - openai_o3_medium → o3_20250416_medium
    - openai_o3_low → o3_20250416_low
    - openai_o3-mini_2025 → o3_mini_20250131_high
    - openai_o3-mini_high → o3_mini_20250131_high
    - openai_o4-mini_2025-04-16_high → o4_mini_20250416_high
    - openai_o4-mini_2025-04-16_low → o4_mini_20250416_low
    - openai_o4-mini_high → o4_mini_20250416_high
    - openai_o4-mini_low → o4_mini_20250416_low
    - openai_gpt-oss-120b → oss_120b
    - DeepSeek-R1 → deepseek_r1
    - deepseek-ai_DeepSeek-V3 → deepseek_v3
    """
    model = trace_model_part

    # Get reasoning_effort from trace config if available
    reasoning_effort = None
    if trace_data:
        reasoning_effort = trace_data.get('config', {}).get('agent_args', {}).get('reasoning_effort')

    # Handle specific patterns
    model_mappings = {
        # GPT-4.1
        'openai_gpt-4_1': 'gpt_4_1_20250414',

        # GPT-4o
        'openai_gpt-4o_2024': 'gpt_4o_20241120',
        'openai_gpt-4o': 'gpt_4o_20241120',

        # GPT-5
        'openai_gpt-5_2025': 'gpt_5_20250807',
        'openai_gpt-5_medium': 'gpt_5_20250807_medium',
        'openai_gpt-5-mini_2025': 'gpt_5_mini_20250807',
        'openai_gpt-5-mini': 'gpt_5_mini_20250807',

        # O3
        'openai_o3_2025': 'o3_20250416',
        'openai_o3_medium': 'o3_20250416_medium',
        'openai_o3_low': 'o3_20250416_low',
        'openai_o3-mini_2025': 'o3_mini_20250131_high',
        'openai_o3-mini_high': 'o3_mini_20250131_high',
        'openai_o3_2025-04-16_low': 'o3_20250416_low',
        'openai_o3_2025-04-16_medium': 'o3_20250416_medium',

        # O4-mini
        'openai_o4-mini_2025-04-16_high': 'o4_mini_20250416_high',
        'openai_o4-mini_2025-04-16_low': 'o4_mini_20250416_low',
        'openai_o4-mini_high': 'o4_mini_20250416_high',
        'openai_o4-mini_low': 'o4_mini_20250416_low',

        # OSS
        'openai_gpt-oss-120b': 'oss_120b',

        # DeepSeek
        'DeepSeek-R1': 'deepseek_r1',
        'deepseek-ai_DeepSeek-V3': 'deepseek_v3',
    }

    # Special handling for models that need reasoning_effort suffix from config
    # These are trace files where filename doesn't include the effort but CSV does
    if reasoning_effort and model in model_mappings:
        base_model = model_mappings[model]
        # Check if this model needs effort suffix added
        if model == 'openai_gpt-5_2025' and reasoning_effort == 'medium':
            return 'gpt_5_20250807_medium'
        if model == 'openai_o3_2025' and reasoning_effort == 'medium':
            return 'o3_20250416_medium'

    if model in model_mappings:
        return model_mappings[model]

    # Fallback: try to construct the name
    print(f"WARNING: Unknown model pattern: {model}")
    return model.replace('-', '_').replace('.', '_').lower()


def extract_trace_info(trace_filename: str) -> Tuple[str, str, str]:
    """
    Extract agent prefix, model name, and any suffix from trace filename.

    Returns: (agent_prefix, model_part, suffix)
    """
    # Remove .json extension and _WITH_DIALOGUES suffix
    name = trace_filename.replace('.json', '').replace('_WITH_DIALOGUES', '')

    # Find which prefix matches
    for prefix in sorted(TRACE_PREFIX_TO_AGENT.keys(), key=len, reverse=True):
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            return prefix, remainder, ''

    return '', name, ''


def get_csv_row_id(trace_filename: str, trace_data: dict = None) -> str:
    """
    Convert trace filename to CSV row ID.

    Example: scicode_honey_openai_gpt-4_1.json → scicode_tool_calling_agent:gpt_4_1_20250414

    If trace_data is provided, uses config to get reasoning_effort for proper model name mapping.
    """
    prefix, model_part, _ = extract_trace_info(trace_filename)

    if prefix not in TRACE_PREFIX_TO_AGENT:
        print(f"WARNING: Unknown trace prefix: {prefix} from {trace_filename}")
        return None

    agent_name, _ = TRACE_PREFIX_TO_AGENT[prefix]
    model_suffix = normalize_model_name(model_part, trace_filename, trace_data)

    return f"{agent_name}:{model_suffix}"


def get_column_prefix(trace_filename: str) -> str:
    """Get the column prefix for the benchmark from trace filename."""
    prefix, _, _ = extract_trace_info(trace_filename)

    if prefix not in TRACE_PREFIX_TO_AGENT:
        return None

    _, col_prefix = TRACE_PREFIX_TO_AGENT[prefix]
    return col_prefix


def task_id_to_column(task_id: str, col_prefix: str) -> str:
    """
    Convert task ID from trace to CSV column name.

    Examples:
    - "2" with prefix "scicode" → "scicode.2"
    - "capsule-1624349" with prefix "corebench_hard" → "corebench_hard.capsule-1624349"
    """
    return f"{col_prefix}.{task_id}"


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load and parse a trace JSON file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def get_binary_result(trace_data: Dict, task_id: str) -> int:
    """
    Get binary result (0 or 1) for a task from trace data.

    Success = 1 if task is in successful_tasks
    Fail = 0 if task is in failed_tasks

    If task appears in both (which can happen), successful takes precedence.
    """
    results = trace_data.get('results', {})
    successful = set(results.get('successful_tasks', []))
    failed = set(results.get('failed_tasks', []))

    # Convert to string for comparison
    task_str = str(task_id)

    if task_str in successful:
        return 1
    elif task_str in failed:
        return 0
    else:
        # Task not found in either - return None to indicate no update
        return None


def main():
    # Paths
    traces_dir = Path('/home/v-tatruong/hal/agent-debug/new_traces')
    csv_path = Path('/home/v-tatruong/hal/reeval-multi/hal/data-hal/result_matrix_merged.csv')
    output_csv_path = Path('/home/v-tatruong/hal/agent-debug/result_matrix_newrun.csv')

    # Load the CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"CSV shape: {df.shape}")
    print(f"Rows (models): {len(df)}")
    print(f"Columns (questions): {len(df.columns)}")

    # Get all trace files
    trace_files = sorted([f.name for f in traces_dir.glob('*.json')])
    print(f"\nFound {len(trace_files)} trace files to process")

    # Track changes
    changes = []
    total_updates = 0
    successful_mappings = 0
    failed_mappings = []

    # Process each trace file
    for trace_file in trace_files:
        trace_path = traces_dir / trace_file

        # Load trace data FIRST (needed for reasoning_effort in model name mapping)
        try:
            trace_data = load_trace(str(trace_path))
        except Exception as e:
            failed_mappings.append((trace_file, f"Failed to load: {e}"))
            continue

        # Get CSV row ID (pass trace_data for reasoning_effort)
        row_id = get_csv_row_id(trace_file, trace_data)
        if row_id is None:
            failed_mappings.append((trace_file, "Could not determine row ID"))
            continue

        # Check if row exists in CSV
        if row_id not in df.index:
            # Try fallback: remove reasoning effort suffix (_high, _low, _medium) and try base model
            fallback_row_id = None
            for suffix in ['_high', '_low', '_medium']:
                if row_id.endswith(suffix):
                    base_row_id = row_id[:-len(suffix)]
                    if base_row_id in df.index:
                        fallback_row_id = base_row_id
                        print(f"  NOTE: Falling back from '{row_id}' to '{fallback_row_id}' (base model without suffix)")
                        break

            if fallback_row_id:
                row_id = fallback_row_id
            else:
                failed_mappings.append((trace_file, f"Row '{row_id}' not found in CSV"))
                continue

        # Get column prefix
        col_prefix = get_column_prefix(trace_file)
        if col_prefix is None:
            failed_mappings.append((trace_file, "Could not determine column prefix"))
            continue

        # Get all tasks from trace
        results = trace_data.get('results', {})
        all_tasks = set(results.get('successful_tasks', [])) | set(results.get('failed_tasks', []))

        if not all_tasks:
            failed_mappings.append((trace_file, "No tasks found in trace"))
            continue

        successful_mappings += 1
        file_updates = 0

        # Process each task
        for task_id in all_tasks:
            col_name = task_id_to_column(task_id, col_prefix)

            # Check if column exists
            if col_name not in df.columns:
                # This is expected for tasks not in the benchmark subset
                continue

            # Get new value
            new_value = get_binary_result(trace_data, task_id)
            if new_value is None:
                continue

            # Get old value
            old_value = df.loc[row_id, col_name]

            # Convert to comparable types
            try:
                old_val_int = int(old_value) if pd.notna(old_value) else None
            except (ValueError, TypeError):
                old_val_int = None

            # Update if different
            if old_val_int != new_value:
                changes.append({
                    'trace_file': trace_file,
                    'row_id': row_id,
                    'column': col_name,
                    'old_value': old_val_int,
                    'new_value': new_value
                })
                df.loc[row_id, col_name] = new_value
                file_updates += 1
                total_updates += 1

        print(f"  Processed {trace_file}: {row_id} → {file_updates} updates")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTrace files processed successfully: {successful_mappings}/{len(trace_files)}")
    print(f"Total values changed: {total_updates}")

    if failed_mappings:
        print(f"\nFailed mappings ({len(failed_mappings)}):")
        for trace_file, reason in failed_mappings:
            print(f"  - {trace_file}: {reason}")

    if changes:
        print(f"\n{'='*60}")
        print("DETAILED CHANGES")
        print("="*60)

        # Group by trace file
        changes_by_file = {}
        for c in changes:
            if c['trace_file'] not in changes_by_file:
                changes_by_file[c['trace_file']] = []
            changes_by_file[c['trace_file']].append(c)

        for trace_file, file_changes in changes_by_file.items():
            print(f"\n{trace_file}:")
            print(f"  Row: {file_changes[0]['row_id']}")
            print(f"  Changes ({len(file_changes)}):")
            for c in file_changes[:10]:  # Show first 10 changes per file
                old = 'N/A' if c['old_value'] is None else c['old_value']
                print(f"    {c['column']}: {old} → {c['new_value']}")
            if len(file_changes) > 10:
                print(f"    ... and {len(file_changes) - 10} more changes")

    # Save the updated CSV
    print(f"\nSaving updated CSV to {output_csv_path}...")
    df.to_csv(output_csv_path)
    print("Done!")

    # Return summary for programmatic use
    return {
        'total_traces': len(trace_files),
        'successful_mappings': successful_mappings,
        'failed_mappings': len(failed_mappings),
        'total_changes': total_updates,
        'changes': changes
    }


if __name__ == '__main__':
    result = main()
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total values changed: {result['total_changes']}")
