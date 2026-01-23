#!/usr/bin/env python3
"""
Generate Binary Response Matrix from Benchmark Results

This script traverses benchmark results for a given prefix and generates a binary
response matrix (resmat) with:
- Columns: benchmark.taskid
- Rows: agent name (model + agent_type)
- Values: 0 (fail) or 1 (success)

Benchmark Scoring Formats:
- SciCode: results.successful_tasks / failed_tasks lists
- ScienceAgentBench: raw_eval_results.eval_result[task_id].success_rate (0/1)
- CoreBench: results.successful_tasks / failed_tasks lists
- ColBench: raw_eval_results is a list of scores [0-1], needs threshold

Usage:
    python scripts/generate_resmat.py --prefix moon4 --output results/resmat_moon4.csv
    python scripts/generate_resmat.py --prefix moon4 --colbench-threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


# Configuration - Tunable thresholds
THRESHOLDS = {
    'colbench': 0.5,  # Score >= threshold is success (tunable)
    'scienceagentbench': 0.5,  # success_rate >= threshold (usually 0 or 1)
}

# Benchmark directories
RESULTS_ROOT = '/Data/home/v-qizhengli/hal_runs/v-qizhengli/agent-debug/results'
BENCHMARK_DIRS = {
    'scicode': 'scicode',
    'scienceagentbench': 'scienceagentbench',
    'corebench': 'corebench_hard',
    'colbench': 'colbench_backend_programming',
}


def find_trace_files(prefix: str, results_root: str = None) -> Dict[str, List[Path]]:
    """Find all UPLOAD.json trace files matching the prefix."""
    if results_root is None:
        results_root = RESULTS_ROOT
    traces = {bm: [] for bm in BENCHMARK_DIRS}

    for benchmark, subdir in BENCHMARK_DIRS.items():
        benchmark_path = Path(results_root) / subdir
        if not benchmark_path.exists():
            print(f"[WARN] Benchmark directory not found: {benchmark_path}")
            continue

        # Find all matching trace files
        for run_dir in benchmark_path.iterdir():
            if not run_dir.is_dir():
                continue
            # Check if run matches prefix (e.g., scicode_moon4_*)
            if f"_{prefix}_" not in run_dir.name and not run_dir.name.startswith(f"{prefix}_"):
                continue

            # Find UPLOAD.json files
            for json_file in run_dir.glob("*_UPLOAD.json"):
                traces[benchmark].append(json_file)

    return traces


def extract_agent_name(trace_path: Path, config: dict) -> str:
    """Extract agent name from trace config or path."""
    # Try to get from config
    agent_name = config.get('agent_name', '')
    agent_args = config.get('agent_args', {})
    model_name = agent_args.get('model_name', '')

    # Parse model name (e.g., "openai/gpt-4.1-2025-04-14" -> "gpt-4.1")
    if model_name:
        model_short = model_name.split('/')[-1]
        # Remove date suffix if present
        parts = model_short.split('-')
        if len(parts) > 2 and parts[-1].isdigit() and len(parts[-1]) == 2:
            model_short = '-'.join(parts[:-3]) if len(parts) > 3 else parts[0]
    else:
        # Fallback: extract from path
        name = trace_path.parent.name
        # Pattern: benchmark_prefix_model_agent_timestamp
        parts = name.split('_')
        model_short = parts[2] if len(parts) > 2 else 'unknown'

    # Get agent type from path
    path_name = trace_path.parent.name
    if 'generalist' in path_name:
        agent_type = 'generalist'
    elif 'scicode_tool_calling_agent' in path_name:
        agent_type = 'scicode_agent'
    elif 'sab_example_agent' in path_name:
        agent_type = 'sab_agent'
    elif 'core_agent' in path_name:
        agent_type = 'core_agent'
    elif 'colbench_example_agent' in path_name:
        agent_type = 'colbench_agent'
    else:
        agent_type = 'agent'

    # Construct readable name
    return f"{model_short}_{agent_type}"


def parse_scicode(data: dict) -> Dict[str, int]:
    """Parse SciCode results to binary scores per task."""
    results = data.get('results', {})
    successful = set(results.get('successful_tasks', []))
    failed = set(results.get('failed_tasks', []))

    scores = {}
    for task_id in successful:
        scores[str(task_id)] = 1
    for task_id in failed:
        scores[str(task_id)] = 0

    return scores


def parse_scienceagentbench(data: dict, threshold: float = 0.5) -> Dict[str, int]:
    """Parse ScienceAgentBench results to binary scores per task."""
    raw_eval = data.get('raw_eval_results', {})
    eval_result = raw_eval.get('eval_result', {})

    scores = {}
    for task_id, task_data in eval_result.items():
        if isinstance(task_data, dict):
            success_rate = task_data.get('success_rate', 0)
            scores[str(task_id)] = 1 if success_rate >= threshold else 0

    return scores


def parse_corebench(data: dict) -> Dict[str, int]:
    """Parse CoreBench results to binary scores per task."""
    results = data.get('results', {})
    successful = set(results.get('successful_tasks', []))
    failed = set(results.get('failed_tasks', []))

    scores = {}
    for task_id in successful:
        scores[str(task_id)] = 1
    for task_id in failed:
        scores[str(task_id)] = 0

    return scores


def parse_colbench(data: dict, threshold: float = 0.5) -> Dict[str, int]:
    """Parse ColBench results to binary scores per task."""
    raw_eval = data.get('raw_eval_results', [])

    scores = {}
    if isinstance(raw_eval, list):
        for idx, score in enumerate(raw_eval):
            task_id = str(idx + 1)  # 1-indexed
            scores[task_id] = 1 if score >= threshold else 0

    return scores


def parse_trace(trace_path: Path, benchmark: str, thresholds: dict) -> Tuple[str, Dict[str, int]]:
    """Parse a single trace file and return agent name and scores."""
    with open(trace_path) as f:
        data = json.load(f)

    config = data.get('config', {})
    agent_name = extract_agent_name(trace_path, config)

    # Parse based on benchmark type
    if benchmark == 'scicode':
        scores = parse_scicode(data)
    elif benchmark == 'scienceagentbench':
        scores = parse_scienceagentbench(data, thresholds.get('scienceagentbench', 0.5))
    elif benchmark == 'corebench':
        scores = parse_corebench(data)
    elif benchmark == 'colbench':
        scores = parse_colbench(data, thresholds.get('colbench', 0.5))
    else:
        print(f"[WARN] Unknown benchmark: {benchmark}")
        scores = {}

    return agent_name, scores


def build_resmat(traces: Dict[str, List[Path]], thresholds: dict) -> pd.DataFrame:
    """Build the response matrix from all traces."""
    # Collect all data points
    all_data = {}  # {agent_name: {benchmark.task_id: score}}
    all_columns = set()  # benchmark.task_id

    for benchmark, trace_files in traces.items():
        print(f"\n[{benchmark.upper()}] Processing {len(trace_files)} trace files...")

        for trace_path in trace_files:
            try:
                agent_name, scores = parse_trace(trace_path, benchmark, thresholds)

                if agent_name not in all_data:
                    all_data[agent_name] = {}

                for task_id, score in scores.items():
                    col_name = f"{benchmark}.{task_id}"
                    all_data[agent_name][col_name] = score
                    all_columns.add(col_name)

                print(f"  [{agent_name}] {len(scores)} tasks, {sum(scores.values())} passed")

            except Exception as e:
                print(f"  [ERROR] Failed to parse {trace_path.name}: {e}")

    # Build DataFrame
    if not all_data:
        print("\n[ERROR] No data collected!")
        return pd.DataFrame()

    # Sort columns by benchmark, then task_id
    def sort_key(col):
        parts = col.split('.')
        benchmark = parts[0]
        task_id = parts[1] if len(parts) > 1 else ''
        # Try to sort numerically for task_id
        try:
            task_num = int(task_id.replace('capsule-', ''))
        except:
            task_num = task_id
        return (benchmark, task_num)

    sorted_columns = sorted(all_columns, key=sort_key)

    # Create DataFrame with agents as rows, tasks as columns
    df = pd.DataFrame(index=sorted(all_data.keys()), columns=sorted_columns)

    for agent_name, scores in all_data.items():
        for col, score in scores.items():
            df.loc[agent_name, col] = score

    # Fill NaN with -1 (indicates task not run)
    df = df.fillna(-1).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate binary response matrix from benchmark results')
    parser.add_argument('--prefix', required=True, help='Run prefix to match (e.g., moon4)')
    parser.add_argument('--output', default=None, help='Output CSV file path')
    parser.add_argument('--results-root', default=None, help='Root directory for results')
    parser.add_argument('--colbench-threshold', type=float, default=0.5,
                        help='Threshold for ColBench binary conversion (default: 0.5)')
    parser.add_argument('--sab-threshold', type=float, default=0.5,
                        help='Threshold for ScienceAgentBench binary conversion (default: 0.5)')
    parser.add_argument('--benchmarks', nargs='+', default=list(BENCHMARK_DIRS.keys()),
                        help='Benchmarks to include (default: all)')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')

    args = parser.parse_args()

    results_root = args.results_root if args.results_root else RESULTS_ROOT

    # Set thresholds
    thresholds = {
        'colbench': args.colbench_threshold,
        'scienceagentbench': args.sab_threshold,
    }

    print(f"=== Response Matrix Generator ===")
    print(f"Prefix: {args.prefix}")
    print(f"Results root: {results_root}")
    print(f"ColBench threshold: {thresholds['colbench']}")
    print(f"SAB threshold: {thresholds['scienceagentbench']}")
    print(f"Benchmarks: {args.benchmarks}")

    # Filter benchmarks
    filtered_dirs = {k: v for k, v in BENCHMARK_DIRS.items() if k in args.benchmarks}

    # Find trace files
    print(f"\n=== Finding Traces ===")
    traces = find_trace_files(args.prefix, results_root)

    # Filter to selected benchmarks
    traces = {k: v for k, v in traces.items() if k in args.benchmarks}

    total_traces = sum(len(v) for v in traces.values())
    print(f"Found {total_traces} trace files total")
    for bm, files in traces.items():
        print(f"  {bm}: {len(files)} files")

    if total_traces == 0:
        print("\n[ERROR] No trace files found!")
        sys.exit(1)

    # Build response matrix
    print(f"\n=== Building Response Matrix ===")
    df = build_resmat(traces, thresholds)

    if df.empty:
        print("[ERROR] Empty response matrix!")
        sys.exit(1)

    # Output path
    if args.output is None:
        output_path = Path('/Data/home/v-qizhengli/workspace/agent-debug/results') / f'resmat_{args.prefix}.csv'
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path)
    print(f"\n=== Output ===")
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape[0]} agents x {df.shape[1]} tasks")

    # Summary statistics
    if args.summary or True:  # Always print summary
        print(f"\n=== Summary Statistics ===")

        # Per-benchmark stats
        for benchmark in args.benchmarks:
            bm_cols = [c for c in df.columns if c.startswith(f"{benchmark}.")]
            if bm_cols:
                bm_df = df[bm_cols]
                valid_mask = bm_df >= 0
                total_valid = valid_mask.sum().sum()
                total_success = (bm_df == 1).sum().sum()
                success_rate = total_success / total_valid * 100 if total_valid > 0 else 0
                print(f"\n{benchmark.upper()}:")
                print(f"  Tasks: {len(bm_cols)}")
                print(f"  Valid entries: {total_valid}")
                print(f"  Success: {total_success} ({success_rate:.1f}%)")

                # Per-agent success rate for this benchmark
                print(f"  Per-agent success rates:")
                for agent in df.index:
                    agent_valid = valid_mask.loc[agent].sum()
                    agent_success = (bm_df.loc[agent] == 1).sum()
                    agent_rate = agent_success / agent_valid * 100 if agent_valid > 0 else 0
                    print(f"    {agent}: {agent_success}/{agent_valid} ({agent_rate:.1f}%)")

        # Overall stats
        print(f"\n=== Overall ===")
        valid_mask = df >= 0
        total_valid = valid_mask.sum().sum()
        total_success = (df == 1).sum().sum()
        overall_rate = total_success / total_valid * 100 if total_valid > 0 else 0
        print(f"Total valid entries: {total_valid}")
        print(f"Total success: {total_success} ({overall_rate:.1f}%)")


if __name__ == '__main__':
    main()
