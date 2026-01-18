#!/usr/bin/env python3
"""
Docent-based rubric evaluation script (primary method).

Evaluates agent traces using benchmark-specific rubrics with:
- SQLite LLM response caching (no repeat API calls)
- Batch processing with retry logic
- Turn-by-turn conversation deduplication
- Support for multiple benchmarks via rubric templates

Usage:
    # SciCode: Evaluate Intrinsic Formation Errors
    OPENAI_BASE_URL="http://localhost:4000/v1" python scripts/eval_rubric.py \
        --trace-file traces/scicode_hal_generalist_agent_gpt4120250414_*.json \
        --rubric rubric_templates/scicode.txt \
        --rubric-model openai:gpt-4o \
        --failed-only -y

    # CoreBench: Evaluate Environmental Barriers
    OPENAI_BASE_URL="http://localhost:4000/v1" python scripts/eval_rubric.py \
        --trace-file traces/corebench_hard_hal_generalist_agentgpt41_*.json \
        --rubric rubric_templates/corebench.txt \
        --rubric-model openai:gpt-4o \
        --failed-only -y

    # Preview mode (stdout, limited tasks)
    python scripts/eval_rubric.py \
        --trace-file traces/scicode_*.json \
        --rubric rubric_templates/scicode.txt \
        --rubric-model openai:gpt-4o \
        --output-mode stdout \
        --max-tasks 3 -y

Output:
    CSV files go to rubrics_output/<rubric_name>/<trace_name>.csv
    Example: rubrics_output/scicode/scicode_hal_generalist_agent_gpt41.csv

See PIPELINE_README.md for full documentation.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubric_evaluator import cli as rubric_cli


def main():
    parser = argparse.ArgumentParser(
        description="Rubric evaluation using Docent. Input: trace file. Output: CSV with same name."
    )

    # Trace selection
    parser.add_argument(
        "--trace-file",
        type=str,
        action="append",
        required=True,
        dest="trace_files",
        help="Path to trace JSON file to evaluate (can be specified multiple times)",
    )

    # Rubric configuration
    parser.add_argument(
        "--rubric",
        type=str,
        help="Path to a single rubric .txt file (overrides --rubrics-dir)",
    )
    parser.add_argument(
        "--rubrics-dir",
        type=str,
        default="rubrics",
        help="Directory containing *.txt rubric definitions (default: rubrics/)",
    )
    parser.add_argument(
        "--rubric-model",
        type=str,
        help="Model as provider:model (e.g., openai:gpt-4o, azure_openai:o3-mini)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rubrics_output",
        help="Directory for CSV output (default: rubrics_output/)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["csv", "stdout"],
        default="csv",
        help="Output mode: csv (write files) or stdout (print only)",
    )

    # Filtering
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit number of tasks to evaluate",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only evaluate tasks in failed_tasks list",
    )

    # Other options
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Force JSON-mode (auto-enabled for OpenAI/Azure)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel rubric evaluations (default: 4)",
    )
    parser.add_argument(
        "--max-batch-messages",
        type=int,
        default=0,
        help="Max total messages per batch (0=disabled). Dynamically adjusts batch size.",
    )
    parser.add_argument(
        "--inter-batch-delay",
        type=float,
        default=0,
        help="Seconds to wait between batches (default: 0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per batch on failure (default: 3).",
    )
    parser.add_argument(
        "--sort-by-messages",
        action="store_true",
        help="Sort tasks from least to most messages before processing.",
    )
    parser.add_argument(
        "--sort-by-file-size",
        action="store_true",
        help="Sort trace files from smallest to largest file size before processing.",
    )
    parser.add_argument(
        "--inbetween",
        type=str,
        help="Bash command to execute after each trace file (e.g., 'TMUX= ./deploy_llm.sh')",
    )
    parser.add_argument(
        "--sleep",
        type=str,
        help="Sleep duration before and after inbetween command (e.g., '5s', '2m')",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (force re-evaluation of all tasks).",
    )

    args = parser.parse_args()

    # Parse sleep duration
    sleep_seconds = 0
    if args.sleep:
        match = re.match(r'^(\d+)(s|m)?$', args.sleep)
        if match:
            value = int(match.group(1))
            unit = match.group(2) or 's'
            sleep_seconds = value * 60 if unit == 'm' else value
        else:
            print(f"Invalid sleep format: {args.sleep}. Use e.g., '5s' or '2m'")
            sys.exit(1)

    # Set trace_dir (required by CLI but not used when trace_file is specified)
    args.trace_dir = str(REPO_ROOT / "traces")

    # Resolve rubric path
    if args.rubric:
        rubric_path = Path(args.rubric)
        if not rubric_path.is_absolute():
            rubric_path = REPO_ROOT / rubric_path
        args.rubric = str(rubric_path)

    # Resolve rubrics directory
    rubrics_dir = Path(args.rubrics_dir)
    if not rubrics_dir.is_absolute():
        rubrics_dir = REPO_ROOT / rubrics_dir
    args.rubrics_dir = str(rubrics_dir)

    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    args.output_dir = str(output_dir)

    # Resolve all trace file paths
    trace_files = []
    for trace_file in args.trace_files:
        trace_path = Path(trace_file)
        if not trace_path.is_absolute():
            trace_path = REPO_ROOT / trace_path
        trace_files.append(trace_path)

    # Sort by file size if requested (smallest to largest)
    if args.sort_by_file_size:
        trace_files.sort(key=lambda p: p.stat().st_size)
        print("Trace files sorted by file size (smallest to largest):")
        for tf in trace_files:
            size_mb = tf.stat().st_size / (1024 * 1024)
            print(f"  {tf.name}: {size_mb:.2f} MB")
        print()

    # Process each trace file independently
    for i, trace_path in enumerate(trace_files):
        args.trace_file = str(trace_path)

        print(f"\n{'='*60}")
        print(f"Processing: {trace_path.name}")
        print(f"{'='*60}\n")

        # Run the evaluator for this trace
        rubric_cli.run(args)

        # Execute inbetween command after each trace file
        if args.inbetween:
            if sleep_seconds:
                print(f"Sleeping for {sleep_seconds}s before inbetween command...")
                time.sleep(sleep_seconds)
            print(f"\n{'='*60}")
            print(f"Running inbetween command: {args.inbetween}")
            print(f"{'='*60}\n")
            subprocess.run(args.inbetween, shell=True, check=True)
            if sleep_seconds:
                print(f"Sleeping for {sleep_seconds}s after inbetween command...")
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
