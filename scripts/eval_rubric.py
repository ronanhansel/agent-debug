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
import sys
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
        required=True,
        help="Path to trace JSON file to evaluate",
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

    args = parser.parse_args()

    # Resolve trace path
    trace_path = Path(args.trace_file)
    if not trace_path.is_absolute():
        trace_path = REPO_ROOT / trace_path
    args.trace_file = str(trace_path)

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

    # Run the evaluator
    rubric_cli.run(args)


if __name__ == "__main__":
    main()
