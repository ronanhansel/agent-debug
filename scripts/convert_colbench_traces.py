#!/usr/bin/env python3
"""
Convert ColBench trace files to include failed_tasks/successful_tasks lists.

ColBench traces have raw_eval_results as scores (0.0-1.0) but no explicit
failed_tasks list like SciCode. This script adds those fields for compatibility
with rubric evaluation pipelines.

Default thresholds (based on evaluation method):
  - Backend (test cases): threshold=1.0 (any test failure = failed)
  - Frontend (CLIP similarity): threshold=0.75 (visual mismatch = failed)

Usage:
    # Convert all ColBench traces with smart defaults
    python scripts/convert_colbench_traces.py

    # Custom threshold for all files
    python scripts/convert_colbench_traces.py --threshold 0.5

    # Specific files
    python scripts/convert_colbench_traces.py --files traces/colbench_backend*.json

    # Dry run (show what would happen)
    python scripts/convert_colbench_traces.py --dry-run
"""

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Smart defaults based on evaluation type
DEFAULT_BACKEND_THRESHOLD = 1.0   # Test cases: any failure counts
DEFAULT_FRONTEND_THRESHOLD = 0.75  # CLIP similarity: below 0.75 is poor


def get_threshold_for_file(filepath: Path, override_threshold: float | None) -> float:
    """Get appropriate threshold based on file type or override."""
    if override_threshold is not None:
        return override_threshold

    if "frontend" in filepath.name.lower():
        return DEFAULT_FRONTEND_THRESHOLD
    else:
        return DEFAULT_BACKEND_THRESHOLD


def convert_trace(input_path: Path, threshold_override: float | None, dry_run: bool) -> Path | None:
    """Convert a ColBench trace to include failed_tasks/successful_tasks."""
    with open(input_path) as f:
        data = json.load(f)

    # Check if already converted
    if "failed_tasks" in data.get("results", {}):
        print(f"  SKIP: {input_path.name} (already has failed_tasks)")
        return None

    # Get scores
    scores = data.get("raw_eval_results", [])
    if not scores:
        print(f"  SKIP: {input_path.name} (no raw_eval_results)")
        return None

    # Get appropriate threshold
    threshold = get_threshold_for_file(input_path, threshold_override)
    is_frontend = "frontend" in input_path.name.lower()
    task_type = "frontend/CLIP" if is_frontend else "backend/test"

    # Determine task IDs - they're indexed 0 to N-1
    failed_tasks = []
    successful_tasks = []

    for idx, score in enumerate(scores):
        task_id = str(idx)
        if score < threshold:
            failed_tasks.append(task_id)
        else:
            successful_tasks.append(task_id)

    # Update results
    if "results" not in data:
        data["results"] = {}

    data["results"]["failed_tasks"] = failed_tasks
    data["results"]["successful_tasks"] = successful_tasks

    # Also add accuracy based on threshold
    accuracy = len(successful_tasks) / len(scores) if scores else 0
    data["results"]["accuracy"] = accuracy

    # Generate output filename
    stem = input_path.stem.replace("_UPLOAD", "")
    output_name = f"{stem}_binary_UPLOAD.json"
    output_path = input_path.parent / output_name

    if dry_run:
        print(f"  DRY RUN: {input_path.name}")
        print(f"    Type: {task_type}, threshold={threshold}")
        print(f"    -> {output_name}")
        print(f"    Tasks: {len(scores)} total, {len(failed_tasks)} failed, {len(successful_tasks)} successful")
        print(f"    Accuracy: {accuracy:.2%}")
        return None

    # Write output
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  OK: {input_path.name}")
    print(f"    Type: {task_type}, threshold={threshold}")
    print(f"    -> {output_name}")
    print(f"    Tasks: {len(scores)} total, {len(failed_tasks)} failed, {len(successful_tasks)} successful")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert ColBench traces to include failed_tasks lists"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific trace files to convert (default: all colbench traces)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score threshold: tasks with score < threshold are 'failed'. "
             f"Default: {DEFAULT_BACKEND_THRESHOLD} for backend, {DEFAULT_FRONTEND_THRESHOLD} for frontend",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing files",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="traces",
        help="Directory containing trace files (default: traces/)",
    )

    args = parser.parse_args()

    # Find trace files
    traces_dir = REPO_ROOT / args.traces_dir
    if args.files:
        trace_files = []
        for pattern in args.files:
            trace_files.extend(glob.glob(str(REPO_ROOT / pattern)))
        trace_files = [Path(f) for f in trace_files]
    else:
        trace_files = list(traces_dir.glob("colbench_*_UPLOAD.json"))
        # Exclude already-converted binary files
        trace_files = [f for f in trace_files if "_binary_" not in f.name]

    if not trace_files:
        print("No ColBench trace files found.")
        sys.exit(1)

    print(f"Found {len(trace_files)} ColBench trace files")
    if args.threshold is not None:
        print(f"Threshold override: {args.threshold}")
    else:
        print(f"Using smart defaults: backend={DEFAULT_BACKEND_THRESHOLD}, frontend={DEFAULT_FRONTEND_THRESHOLD}")
    print()

    converted = []
    for trace_file in sorted(trace_files):
        result = convert_trace(trace_file, args.threshold, args.dry_run)
        if result:
            converted.append(result)

    print()
    if args.dry_run:
        print(f"Dry run complete. Would convert {len(trace_files)} files.")
    else:
        print(f"Converted {len(converted)} files.")
        if converted:
            print("\nOutput files:")
            for f in converted:
                print(f"  {f}")


if __name__ == "__main__":
    main()
