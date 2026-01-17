#!/usr/bin/env python3
"""
Re-evaluate existing SciCode submissions without re-running the agent.

This script takes existing RAW_SUBMISSIONS.jsonl files and re-runs the evaluation
step using the SciCode benchmark harness. Useful when:
- test_data.h5 was missing during original evaluation
- Evaluation code was updated
- Need to re-score existing submissions

Usage:
    python scripts/reevaluate_scicode.py --prefix scicode_tomato
    python scripts/reevaluate_scicode.py --prefix scicode_tomato --parallel 20
    python scripts/reevaluate_scicode.py --prefix scicode_tomato --dry-run
    python scripts/reevaluate_scicode.py --prefix scicode_tomato --task-id 12 --task-id 35
"""

import argparse
import json
import os
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add hal-harness to path
REPO_ROOT = Path(__file__).resolve().parent.parent
HAL_HARNESS = REPO_ROOT / "hal-harness"
sys.path.insert(0, str(HAL_HARNESS))

from hal.benchmarks.scicode import SciCodeBenchmark


def find_submissions(results_dir: Path, prefix: str, task_ids: Optional[List[str]] = None) -> List[Path]:
    """Find all RAW_SUBMISSIONS.jsonl files matching the prefix."""
    submissions = []

    scicode_results = results_dir / "scicode"
    if not scicode_results.exists():
        print(f"Results directory not found: {scicode_results}")
        return submissions

    for run_dir in scicode_results.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith(prefix):
            continue

        # Check if task_id filter applies
        if task_ids:
            # Extract task_id from run_id (format: prefix_model_taskid_benchmark_timestamp)
            parts = run_dir.name.split("_")
            # Find the task id - it's usually after the model name and before 'scicode'
            found_task = False
            for i, part in enumerate(parts):
                if part == "scicode" and i > 0:
                    potential_task = parts[i-1]
                    if potential_task in task_ids:
                        found_task = True
                        break
            if not found_task:
                continue

        # Find RAW_SUBMISSIONS.jsonl
        for f in run_dir.iterdir():
            if f.name.endswith("_RAW_SUBMISSIONS.jsonl"):
                submissions.append(f)
                break

    return sorted(submissions)


def load_submission(submission_path: Path) -> Dict[str, Any]:
    """Load agent output from RAW_SUBMISSIONS.jsonl."""
    with open(submission_path, "r", encoding="utf-8") as f:
        # JSONL format - each line is a JSON object
        # For SciCode, there's typically one line per run
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def extract_run_info(submission_path: Path) -> Dict[str, Any]:
    """Extract run information from the submission path."""
    run_dir = submission_path.parent
    run_id = run_dir.name

    # Try to load original config from UPLOAD.json
    for f in run_dir.iterdir():
        if f.name.endswith("_UPLOAD.json"):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.loads(fh.read())
                return {
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "original_config": data.get("config", {}),
                    "agent_args": data.get("config", {}).get("agent_args", {}),
                }

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "original_config": {},
        "agent_args": {},
    }


def reevaluate_submission_worker(
    submission_path: Path,
    output_prefix: str,
    traces_dir: Path,
    benchmark_name: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Worker function for parallel re-evaluation.
    Returns (run_id, result_data, error_message)
    """
    # Import inside worker to avoid pickling issues
    from hal.benchmarks.scicode import SciCodeBenchmark

    run_info = extract_run_info(submission_path)
    run_id = run_info["run_id"]
    original_config = run_info["original_config"]

    try:
        # Initialize benchmark in worker
        benchmark = SciCodeBenchmark(
            agent_dir="",
            config={},
            benchmark_name=benchmark_name,
        )

        # Load the agent output
        agent_output = load_submission(submission_path)
        if not agent_output:
            return (run_id, None, "No agent output found")

        # Create a unique run_id for re-evaluation
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        reeval_run_id = f"{output_prefix}{run_id}_{timestamp}"

        # Run evaluation
        eval_results = benchmark.evaluate_output(agent_output, reeval_run_id)
        metrics = benchmark.get_metrics(eval_results)

        # Build trace data
        trace_data = {
            "config": {
                "agent_name": original_config.get("agent_name", "reeval"),
                "benchmark_name": original_config.get("benchmark_name", "scicode"),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "run_id": reeval_run_id,
                "agent_args": original_config.get("agent_args", {}),
                "original_run_id": run_id,
                "reevaluation": True,
            },
            "results": {
                **metrics,
                "total_cost": 0,
                "latencies": {},
            },
            "raw_eval_results": eval_results,
            "raw_logging_results": [],
            "total_usage": {},
            "total_cost": 0,
        }

        # Save trace
        trace_filename = f"{reeval_run_id}_UPLOAD.json"
        trace_path = traces_dir / trace_filename

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)

        return (run_id, trace_data, None)

    except Exception as e:
        import traceback
        return (run_id, None, f"{e}\n{traceback.format_exc()}")


def reevaluate_submission(
    submission_path: Path,
    benchmark: SciCodeBenchmark,
    output_prefix: str,
    traces_dir: Path,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Re-evaluate a single submission (sequential mode)."""
    run_info = extract_run_info(submission_path)
    run_id = run_info["run_id"]
    original_config = run_info["original_config"]

    print(f"\n{'='*60}")
    print(f"Re-evaluating: {run_id}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Would re-evaluate this submission")
        return None

    # Load the agent output
    agent_output = load_submission(submission_path)
    if not agent_output:
        print(f"  ERROR: No agent output found in {submission_path}")
        return None

    print(f"  Loaded {len(agent_output)} task(s) from submission")

    # Create a unique run_id for re-evaluation
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reeval_run_id = f"{output_prefix}{run_id}_{timestamp}"

    # Run evaluation
    print(f"  Running evaluation...")
    try:
        eval_results = benchmark.evaluate_output(agent_output, reeval_run_id)
        metrics = benchmark.get_metrics(eval_results)
    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"  Evaluation complete:")
    print(f"    Successful tasks: {metrics.get('successful_tasks', [])}")
    print(f"    Failed tasks: {metrics.get('failed_tasks', [])}")
    print(f"    Accuracy: {metrics.get('accuracy', 0):.1%}")
    if 'subtask_accuracy' in metrics:
        print(f"    Subtask accuracy: {metrics.get('subtask_accuracy', 0):.1%}")

    # Build trace data
    trace_data = {
        "config": {
            "agent_name": original_config.get("agent_name", "reeval"),
            "benchmark_name": original_config.get("benchmark_name", "scicode"),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "run_id": reeval_run_id,
            "agent_args": original_config.get("agent_args", {}),
            "original_run_id": run_id,
            "reevaluation": True,
        },
        "results": {
            **metrics,
            "total_cost": 0,
            "latencies": {},
        },
        "raw_eval_results": eval_results,
        "raw_logging_results": [],
        "total_usage": {},
        "total_cost": 0,
    }

    # Save trace
    trace_filename = f"{reeval_run_id}_UPLOAD.json"
    trace_path = traces_dir / trace_filename

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    print(f"  Trace saved: {trace_path.name}")

    return trace_data


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing SciCode submissions"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix of run IDs to re-evaluate (e.g., 'scicode_tomato')",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="reeval_",
        help="Prefix for output traces (default: 'reeval_')",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        dest="task_ids",
        help="Only re-evaluate specific task IDs (can be repeated)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Directory containing results (default: results/)",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default=str(REPO_ROOT / "traces"),
        help="Directory for output traces (default: traces/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without running evaluations",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="scicode",
        choices=["scicode", "scicode_easy", "scicode_hard"],
        help="Benchmark variant (default: scicode)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    traces_dir = Path(args.traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Ensure prefix doesn't have trailing underscore for matching
    prefix = args.prefix.rstrip("_")

    # Build output prefix: reeval_original_prefix_
    output_prefix = f"{args.output_prefix}{prefix}_"

    print(f"{'='*60}")
    print(f"SCICODE RE-EVALUATION")
    print(f"{'='*60}")
    print(f"Input prefix: {prefix}")
    print(f"Output prefix: {output_prefix}")
    print(f"Results dir: {results_dir}")
    print(f"Traces dir: {traces_dir}")
    print(f"Benchmark: {args.benchmark}")
    if args.task_ids:
        print(f"Task filter: {args.task_ids}")
    if args.dry_run:
        print(f"MODE: DRY RUN")
    print()

    # Find submissions
    submissions = find_submissions(results_dir, prefix, args.task_ids)

    if not submissions:
        print(f"No submissions found matching prefix '{prefix}'")
        return

    print(f"Found {len(submissions)} submission(s) to re-evaluate:")
    for s in submissions[:10]:
        print(f"  - {s.parent.name}")
    if len(submissions) > 10:
        print(f"  ... and {len(submissions) - 10} more")
    print()

    # Initialize benchmark
    print("Initializing SciCode benchmark...")
    benchmark = SciCodeBenchmark(
        agent_dir="",  # Not needed for evaluation
        config={},
        benchmark_name=args.benchmark,
    )
    print(f"Loaded {len(benchmark.benchmark)} tasks from dataset")
    print()

    # Check test_data.h5 exists
    test_data_path = Path(benchmark.benchmark_dir) / "eval" / "data" / "test_data.h5"
    if not test_data_path.exists():
        print(f"ERROR: test_data.h5 not found at {test_data_path}")
        print("Please download it from: https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR")
        return
    print(f"✓ test_data.h5 found ({test_data_path.stat().st_size / 1e9:.2f} GB)")
    print()

    # Re-evaluate submissions
    results = []

    if args.dry_run:
        # Dry run mode - sequential preview
        for i, submission_path in enumerate(submissions, 1):
            print(f"\n[{i}/{len(submissions)}] Processing...")
            result = reevaluate_submission(
                submission_path,
                benchmark,
                output_prefix,
                traces_dir,
                dry_run=True,
            )
            if result:
                results.append(result)

    elif args.parallel > 1:
        # Parallel mode
        print(f"Running {len(submissions)} evaluations with {args.parallel} parallel workers...")
        print()

        completed = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    reevaluate_submission_worker,
                    submission_path,
                    output_prefix,
                    traces_dir,
                    args.benchmark,
                ): submission_path
                for submission_path in submissions
            }

            # Process results as they complete
            for future in as_completed(futures):
                submission_path = futures[future]
                try:
                    run_id, trace_data, error = future.result()
                    completed += 1

                    if error:
                        errors += 1
                        print(f"[{completed}/{len(submissions)}] ✗ {run_id}: {error.split(chr(10))[0]}")
                    else:
                        results.append(trace_data)
                        successful = trace_data["results"].get("successful_tasks", [])
                        failed = trace_data["results"].get("failed_tasks", [])
                        print(f"[{completed}/{len(submissions)}] ✓ {run_id}: {len(successful)} passed, {len(failed)} failed")

                except Exception as e:
                    completed += 1
                    errors += 1
                    print(f"[{completed}/{len(submissions)}] ✗ {submission_path.parent.name}: {e}")

        print(f"\nCompleted: {completed}, Errors: {errors}")

    else:
        # Sequential mode
        for i, submission_path in enumerate(submissions, 1):
            print(f"\n[{i}/{len(submissions)}] Processing...")
            result = reevaluate_submission(
                submission_path,
                benchmark,
                output_prefix,
                traces_dir,
                dry_run=False,
            )
            if result:
                results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"RE-EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {len(submissions)} submissions")
    if not args.dry_run:
        print(f"Successful: {len(results)}")

        # Aggregate stats
        total_successful = sum(len(r["results"].get("successful_tasks", [])) for r in results)
        total_failed = sum(len(r["results"].get("failed_tasks", [])) for r in results)
        print(f"Total successful tasks: {total_successful}")
        print(f"Total failed tasks: {total_failed}")

        if results:
            print(f"\nTraces saved to: {traces_dir}")
            print(f"Trace prefix: {output_prefix}")


if __name__ == "__main__":
    main()
