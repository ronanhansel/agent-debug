#!/usr/bin/env python3
"""
Unified Benchmark Fix Runner

This script runs HAL evaluations for benchmarks that have fixes available.
It automatically detects which tasks have fixes and runs them ALL.

Usage:
    # List available configurations for a benchmark
    python scripts/run_benchmark_fixes.py --benchmark scicode --list-configs

    # List all benchmarks with fixes
    python scripts/run_benchmark_fixes.py --list-benchmarks

    # Run ALL models on ALL benchmarks that have fixes
    python scripts/run_benchmark_fixes.py --all-benchmarks --all-configs --prefix iter1_ --docker

    # Run a specific configuration on a specific benchmark
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --config gpt-5_scicode_tool_calling \
        --prefix test_ \
        --docker

    # Run all configurations for a single benchmark
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --all-configs \
        --prefix iter1_ \
        --docker

    # Filter by agent
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --agent scicode_tool_calling_agent \
        --prefix test_

    # Filter by model pattern
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --model-filter gpt-5 \
        --prefix test_

    # Dry run
    python scripts/run_benchmark_fixes.py --all-benchmarks --all-configs --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import threading

# Load .env file early to ensure Azure config is available
from dotenv import load_dotenv
load_dotenv()

# If using direct Azure, remove proxy URLs
if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
    for key in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_BASE_URL', 'LITELLM_BASE_URL'):
        os.environ.pop(key, None)
    print("[INFO] Direct Azure mode: removed proxy URLs from environment")

# =============================================================================
# Path Configuration
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS = REPO_ROOT / "hal-harness"
FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = REPO_ROOT / "traces"
RESULTS_DIR = REPO_ROOT / "results"
TMP_DIR = REPO_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

# Add shared module to path
sys.path.insert(0, str(HAL_HARNESS / "agents"))

# =============================================================================
# Benchmark to HAL benchmark name mapping
# =============================================================================
# Maps fixes directory names to HAL benchmark names
BENCHMARK_HAL_NAME_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "corebench_hard": "corebench_hard",
    "usaco": "usaco",
    # ColBench: ONLY backend is supported, frontend uses CLIP which is different
    "colbench": "colbench_backend_programming",
    "colbench_backend_programming": "colbench_backend_programming",
}

# ColBench warning - frontend is NOT supported
COLBENCH_WARNING = """
================================================================================
WARNING: ColBench only supports colbench_backend_programming tasks!

The frontend tasks (colbench_frontend_design) use CLIP similarity evaluation
which requires different handling and is NOT supported by this script.

Running: colbench_backend_programming
================================================================================
"""


def log(msg: str, prefix: str = "main") -> None:
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


# =============================================================================
# Benchmark and Fix Detection
# =============================================================================

def get_benchmarks_with_fixes() -> List[str]:
    """
    Get list of benchmarks that have fixes available.

    Returns benchmark names (keys for model_to_baseline_*.json files).
    """
    benchmarks = set()

    if not FIXES_DIR.exists():
        return []

    for subdir in FIXES_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            # Check if there are actual task fixes inside
            task_dirs = [d for d in subdir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if task_dirs:
                # Map directory name to config file name
                dir_name = subdir.name

                # Map to config file names
                if dir_name in ("colbench", "colbench_backend_programming"):
                    benchmarks.add("colbench")
                elif dir_name == "corebench_hard":
                    benchmarks.add("corebench")
                else:
                    benchmarks.add(dir_name)

    return sorted(benchmarks)


def get_task_ids_with_fixes(benchmark: str) -> List[str]:
    """
    Get list of task IDs that have fixes for a benchmark.

    Args:
        benchmark: Benchmark name (config file key)

    Returns:
        List of task IDs with fixes
    """
    task_ids = []

    # Map benchmark config name to fixes directory name(s)
    if benchmark == "colbench":
        fix_dirs = [FIXES_DIR / "colbench", FIXES_DIR / "colbench_backend_programming"]
    elif benchmark == "corebench":
        fix_dirs = [FIXES_DIR / "corebench_hard"]
    else:
        fix_dirs = [FIXES_DIR / benchmark]

    for fix_dir in fix_dirs:
        if fix_dir.exists():
            for task_dir in fix_dir.iterdir():
                if task_dir.is_dir() and not task_dir.name.startswith("."):
                    # Check if there's at least one fix file inside
                    fix_files = [f for f in task_dir.iterdir() if f.suffix == ".json" and f.name != "status.json"]
                    if fix_files or (task_dir / "README.md").exists():
                        task_ids.append(task_dir.name)

    return sorted(set(task_ids), key=lambda x: (int(x) if x.isdigit() else float('inf'), x))


# =============================================================================
# Configuration Loading
# =============================================================================

def load_benchmark_config(benchmark: str) -> Dict[str, Any]:
    """Load model_to_baseline_<benchmark>.json configuration."""
    config_path = REPO_ROOT / f"model_to_baseline_{benchmark}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def get_model_entries(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get all model entries from config (excluding _meta)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def get_agent_info(entry: Dict[str, Any]) -> Tuple[Path, str]:
    """Extract agent directory and function from entry."""
    agent_dir = entry.get("agent_dir", "")
    agent_function = entry.get("agent_function", "main.run")

    if not agent_dir:
        raise ValueError("agent_dir not found in entry")

    # Resolve to absolute path
    agent_path = Path(agent_dir)
    if not agent_path.is_absolute():
        agent_path = REPO_ROOT / agent_path

    return agent_path, agent_function


def build_agent_args(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build agent_args dict from model entry."""
    args: Dict[str, Any] = {}

    if "model_id" in entry:
        args["model_name"] = entry["model_id"]

    for key in ["reasoning_effort", "temperature", "max_steps", "budget"]:
        if key in entry:
            args[key] = entry[key]

    return args


# =============================================================================
# Retry Logic
# =============================================================================

def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (TRAPI timeout, auth, rate limit)."""
    retryable_patterns = [
        "timeout", "timed out", "connection reset", "connection refused",
        "503", "504", "502", "500", "429", "rate limit",
        "401", "403", "unauthorized", "authentication", "invalid token",
        "expired token", "token expired", "authenticationerror",
        "overloaded", "service unavailable", "bad gateway",
    ]
    error_lower = str(error_msg).lower()
    return any(pattern in error_lower for pattern in retryable_patterns)


def run_with_retry(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    max_retries: int = 3,
    base_timeout: int = 3600,
) -> Tuple[bool, str, Optional[subprocess.CompletedProcess]]:
    """Run subprocess with retry logic."""
    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)

        try:
            if attempt > 0:
                log(f"Attempt {attempt + 1}/{max_retries} (timeout={timeout}s)", "retry")

            result = subprocess.run(
                cmd, cwd=cwd, env=env, timeout=timeout,
                capture_output=True, text=True,
            )

            if result.returncode == 0:
                return True, "Success", result

            combined_output = (result.stderr or "") + (result.stdout or "")

            if is_retryable_error(combined_output):
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                log(f"Retryable error. Waiting {wait_time:.1f}s...", "retry")
                time.sleep(wait_time)
                continue
            else:
                return False, f"Exit code {result.returncode}", result

        except subprocess.TimeoutExpired:
            log(f"Timeout on attempt {attempt + 1}/{max_retries}", "retry")
            if attempt < max_retries - 1:
                time.sleep(min(30, (2 ** attempt)))
            else:
                return False, "Timeout after retries", None

        except Exception as e:
            if is_retryable_error(str(e)) and attempt < max_retries - 1:
                time.sleep(min(60, (2 ** attempt)))
            else:
                return False, str(e), None

    return False, "Max retries exceeded", None


# =============================================================================
# HAL Evaluation Runner
# =============================================================================

def run_hal_eval(
    benchmark: str,
    config_key: str,
    entry: Dict[str, Any],
    prefix: str,
    docker: bool = False,
    task_ids: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Run HAL evaluation for a specific configuration.

    Args:
        benchmark: HAL benchmark name (e.g., "scicode", "colbench_backend_programming")
        config_key: Configuration key (e.g., "gpt-5_scicode_tool_calling")
        entry: Model entry dict from config
        prefix: Output prefix for traces
        docker: Whether to use Docker isolation
        task_ids: Specific task IDs to run (None = all)
        max_tasks: Maximum number of tasks to run

    Returns:
        (success, message, trace_path)
    """
    agent_path, agent_function = get_agent_info(entry)
    agent_args = build_agent_args(entry)

    # Build run_id from config_key (e.g., "o4-mini_core_agent")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}{config_key}_{timestamp}"

    # Build command
    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", f"{prefix}{config_key}",
        "--agent_function", agent_function,
        "--agent_dir", str(agent_path),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    if docker:
        cmd.append("--docker")

    if max_tasks:
        cmd.extend(["--max_tasks", str(max_tasks)])

    # Add task IDs if specified
    if task_ids:
        # Create a temporary task filter file
        filter_file = TMP_DIR / f"task_filter_{run_id}.json"
        filter_file.write_text(json.dumps(task_ids))
        cmd.extend(["--task_ids", str(filter_file)])

    # Add agent args
    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd.extend(["-A", f"{key}={json.dumps(value)}"])
        else:
            cmd.extend(["-A", f"{key}={value}"])

    # Add benchmark name as agent arg (some agents need it)
    cmd.extend(["-A", f"benchmark_name={benchmark}"])

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HAL_HARNESS}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("WANDB_SILENT", "true")
    env["HAL_PRICING_MODEL_NAME"] = config_key
    env["HAL_WEAVE_PROJECT"] = f"{prefix.rstrip('_')}_{benchmark}"

    log(f"Running: {config_key}", "hal")
    log(f"Agent: {agent_path.name}", "hal")
    log(f"Model: {agent_args.get('model_name')}", "hal")
    log(f"Benchmark: {benchmark}", "hal")
    if task_ids:
        log(f"Tasks: {len(task_ids)} tasks with fixes", "hal")
    log(f"Run ID: {run_id}", "hal")

    # Run with retry
    success, error_msg, result = run_with_retry(
        cmd=cmd, env=env, cwd=REPO_ROOT,
        max_retries=3, base_timeout=7200,  # 2 hours base timeout
    )

    if not success:
        log(f"Failed: {error_msg}", "hal")
        return False, error_msg, None

    # Find output trace
    results_dir = RESULTS_DIR / benchmark / run_id
    trace_path = results_dir / f"{run_id}_UPLOAD.json"

    # Also check HAL_HARNESS results as fallback
    if not trace_path.exists():
        alt_results = HAL_HARNESS / "results" / benchmark / run_id
        alt_trace = alt_results / f"{run_id}_UPLOAD.json"
        if alt_trace.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(alt_trace, trace_path)

    if trace_path.exists():
        # Copy to traces directory
        dest = TRACES_DIR / f"{prefix}{trace_path.name}"
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(trace_path, dest)
        log(f"Trace saved: {dest.name}", "hal")
        return True, "Success", dest

    return True, "Success (no trace)", None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark fix runner - runs ALL tasks with fixes."
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark", "-b",
        help="Benchmark name (scicode, scienceagentbench, corebench, colbench, usaco)."
    )
    parser.add_argument(
        "--all-benchmarks", action="store_true",
        help="Run ALL benchmarks that have fixes available."
    )

    # Configuration selection
    parser.add_argument(
        "--config", "-c", dest="configs", action="append",
        help="Config key(s) to run. Can be repeated."
    )
    parser.add_argument(
        "--all-configs", action="store_true",
        help="Run all configurations in the config file."
    )
    parser.add_argument(
        "--agent", "-a",
        help="Filter configs by agent name (e.g., 'scicode_tool_calling_agent')."
    )
    parser.add_argument(
        "--model-filter", "-m",
        help="Filter configs by model pattern (e.g., 'gpt-5')."
    )

    # Output
    parser.add_argument(
        "--prefix", "-p", default="run_",
        help="Prefix for run IDs and output files (default: run_)."
    )

    # Execution options
    parser.add_argument(
        "--docker", action="store_true",
        help="Run with Docker isolation."
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel runs (default: 1)."
    )
    parser.add_argument(
        "--max-tasks", type=int,
        help="Maximum tasks per config (for testing)."
    )

    # Modes
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List available configurations and exit."
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true",
        help="List benchmarks with fixes and exit."
    )
    parser.add_argument(
        "--list-fixes", action="store_true",
        help="List task IDs with fixes for each benchmark."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing."
    )

    args = parser.parse_args()

    # List benchmarks mode
    if args.list_benchmarks:
        benchmarks = get_benchmarks_with_fixes()
        print(f"\n{'='*70}")
        print("Benchmarks with fixes available:")
        print(f"{'='*70}\n")
        for b in benchmarks:
            task_count = len(get_task_ids_with_fixes(b))
            hal_name = BENCHMARK_HAL_NAME_MAP.get(b, b)
            if b == "colbench":
                print(f"  {b}: {task_count} tasks (runs as {hal_name} - BACKEND ONLY)")
            else:
                print(f"  {b}: {task_count} tasks")
        print(f"\n{'='*70}\n")
        return

    # Determine which benchmarks to run
    if args.all_benchmarks:
        benchmarks_to_run = get_benchmarks_with_fixes()
        if not benchmarks_to_run:
            log("ERROR: No benchmarks with fixes found", "main")
            sys.exit(1)
        log(f"Running ALL benchmarks with fixes: {', '.join(benchmarks_to_run)}", "main")
    elif args.benchmark:
        benchmarks_to_run = [args.benchmark]
    else:
        if not args.list_configs and not args.list_fixes:
            log("ERROR: Specify --benchmark or --all-benchmarks", "main")
            sys.exit(1)
        benchmarks_to_run = []

    # List fixes mode
    if args.list_fixes:
        if not benchmarks_to_run:
            benchmarks_to_run = get_benchmarks_with_fixes()

        print(f"\n{'='*70}")
        print("Tasks with fixes:")
        print(f"{'='*70}")

        for benchmark in benchmarks_to_run:
            task_ids = get_task_ids_with_fixes(benchmark)
            hal_name = BENCHMARK_HAL_NAME_MAP.get(benchmark, benchmark)
            print(f"\n{benchmark} ({len(task_ids)} tasks) -> HAL: {hal_name}")
            if benchmark == "colbench":
                print("  WARNING: Only colbench_backend_programming is supported!")
            for tid in task_ids:
                print(f"    - {tid}")

        print(f"\n{'='*70}\n")
        return

    # Process each benchmark
    all_results: List[Tuple[str, str, bool, str, Optional[Path]]] = []

    for benchmark in benchmarks_to_run:
        # Show ColBench warning
        if benchmark == "colbench":
            print(COLBENCH_WARNING)

        # Get HAL benchmark name
        hal_benchmark = BENCHMARK_HAL_NAME_MAP.get(benchmark, benchmark)

        # Get task IDs with fixes
        task_ids = get_task_ids_with_fixes(benchmark)
        if not task_ids:
            log(f"WARNING: No fixes found for {benchmark}, skipping", "main")
            continue

        log(f"Benchmark: {benchmark} (HAL: {hal_benchmark})", "main")
        log(f"Tasks with fixes: {len(task_ids)}", "main")

        # Load configuration
        try:
            config = load_benchmark_config(benchmark)
        except FileNotFoundError as e:
            log(f"ERROR: {e}", "main")
            continue

        entries = get_model_entries(config)
        meta = config.get("_meta", {})

        log(f"Found {len(entries)} configurations", "main")

        # List configs mode
        if args.list_configs:
            print(f"\n{'='*70}")
            print(f"Available configurations for {benchmark}")
            print(f"{'='*70}\n")

            # Group by agent
            by_agent: Dict[str, List[Tuple[str, Dict]]] = {}
            for key, entry in entries.items():
                agent_dir = entry.get("agent_dir", "unknown")
                agent_name = Path(agent_dir).name
                by_agent.setdefault(agent_name, []).append((key, entry))

            for agent_name, configs in sorted(by_agent.items()):
                print(f"\n{agent_name}:")
                for key, entry in configs:
                    model_id = entry.get("model_id", "?")
                    effort = entry.get("reasoning_effort", "")
                    effort_str = f" [{effort}]" if effort else ""
                    print(f"  {key}")
                    print(f"    model: {model_id}{effort_str}")

            print(f"\n{'='*70}")
            continue

        # Determine which configs to run
        selected: Dict[str, Dict[str, Any]] = {}

        if args.configs:
            for key in args.configs:
                if key in entries:
                    selected[key] = entries[key]
                else:
                    log(f"WARNING: Config '{key}' not found", "main")

        elif args.all_configs:
            selected = entries.copy()

        else:
            log("ERROR: Specify --config or --all-configs", "main")
            continue

        # Apply filters
        if args.agent:
            selected = {
                k: v for k, v in selected.items()
                if args.agent in v.get("agent_dir", "")
            }
            log(f"Filtered to {len(selected)} configs with agent '{args.agent}'", "main")

        if args.model_filter:
            pattern = args.model_filter.lower()
            selected = {
                k: v for k, v in selected.items()
                if pattern in v.get("model_id", "").lower() or pattern in v.get("short_name", "").lower()
            }
            log(f"Filtered to {len(selected)} configs matching '{args.model_filter}'", "main")

        if not selected:
            log(f"No configurations selected for {benchmark}", "main")
            continue

        # Ensure prefix ends with underscore
        prefix = args.prefix
        if not prefix.endswith("_"):
            prefix = prefix + "_"

        # Dry run
        if args.dry_run:
            print(f"\n{'='*70}")
            print(f"DRY RUN - {benchmark} (HAL: {hal_benchmark})")
            print(f"{'='*70}")
            print(f"\nTasks with fixes ({len(task_ids)}):")
            for tid in task_ids[:10]:
                print(f"  - {tid}")
            if len(task_ids) > 10:
                print(f"  ... and {len(task_ids) - 10} more")

            print(f"\nConfigurations ({len(selected)}):")
            for key, entry in selected.items():
                agent_path, agent_func = get_agent_info(entry)
                agent_args = build_agent_args(entry)
                print(f"\n  {key}:")
                print(f"    Agent: {agent_path.name}")
                print(f"    Model: {agent_args.get('model_name')}")
                if "reasoning_effort" in agent_args:
                    print(f"    Reasoning: {agent_args['reasoning_effort']}")

            print(f"\n{'='*70}")
            continue

        # Run evaluations
        log(f"Running {len(selected)} configurations with prefix '{prefix}'", "main")

        results: List[Tuple[str, bool, str, Optional[Path]]] = []
        lock = threading.Lock()

        def run_job(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, bool, str, Optional[Path]]:
            key, entry = item
            success, msg, trace = run_hal_eval(
                benchmark=hal_benchmark,
                config_key=key,
                entry=entry,
                prefix=prefix,
                docker=args.docker,
                task_ids=task_ids,  # Run ALL tasks with fixes
                max_tasks=args.max_tasks,
            )
            return key, success, msg, trace

        jobs = list(selected.items())

        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(run_job, job): job for job in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    with lock:
                        results.append(result)
                        key, success, msg, _ = result
                        status = "SUCCESS" if success else "FAILED"
                        log(f"[{len(results)}/{len(jobs)}] {status}: {key}", "main")
        else:
            for i, job in enumerate(jobs):
                result = run_job(job)
                results.append(result)
                key, success, msg, _ = result
                status = "SUCCESS" if success else "FAILED"
                log(f"[{i+1}/{len(jobs)}] {status}: {key}", "main")

        # Collect results
        for r in results:
            all_results.append((benchmark, r[0], r[1], r[2], r[3]))

    # Final Summary
    if all_results:
        succeeded = [r for r in all_results if r[2]]
        failed = [r for r in all_results if not r[2]]

        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total: {len(all_results)}")
        print(f"Succeeded: {len(succeeded)}")
        print(f"Failed: {len(failed)}")

        if succeeded:
            print("\nSuccessful runs:")
            for benchmark, key, _, _, trace in succeeded:
                trace_name = trace.name if trace else "no trace"
                print(f"  - [{benchmark}] {key}: {trace_name}")

        if failed:
            print("\nFailed runs:")
            for benchmark, key, _, msg, _ in failed:
                print(f"  - [{benchmark}] {key}: {msg}")

        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
