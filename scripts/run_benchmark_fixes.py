#!/usr/bin/env python3
"""
Unified Benchmark Fix Runner

This script runs HAL evaluations for any benchmark using the unified
model_to_baseline_*.json configuration format with per-model agent_dir.

Usage:
    # List available configurations
    python scripts/run_benchmark_fixes.py --benchmark scicode --list-configs

    # Run a specific configuration
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --config gpt-5_scicode_tool_calling \
        --prefix test_ \
        --docker

    # Run all configurations for a benchmark
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --all-configs \
        --prefix iter1_ \
        --docker

    # Filter by agent
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --agent scicode_tool_calling_agent \
        --prefix test_ \
        --docker

    # Filter by model pattern
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --model-filter gpt-5 \
        --prefix test_

    # Run specific task IDs
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --config gpt-5_scicode_tool_calling \
        --task-id 11 --task-id 12 \
        --prefix fix_

    # Dry run
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --all-configs --dry-run
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


def log(msg: str, prefix: str = "main") -> None:
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


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
        benchmark: Benchmark name (e.g., "scicode")
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
    short_name = entry.get("short_name", "model")

    # Build run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}{short_name}_{benchmark}_{timestamp}"

    # Build command
    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", f"{prefix}{short_name}",
        "--agent_function", agent_function,
        "--agent_dir", str(agent_path),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    if docker:
        cmd.append("--docker")

    if max_tasks:
        cmd.extend(["--max_tasks", str(max_tasks)])

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
        description="Unified benchmark fix runner using model_to_baseline configs."
    )

    # Required
    parser.add_argument(
        "--benchmark", "-b", required=True,
        help="Benchmark name (scicode, scienceagentbench, corebench, colbench, usaco)."
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
    parser.add_argument(
        "--task-id", dest="task_ids", action="append",
        help="Specific task ID(s) to run. Can be repeated."
    )

    # Modes
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List available configurations and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing."
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_benchmark_config(args.benchmark)
    except FileNotFoundError as e:
        log(f"ERROR: {e}", "main")
        sys.exit(1)

    entries = get_model_entries(config)
    meta = config.get("_meta", {})

    log(f"Benchmark: {args.benchmark}", "main")
    log(f"Description: {meta.get('description', 'N/A')}", "main")
    log(f"Found {len(entries)} configurations", "main")

    # List mode
    if args.list_configs:
        print(f"\n{'='*70}")
        print(f"Available configurations for {args.benchmark}")
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
                short_name = entry.get("short_name", "?")
                effort = entry.get("reasoning_effort", "")
                effort_str = f" [{effort}]" if effort else ""
                print(f"  {key}")
                print(f"    model: {model_id}{effort_str}")
                print(f"    short_name: {short_name}")

        print(f"\n{'='*70}")
        return

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
        log("ERROR: Specify --config, --all-configs, --agent, or --model-filter", "main")
        sys.exit(1)

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
        log("No configurations selected", "main")
        sys.exit(1)

    # Ensure prefix ends with underscore
    prefix = args.prefix
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    # Dry run
    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - Would execute:")
        print(f"{'='*70}\n")

        for key, entry in selected.items():
            agent_path, agent_func = get_agent_info(entry)
            agent_args = build_agent_args(entry)
            print(f"\n{key}:")
            print(f"  Agent: {agent_path.name}")
            print(f"  Function: {agent_func}")
            print(f"  Model: {agent_args.get('model_name')}")
            if "reasoning_effort" in agent_args:
                print(f"  Reasoning: {agent_args['reasoning_effort']}")

        print(f"\n{'='*70}")
        print(f"Total: {len(selected)} configurations")
        print(f"Prefix: {prefix}")
        print(f"Docker: {args.docker}")
        print(f"Parallel: {args.parallel}")
        print(f"{'='*70}\n")
        return

    # Run evaluations
    log(f"Running {len(selected)} configurations with prefix '{prefix}'", "main")

    results: List[Tuple[str, bool, str, Optional[Path]]] = []
    lock = threading.Lock()

    def run_job(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, bool, str, Optional[Path]]:
        key, entry = item
        success, msg, trace = run_hal_eval(
            benchmark=args.benchmark,
            config_key=key,
            entry=entry,
            prefix=prefix,
            docker=args.docker,
            task_ids=args.task_ids,
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

    # Summary
    succeeded = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(results)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if succeeded:
        print("\nSuccessful runs:")
        for key, _, _, trace in succeeded:
            trace_name = trace.name if trace else "no trace"
            print(f"  - {key}: {trace_name}")

    if failed:
        print("\nFailed runs:")
        for key, _, msg, _ in failed:
            print(f"  - {key}: {msg}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
