#!/usr/bin/env python3
"""
Run SciCode baseline evaluations based on model_to_baseline_scicode.json.

Usage:
    python scripts/run_scicode_baseline.py
    python scripts/run_scicode_baseline.py --config model_to_baseline_scicode.json
    python scripts/run_scicode_baseline.py --model-id gpt-4o-2024-11-20
    python scripts/run_scicode_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS = REPO_ROOT / "hal-harness"


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the model configuration JSON."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_hal_eval(
    model: Dict[str, Any],
    benchmark: str,
    agent_dir: Path,
    max_concurrent: int = 1,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Run HAL evaluation for a single model."""

    model_id = model["model_id"]
    model_name = model["name"]
    provider = model.get("provider", "openai")
    reasoning_effort = model.get("reasoning_effort")

    # Build run ID
    run_id = f"{benchmark}_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build command
    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", model_name,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", str(max_concurrent),
        "-A", f"model_name={model_id}",
    ]

    # Add reasoning effort for reasoning models
    if reasoning_effort:
        cmd += ["-A", f"reasoning_effort={reasoning_effort}"]

    log(f"Model: {model_name}", "eval")
    log(f"Model ID: {model_id}", "eval")
    log(f"Run ID: {run_id}", "eval")
    log(f"Command: {' '.join(cmd)}", "eval")

    if dry_run:
        log("DRY RUN - skipping execution", "eval")
        return True, "dry_run"

    try:
        result = subprocess.run(
            cmd,
            cwd=HAL_HARNESS,
            capture_output=False,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode == 0:
            log(f"SUCCESS: {model_name}", "eval")
            return True, run_id
        else:
            log(f"FAILED: {model_name} (exit code {result.returncode})", "eval")
            return False, f"exit_code_{result.returncode}"

    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {model_name}", "eval")
        return False, "timeout"
    except Exception as e:
        log(f"ERROR: {model_name} - {e}", "eval")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run SciCode baseline evaluations from config."
    )

    parser.add_argument(
        "--config",
        default="model_to_baseline_scicode.json",
        help="Path to model config JSON (default: model_to_baseline_scicode.json)",
    )
    parser.add_argument(
        "--model-id",
        dest="model_ids",
        action="append",
        help="Only run specific model ID(s). Can be repeated.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Override max_concurrent from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay in seconds between model runs.",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    config = load_config(config_path)
    models = config.get("models", [])
    benchmark = config.get("benchmark", "scicode")
    agent_dir = Path(config.get("agent_dir", "hal-harness/agents/scicode_tool_calling_agent"))
    max_concurrent = args.max_concurrent or config.get("max_concurrent", 1)

    if not agent_dir.is_absolute():
        agent_dir = REPO_ROOT / agent_dir

    # List models mode
    if args.list_models:
        print(f"\nModels in {config_path.name}:\n")
        for m in models:
            effort = f" (reasoning: {m['reasoning_effort']})" if m.get('reasoning_effort') else ""
            print(f"  - {m['model_id']}: {m['name']}{effort}")
        print(f"\nTotal: {len(models)} models")
        return

    # Filter models if specific IDs requested
    if args.model_ids:
        models = [m for m in models if m["model_id"] in args.model_ids]
        if not models:
            print(f"Error: No models found matching: {args.model_ids}")
            sys.exit(1)

    log(f"Running {len(models)} model(s) on {benchmark}", "main")
    log(f"Agent dir: {agent_dir}", "main")
    log(f"Max concurrent: {max_concurrent}", "main")

    # Run evaluations
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        log(f"[{i}/{len(models)}] Starting: {model['name']}", "main")
        print("="*60)

        success, result = run_hal_eval(
            model=model,
            benchmark=benchmark,
            agent_dir=agent_dir,
            max_concurrent=max_concurrent,
            dry_run=args.dry_run,
        )

        results.append({
            "model_id": model["model_id"],
            "name": model["name"],
            "success": success,
            "result": result,
        })

        # Delay between runs
        if args.delay > 0 and i < len(models):
            log(f"Waiting {args.delay}s before next model...", "main")
            time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nSucceeded: {len(succeeded)}/{len(results)}")
    for r in succeeded:
        print(f"  + {r['model_id']}: {r['result']}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  - {r['model_id']}: {r['result']}")


if __name__ == "__main__":
    main()
