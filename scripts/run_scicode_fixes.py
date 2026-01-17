#!/usr/bin/env python3
"""
Apply SciCode fix packages and re-run HAL evaluations.

This script:
1. Loads fixes from fixes/scicode/<task_id>/
2. Creates a modified dataset with instruction clarifications injected
3. Runs HAL evaluation for fixed tasks only
4. Outputs traces with a configurable prefix

Usage:
    # List available fixes
    python scripts/run_scicode_fixes.py --list-fixes

    # Dry run - see what would happen
    python scripts/run_scicode_fixes.py \
        --task-id 11 --task-id 12 \
        --dry-run

    # Actually run fixes
    python scripts/run_scicode_fixes.py \
        --agent-dir hal-harness/agents/scicode_tool_calling_agent \
        --agent-args agent_args.json \
        --task-id 11 \
        --output-prefix fixed_scicode
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def load_fix_package(task_id: str, fixes_root: Path) -> Dict[str, Any]:
    """Load all fix files for a task."""
    fix_dir = fixes_root / task_id
    if not fix_dir.exists():
        return {}

    fix = {"task_id": task_id, "fix_dir": fix_dir}

    # Load different override types
    for name in ["dependency_override", "instruction_override", "evaluation_override", "env_override", "input_override"]:
        path = fix_dir / f"{name}.json"
        if path.exists():
            fix[name] = json.loads(path.read_text())

    # Load problem_statement.txt if exists (common format)
    ps_path = fix_dir / "problem_statement.txt"
    if ps_path.exists():
        fix["problem_statement"] = ps_path.read_text()

    # Load README
    readme = fix_dir / "README.md"
    if readme.exists():
        fix["readme"] = readme.read_text()

    return fix


def list_available_fixes(fixes_root: Path) -> List[str]:
    """List all task IDs that have fixes."""
    if not fixes_root.exists():
        return []

    task_ids = []
    for item in fixes_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            fix_files = [
                "dependency_override.json", "instruction_override.json",
                "evaluation_override.json", "env_override.json",
                "input_override.json", "problem_statement.txt", "README.md",
            ]
            if any((item / f).exists() for f in fix_files):
                task_ids.append(item.name)

    return sorted(task_ids)


def load_scicode_dataset() -> List[Dict[str, Any]]:
    """Load the SciCode dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        dataset = list(load_dataset("SciCode1/SciCode", split="test"))
        return dataset
    except Exception as e:
        log(f"Failed to load SciCode dataset: {e}", "data")
        return []


def apply_fix_to_task(task: Dict[str, Any], fix: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a fix to a task, modifying its prompt/instructions."""
    modified = json.loads(json.dumps(task))  # Deep copy

    # Build clarification text
    clarifications = []

    # From instruction_override
    if fix.get("instruction_override"):
        instr = fix["instruction_override"]
        if instr.get("clarifications"):
            clarifications.extend(instr["clarifications"])
        if instr.get("corrected_signature"):
            clarifications.append(f"Corrected function signature: {instr['corrected_signature']}")

    # From dependency_override
    if fix.get("dependency_override"):
        dep = fix["dependency_override"]
        if dep.get("additional_imports"):
            imports = ", ".join(dep["additional_imports"])
            clarifications.append(f"Additional allowed imports: {imports}")

    # From problem_statement.txt
    if fix.get("problem_statement"):
        clarifications.append(fix["problem_statement"])

    # Inject clarifications into the task
    if clarifications:
        clarification_text = "\n\n## CLARIFICATIONS\n" + "\n".join(f"- {c}" for c in clarifications)

        # Modify the problem description or required_dependencies
        if "problem_description" in modified:
            modified["problem_description"] += clarification_text
        elif "sub_steps" in modified and modified["sub_steps"]:
            # Add to first sub-step's problem text
            modified["sub_steps"][0]["problem_description_step"] = \
                modified["sub_steps"][0].get("problem_description_step", "") + clarification_text

    return modified


def create_filtered_dataset(
    tasks: List[Dict[str, Any]],
    fixes: Dict[str, Dict[str, Any]],
    task_ids: List[str],
) -> List[Dict[str, Any]]:
    """Create a filtered dataset with only the specified tasks, with fixes applied."""
    filtered = []

    for task in tasks:
        problem_id = str(task.get("problem_id", ""))
        if problem_id in task_ids:
            if problem_id in fixes:
                modified = apply_fix_to_task(task, fixes[problem_id])
                log(f"Applied fix to task {problem_id}", "data")
            else:
                modified = task
            filtered.append(modified)

    return filtered


def run_hal_eval(
    agent_dir: Path,
    agent_args: Dict[str, Any],
    dataset_path: Path,
    output_prefix: str,
    benchmark: str = "scicode",
) -> Tuple[bool, str, Optional[Path]]:
    """Run HAL evaluation with a custom dataset."""

    run_id = f"{output_prefix}_{int(time.time())}"

    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", output_prefix,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd += ["-A", f"{key}={json.dumps(value)}"]
        else:
            cmd += ["-A", f"{key}={value}"]

    # Set environment to use our custom dataset
    env = os.environ.copy()
    env["SCICODE_DATASET_PATH"] = str(dataset_path)

    log(f"Running HAL eval with run_id: {run_id}", "hal")
    log(f"Dataset: {dataset_path}", "hal")

    try:
        result = subprocess.run(
            cmd,
            cwd=HAL_HARNESS,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        # Find output trace
        results_dir = HAL_HARNESS / "results" / benchmark
        traces = sorted(results_dir.glob(f"*{run_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if result.returncode == 0:
            trace_path = traces[0] if traces else None
            # Copy trace to our traces directory with prefix
            if trace_path:
                dest = TRACES_DIR / f"{output_prefix}_{trace_path.name}"
                shutil.copy(trace_path, dest)
                log(f"Trace saved to: {dest}", "hal")
                return True, "Success", dest

            return True, "Success (no trace found)", None
        else:
            log(f"HAL eval failed: {result.stderr[:500]}", "hal")
            return False, result.stderr[:200], None

    except subprocess.TimeoutExpired:
        return False, "Timeout", None
    except Exception as e:
        return False, str(e), None


def main():
    parser = argparse.ArgumentParser(
        description="Apply SciCode fixes and re-run HAL evaluations."
    )

    parser.add_argument(
        "--fixes-root",
        default=str(FIXES_DIR / "scicode"),
        help="Directory containing fix packages (default: fixes/scicode).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(HAL_HARNESS / "agents" / "scicode_tool_calling_agent"),
        help="Path to the agent directory.",
    )
    parser.add_argument(
        "--agent-args",
        help="Path to JSON file with agent arguments.",
    )
    parser.add_argument(
        "--benchmark",
        default="scicode",
        help="Benchmark name (default: scicode).",
    )
    parser.add_argument(
        "--output-prefix",
        default="fixed",
        help="Prefix for output traces (default: fixed).",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Task ID(s) to process. Can be repeated.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done.",
    )
    parser.add_argument(
        "--list-fixes",
        action="store_true",
        help="List available fixes and exit.",
    )
    parser.add_argument(
        "--show-fix",
        help="Show details of a specific fix and exit.",
    )

    args = parser.parse_args()

    # Resolve paths
    fixes_root = Path(args.fixes_root)
    if not fixes_root.is_absolute():
        fixes_root = REPO_ROOT / fixes_root

    agent_dir = Path(args.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = REPO_ROOT / agent_dir

    # List fixes mode
    if args.list_fixes:
        available = list_available_fixes(fixes_root)
        print(f"\nAvailable fixes in {fixes_root}:\n")
        for task_id in available:
            fix = load_fix_package(task_id, fixes_root)
            types = []
            if fix.get("dependency_override"): types.append("dep")
            if fix.get("instruction_override"): types.append("instr")
            if fix.get("evaluation_override"): types.append("eval")
            if fix.get("problem_statement"): types.append("prompt")
            print(f"  {task_id}: [{', '.join(types) or 'readme only'}]")
        print(f"\nTotal: {len(available)} fixes")
        return

    # Show specific fix
    if args.show_fix:
        fix = load_fix_package(args.show_fix, fixes_root)
        if not fix:
            print(f"No fix found for task {args.show_fix}")
            return
        print(f"\n=== Fix for task {args.show_fix} ===\n")
        for key, value in fix.items():
            if key in ["fix_dir"]:
                continue
            if key == "readme":
                print(f"README:\n{value[:500]}...")
            elif isinstance(value, dict):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value[:200] if isinstance(value, str) else value}")
        return

    # Determine tasks to process
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = list_available_fixes(fixes_root)

    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]

    if not task_ids:
        log("No tasks to process. Use --task-id or create fixes first.", "main")
        return

    log(f"Processing {len(task_ids)} tasks: {task_ids}", "main")

    # Load fixes
    fixes = {}
    for task_id in task_ids:
        fix = load_fix_package(task_id, fixes_root)
        if fix:
            fixes[task_id] = fix

    if not fixes:
        log("No valid fixes found for specified tasks.", "main")
        return

    log(f"Loaded {len(fixes)} fixes", "main")

    # Dry run - show what would be done
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Would apply the following fixes:")
        print("="*60)

        for task_id, fix in fixes.items():
            print(f"\n--- Task {task_id} ---")
            if fix.get("instruction_override"):
                instr = fix["instruction_override"]
                if instr.get("clarifications"):
                    print("  Clarifications:")
                    for c in instr["clarifications"]:
                        print(f"    - {c[:80]}...")
            if fix.get("dependency_override"):
                dep = fix["dependency_override"]
                if dep.get("additional_imports"):
                    print(f"  Additional imports: {dep['additional_imports']}")
            if fix.get("problem_statement"):
                print(f"  Problem statement addition: {fix['problem_statement'][:100]}...")

        print(f"\n{'='*60}")
        print(f"Would run HAL eval with prefix: {args.output_prefix}")
        print(f"Output trace: traces/{args.output_prefix}_*.json")
        return

    # Load SciCode dataset
    log("Loading SciCode dataset from HuggingFace...", "data")
    dataset = load_scicode_dataset()
    if not dataset:
        log("Failed to load dataset", "main")
        return

    log(f"Loaded {len(dataset)} tasks from dataset", "data")

    # Create filtered dataset with fixes applied
    filtered = create_filtered_dataset(dataset, fixes, task_ids)
    log(f"Created filtered dataset with {len(filtered)} tasks", "data")

    if not filtered:
        log("No matching tasks found in dataset", "main")
        return

    # Write temporary dataset file
    temp_dataset = Path(tempfile.mktemp(suffix=".json", prefix="scicode_fixed_"))
    temp_dataset.write_text(json.dumps(filtered, indent=2))
    log(f"Wrote temporary dataset to {temp_dataset}", "data")

    # Load agent args
    agent_args = {"model_name": "gpt-4o"}
    if args.agent_args:
        args_path = Path(args.agent_args)
        if not args_path.is_absolute():
            args_path = REPO_ROOT / args_path
        if args_path.exists():
            agent_args = json.loads(args_path.read_text())
            log(f"Loaded agent args: {list(agent_args.keys())}", "main")

    # Run HAL eval
    success, message, trace_path = run_hal_eval(
        agent_dir=agent_dir,
        agent_args=agent_args,
        dataset_path=temp_dataset,
        output_prefix=args.output_prefix,
        benchmark=args.benchmark,
    )

    # Cleanup temp dataset
    temp_dataset.unlink(missing_ok=True)

    # Summary
    print(f"\n{'='*60}")
    print("RESULT")
    print("="*60)
    if success:
        print(f"✓ SUCCESS: {message}")
        if trace_path:
            print(f"  Trace: {trace_path}")
    else:
        print(f"✗ FAILED: {message}")


if __name__ == "__main__":
    main()
