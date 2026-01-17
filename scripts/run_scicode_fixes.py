#!/usr/bin/env python3
"""
Apply SciCode fix packages and re-run HAL evaluations.

This script:
1. Loads fixes from fixes/scicode/<task_id>/
2. Creates a modified dataset with instruction clarifications injected
3. Runs HAL evaluation for fixed tasks only (filtered by model failures)
4. Outputs traces with a configurable prefix

Usage:
    # List available fixes
    python scripts/run_scicode_fixes.py --list-fixes

    # Dry run - see what would happen
    python scripts/run_scicode_fixes.py \
        --task-id 11 --task-id 12 \
        --dry-run

    # Run fixes for all failed tasks that have fixes (recommended)
    python scripts/run_scicode_fixes.py \
        --prefix iter1_ \
        --rubric-csv rubrics_output/scicode/scicode_combined.csv \
        --docker

    # Force a specific model for all tasks
    python scripts/run_scicode_fixes.py \
        --prefix iter1_ \
        --model gpt-4o-2024-11-20 \
        --docker

    # Actually run fixes (legacy mode - all fixes, default model)
    python scripts/run_scicode_fixes.py \
        --agent-dir hal-harness/agents/scicode_tool_calling_agent \
        --agent-args agent_args.json \
        --task-id 11 \
        --output-prefix fixed_scicode
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "model_to_baseline_scicode.json"


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def _slugify(value: str, fallback: str) -> str:
    """Keep readable identifiers for run_ids / filenames."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline_scicode.json and return a dict mapping model_id to config.

    The scicode config format has a 'models' array, unlike corebench which uses a flat dict.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))

    # Handle scicode format: {"models": [...], "benchmark": "scicode", ...}
    if "models" in data and isinstance(data["models"], list):
        result = {}
        for model_entry in data["models"]:
            model_id = model_entry.get("model_id")
            if model_id:
                result[model_id] = model_entry
                # Also index by name for convenience
                name = model_entry.get("name", "")
                if name and name != model_id:
                    result[name] = model_entry
        return result

    # Fallback: assume flat dict format like corebench
    if isinstance(data, dict):
        return data

    return {}


def load_rubric_task_models(csv_path: Path) -> List[Tuple[str, str]]:
    """Load rubric CSV and return list of (task_id, model_id) pairs for failed tasks.

    CSV format: task_id,criteria,grade,correct,explanation,model_run
    We extract failed tasks (correct=0) and parse model_id from model_run column.
    """
    task_model_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    if not csv_path.exists():
        return task_model_pairs

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            correct = row.get("correct", "").strip()
            model_run = row.get("model_run", "").strip()

            # Only include failed tasks (correct=0)
            if task_id and correct == "0" and model_run:
                # Extract model_id from model_run
                # Format examples:
                # - scicode_hal_generalist_agent_gpt4120250414_1745597325_UPLOAD
                # - scicode_scicode_tool_calling_agent_claudesonnet45_high_1759429729_UPLOAD
                model_id = _extract_model_from_run_name(model_run)
                if model_id and (task_id, model_id) not in seen:
                    task_model_pairs.append((task_id, model_id))
                    seen.add((task_id, model_id))

    return task_model_pairs


def _extract_model_from_run_name(model_run: str) -> Optional[str]:
    """Extract a model identifier from the model_run column.

    Examples:
    - scicode_hal_generalist_agent_gpt4120250414_... -> gpt-4.1-2025-04-14
    - scicode_scicode_tool_calling_agent_claudesonnet45_high_... -> claude-sonnet-4-20250514
    - scicode_hal_generalist_agent_o4mini20250416_low_... -> o4-mini-2025-04-16
    """
    # Known model patterns and their mappings
    patterns = {
        r"gpt4o[\-_]?2024[\-_]?11[\-_]?20": "gpt-4o-2024-11-20",
        r"gpt[\-_]?4[\-_]?1[\-_]?2025[\-_]?04[\-_]?14|gpt4120250414": "gpt-4.1-2025-04-14",
        r"o3[\-_]?mini[\-_]?2025[\-_]?01[\-_]?31|o3mini20250131": "o3-mini-2025-01-31",
        r"o4[\-_]?mini[\-_]?2025[\-_]?04[\-_]?16|o4mini20250416": "o4-mini-2025-04-16",
        r"claude[\-_]?sonnet[\-_]?4[\-_]?5|claudesonnet45": "claude-sonnet-4-20250514",
        r"claude[\-_]?sonnet[\-_]?4[\-_]?20250514|claudesonnet420250514": "claude-sonnet-4-20250514",
        r"claude[\-_]?3[\-_]?7[\-_]?sonnet|claude37sonnet": "claude-3-7-sonnet-20250219",
        r"claude[\-_]?opus[\-_]?4[\-_]?1|claudeopus41": "claude-opus-4-1-20250514",
        r"deepseek[\-_]?v3|DeepSeekV3": "deepseek-ai/DeepSeek-V3",
    }

    model_run_lower = model_run.lower()
    for pattern, model_id in patterns.items():
        if re.search(pattern, model_run_lower, re.IGNORECASE):
            return model_id

    return None


def get_agent_args_for_model(
    model_key: str,
    model_config: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Get the agent_args for a specific model.

    Args:
        model_key: The model name/key to look up
        model_config: The model config dict

    Returns:
        Agent args dict with model_name, reasoning_effort, etc.
        Returns None if model not found in config.
    """
    if not model_key:
        return None

    # Direct lookup
    model_entry = model_config.get(model_key)

    # Try matching by model_id
    if not model_entry:
        for key, entry in model_config.items():
            if entry.get("model_id") == model_key:
                model_entry = entry
                break

    if not model_entry:
        return None

    # Build agent_args from model config
    model_id = model_entry.get("model_id")
    if not model_id:
        return None

    agent_args: Dict[str, Any] = {
        "model_name": model_id,
    }
    if "reasoning_effort" in model_entry:
        agent_args["reasoning_effort"] = model_entry["reasoning_effort"]
    if "max_steps" in model_entry:
        agent_args["max_steps"] = model_entry["max_steps"]

    return agent_args


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
    docker: bool = False,
    task_id: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Run HAL evaluation with a custom dataset."""

    # Build run_id with model info for better traceability
    model_name = str(agent_args.get("model_name", "model"))
    model_slug = _slugify(model_name.replace("/", "_"), "model")
    effort = str(agent_args.get("reasoning_effort") or "").strip().lower()
    effort_part = f"_{_slugify(effort, '')}" if effort else ""
    task_part = f"_{task_id}" if task_id else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{output_prefix}{model_slug}{effort_part}{task_part}_{benchmark}_{timestamp}"

    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", output_prefix,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    if docker:
        cmd.append("--docker")

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
        "--prefix",
        help="Prefix for run IDs and output files (e.g., 'iter1_'). Takes precedence over --output-prefix.",
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
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run HAL evaluation with --docker flag.",
    )
    parser.add_argument(
        "--rubric-csv",
        help="Path to rubric CSV to determine which model failed for each task.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model_to_baseline_scicode.json mapping model names to configs.",
    )
    parser.add_argument(
        "--model",
        help="Force a specific model for all tasks (overrides rubric CSV). Use model_id from model config.",
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
