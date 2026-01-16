#!/usr/bin/env python3
"""
Unified Pipeline for HAL Agent Debugging

This script orchestrates the full pipeline:
1. rubric    - Evaluate traces against rubrics (identify env barriers vs skill issues)
2. inspect   - Analyze failures and generate fix recommendations
3. fix       - Apply fixes and re-run evaluation
4. extract   - Extract conversation logs from Weave
5. merge     - Combine individual traces into merged trace

All operations support --prefix for iteration tracking (e.g., "orange_", "mango_").

Usage:
    # Full pipeline
    python scripts/pipeline.py full --prefix orange --trace-file traces/baseline.json

    # Individual steps
    python scripts/pipeline.py rubric --prefix orange --trace-file traces/baseline.json
    python scripts/pipeline.py extract --prefix orange --project hal-agent-debug
    python scripts/pipeline.py merge --prefix orange
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "traces"
RUBRICS_DIR = REPO_ROOT / "rubrics"
FIXES_DIR = REPO_ROOT / "fixes"
DEFAULT_BENCHMARK = "corebench_hard"


def log(msg: str, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)


# =============================================================================
# RUBRIC EVALUATION
# =============================================================================

def cmd_rubric(args: argparse.Namespace) -> int:
    """Evaluate traces against rubrics using gpt-5.2 model."""
    from openai import OpenAI
    import re

    prefix = args.prefix or ""
    trace_path = Path(args.trace_file)
    output_dir = Path(args.output_dir)

    if not trace_path.exists():
        log(f"Trace file not found: {trace_path}", prefix)
        return 1

    # Setup OpenAI client
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    client = OpenAI(base_url=base_url, api_key=args.api_key or "dummy")
    model = args.model

    # Load rubric
    rubric_files = list(Path(args.rubrics_dir).glob("*.txt"))
    if not rubric_files:
        log(f"No rubric files found in {args.rubrics_dir}", prefix)
        return 1
    rubric_text = rubric_files[0].read_text()
    rubric_name = rubric_files[0].stem
    log(f"Loaded rubric: {rubric_name}", prefix)

    # Load trace
    trace_data = json.loads(trace_path.read_text())
    results_block = trace_data.get("results", {})
    failed_tasks = set(results_block.get("failed_tasks", []))
    successful_tasks = set(results_block.get("successful_tasks", []))
    log(f"Trace: {len(failed_tasks)} failed, {len(successful_tasks)} successful", prefix)

    # Extract conversations by task
    raw_entries = trace_data.get("raw_logging_results", [])
    if not raw_entries:
        log("WARNING: No raw_logging_results in trace. Run 'extract' first.", prefix)
        return 1

    tasks: Dict[str, List[Dict]] = {}
    for entry in raw_entries:
        task_id = (
            entry.get("attributes", {}).get("weave_task_id")
            or entry.get("weave_task_id")
            or "unknown"
        )
        tasks.setdefault(task_id, []).append(entry)
    log(f"Extracted {len(tasks)} task conversations", prefix)

    # Filter to failed tasks if requested
    if args.failed_only:
        tasks = {k: v for k, v in tasks.items() if k in failed_tasks}
        log(f"Filtered to {len(tasks)} failed tasks", prefix)

    # Limit tasks
    task_ids = list(tasks.keys())
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]

    # Format conversation helper
    def format_conv(entries: List[Dict]) -> str:
        lines = []
        for entry in entries[:50]:
            messages = entry.get("inputs", {}).get("messages", [])
            for msg in messages[-5:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if content:
                    lines.append(f"[{role}]: {content[:1500]}")
            output = entry.get("output", {})
            if output:
                choices = output.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    if content:
                        lines.append(f"[assistant]: {content[:1500]}")
        return "\n\n".join(lines[-100:])

    # Evaluate each task
    results = []
    for i, task_id in enumerate(task_ids):
        log(f"Evaluating {i+1}/{len(task_ids)}: {task_id}", prefix)

        entries = tasks[task_id]
        conversation = format_conv(entries)
        if not conversation.strip():
            log(f"  Skipping - no conversation", prefix)
            continue

        prompt = f"""You are evaluating an agent transcript against the following rubric.

RUBRIC:
{rubric_text}

AGENT TRANSCRIPT FOR TASK {task_id}:
{conversation}

Based on the rubric criteria, evaluate this transcript and provide your response as a JSON object with:
- "score": 0 or 1 (1 = environmental barrier present, 0 = no environmental barrier)
- "explanation": detailed explanation citing evidence from the transcript

Remember: Only score 1 if there is a TRUE environmental barrier that NO agent could overcome. If the agent could have succeeded with different actions/tools, score 0.

Respond with ONLY the JSON object, no other text."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=15000,
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON
            if result_text.startswith("```"):
                result_text = re.sub(r"```(?:json)?\s*", "", result_text)
                result_text = result_text.rstrip("`")

            result = json.loads(result_text)
            score = float(result.get("score", 0))
            explanation = result.get("explanation", "")
            log(f"  Score: {score}", prefix)
            results.append({
                "task_id": task_id,
                "score": score,
                "explanation": explanation,
            })
        except Exception as e:
            log(f"  Error: {e}", prefix)
            results.append({
                "task_id": task_id,
                "score": 0,
                "explanation": "",
                "error": str(e),
            })

    # Write CSV
    output_subdir = output_dir / rubric_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    trace_label = f"{prefix}{trace_path.stem}" if prefix else trace_path.stem
    output_path = output_subdir / f"{trace_label}.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "criteria", "grade", "correct", "explanation", "model_run"])
        for r in results:
            correct = ""
            if r["task_id"] in successful_tasks:
                correct = "1"
            elif r["task_id"] in failed_tasks:
                correct = "0"
            writer.writerow([
                r["task_id"],
                rubric_name,
                f"{r['score']:.2f}",
                correct,
                r["explanation"],
                trace_label,
            ])

    env_barrier_count = sum(1 for r in results if r["score"] >= 1)
    log(f"Results: {env_barrier_count} env barriers, {len(results) - env_barrier_count} capability issues", prefix)
    log(f"Written to: {output_path}", prefix)
    return 0


# =============================================================================
# WEAVE TRACE EXTRACTION
# =============================================================================

def cmd_extract(args: argparse.Namespace) -> int:
    """Extract conversation logs from Weave for a prefix."""
    prefix = args.prefix
    if not prefix:
        log("--prefix is required for extract command")
        return 1

    project = args.project
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "extract_weave_traces.py"),
        "--project", project,
        "--prefix", prefix,
    ]
    if args.include_costs:
        cmd.append("--include-costs")
    else:
        cmd.append("--no-include-costs")

    if args.merge_input:
        for mi in args.merge_input:
            cmd.extend(["--merge-input", mi])

    log(f"Extracting Weave traces for prefix: {prefix}", prefix)
    log(f"Command: {' '.join(cmd)}", prefix)

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# =============================================================================
# TRACE MERGING
# =============================================================================

def cmd_merge(args: argparse.Namespace) -> int:
    """Merge individual traces into a combined trace."""
    prefix = args.prefix or ""

    # Find traces matching prefix
    if args.input_pattern:
        patterns = args.input_pattern
    else:
        patterns = [f"{TRACES_DIR}/{prefix}*_UPLOAD.json"]

    files: List[Path] = []
    for pattern in patterns:
        matched = [Path(p).resolve() for p in glob.glob(pattern)]
        files.extend(sorted(matched))

    # Dedupe
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    if not unique_files:
        log(f"No traces matched pattern: {patterns}", prefix)
        return 1

    log(f"Merging {len(unique_files)} traces", prefix)

    # Load and merge
    first_trace = json.loads(unique_files[0].read_text())
    config = json.loads(json.dumps(first_trace.get("config", {})))
    run_id = args.run_id or f"{prefix}MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config["run_id"] = run_id
    config["source_traces"] = [p.name for p in unique_files]

    merged_results: Dict[str, Any] = {
        "successful_tasks": [],
        "failed_tasks": [],
        "latencies": {},
    }
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_eval_results: Dict[str, Any] = {}
    raw_logging_results: List[Any] = []

    for path in unique_files:
        data = json.loads(path.read_text())
        results = data.get("results", {})
        merged_results["successful_tasks"].extend(results.get("successful_tasks") or [])
        merged_results["failed_tasks"].extend(results.get("failed_tasks") or [])

        # Latencies
        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_results["latencies"].setdefault(key, {
                        "first_call_timestamp": value.get("first_call_timestamp"),
                        "last_call_timestamp": value.get("last_call_timestamp"),
                    })
                    if value.get("first_call_timestamp"):
                        if not existing.get("first_call_timestamp") or value["first_call_timestamp"] < existing["first_call_timestamp"]:
                            existing["first_call_timestamp"] = value["first_call_timestamp"]
                    if value.get("last_call_timestamp"):
                        if not existing.get("last_call_timestamp") or value["last_call_timestamp"] > existing["last_call_timestamp"]:
                            existing["last_call_timestamp"] = value["last_call_timestamp"]
                else:
                    merged_results["latencies"][key] = merged_results["latencies"].get(key, 0) + value

        # Raw logging - THIS IS THE KEY FIX
        source_logging = data.get("raw_logging_results") or results.get("raw_logging_results")
        if source_logging:
            raw_logging_results.extend(source_logging)

        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        # Usage
        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(model_name, {"input_tokens": 0, "output_tokens": 0})
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

        # Eval results
        capsule_eval = data.get("raw_eval_results") or {}
        for task_id, stats in capsule_eval.items():
            raw_eval_results[task_id] = stats

    # Compute accuracy
    total_tasks = len(merged_results["successful_tasks"]) + len(merged_results["failed_tasks"])
    merged_results["accuracy"] = len(merged_results["successful_tasks"]) / total_tasks if total_tasks else 0.0
    merged_results["total_cost"] = total_cost

    merged: Dict[str, Any] = {
        "config": config,
        "results": merged_results,
        "raw_eval_results": raw_eval_results,
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": first_trace.get("git_info"),
    }

    # Output
    output_path = args.output or (TRACES_DIR / f"{run_id}_UPLOAD.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))

    log(f"Merged {len(unique_files)} traces into {output_path}", prefix)
    log(f"  Tasks: {len(merged_results['successful_tasks'])} success, {len(merged_results['failed_tasks'])} failed", prefix)
    log(f"  Raw logging entries: {len(raw_logging_results)}", prefix)
    return 0


# =============================================================================
# FIX APPLICATION
# =============================================================================

def cmd_fix(args: argparse.Namespace) -> int:
    """Apply fixes and re-run evaluation."""
    prefix = args.prefix or ""

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_corebench_fixes.py"),
        "--fixes-root", str(args.fixes_root or FIXES_DIR / DEFAULT_BENCHMARK),
        "--agent-dir", str(args.agent_dir or REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"),
        "--benchmark", args.benchmark or DEFAULT_BENCHMARK,
    ]

    if args.agent_args:
        cmd.extend(["--agent-args", args.agent_args])
    if args.docker:
        cmd.append("--docker")
    if args.task_id:
        for tid in args.task_id:
            cmd.extend(["--task-id", tid])
    if prefix:
        cmd.extend(["--prefix", prefix])
    if args.rubric_model:
        cmd.extend(["--rubric-model", args.rubric_model])
    if args.skip_rubrics:
        cmd.append("--skip-rubrics")

    log(f"Running fixes with prefix: {prefix}", prefix)
    log(f"Command: {' '.join(cmd)}", prefix)

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# =============================================================================
# INSPECTION
# =============================================================================

def cmd_inspect(args: argparse.Namespace) -> int:
    """Analyze failures and generate fix recommendations."""
    prefix = args.prefix or ""

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "item_fixer.py"),
        "--trace-file", str(args.trace_file),
        "--benchmark", args.benchmark or DEFAULT_BENCHMARK,
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.task_id:
        for tid in args.task_id:
            cmd.extend(["--task-id", tid])

    log(f"Inspecting failures from trace", prefix)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# =============================================================================
# FULL PIPELINE
# =============================================================================

def cmd_full(args: argparse.Namespace) -> int:
    """Run full pipeline: rubric -> inspect -> fix -> extract -> merge."""
    prefix = args.prefix
    if not prefix:
        log("--prefix is required for full pipeline")
        return 1

    log(f"Starting full pipeline with prefix: {prefix}", prefix)

    # Step 1: Rubric evaluation (if trace has raw_logging_results)
    if args.trace_file:
        trace_data = json.loads(Path(args.trace_file).read_text())
        if trace_data.get("raw_logging_results"):
            log("Step 1: Rubric evaluation", prefix)
            args_rubric = argparse.Namespace(
                prefix=prefix,
                trace_file=args.trace_file,
                rubrics_dir=str(RUBRICS_DIR),
                output_dir=str(REPO_ROOT / "rubrics_output"),
                model=args.model or "gpt-5.2",
                failed_only=True,
                max_tasks=args.max_tasks,
                base_url=args.base_url,
                api_key=args.api_key,
            )
            rc = cmd_rubric(args_rubric)
            if rc != 0:
                log("Rubric evaluation failed", prefix)
                return rc
        else:
            log("Trace has no raw_logging_results - skipping rubric (run extract first)", prefix)

    # Step 2: Apply fixes (if requested)
    if not args.skip_fix and args.run_fixes:
        log("Step 2: Applying fixes", prefix)
        args_fix = argparse.Namespace(
            prefix=prefix,
            fixes_root=args.fixes_root,
            agent_dir=args.agent_dir,
            agent_args=args.agent_args,
            benchmark=args.benchmark,
            docker=args.docker,
            task_id=args.task_id,
            rubric_model=args.rubric_model,
            skip_rubrics=args.skip_rubrics,
        )
        rc = cmd_fix(args_fix)
        if rc != 0:
            log("Fix application failed", prefix)
            return rc

    # Step 3: Extract from Weave (if project specified)
    if args.project and not args.skip_extract:
        log("Step 3: Extracting from Weave", prefix)
        args_extract = argparse.Namespace(
            prefix=prefix,
            project=args.project,
            include_costs=True,
            merge_input=None,
        )
        rc = cmd_extract(args_extract)
        if rc != 0:
            log("Weave extraction failed", prefix)
            return rc

    # Step 4: Merge traces
    if not args.skip_merge:
        log("Step 4: Merging traces", prefix)
        args_merge = argparse.Namespace(
            prefix=prefix,
            input_pattern=None,
            output=None,
            run_id=None,
        )
        rc = cmd_merge(args_merge)
        if rc != 0:
            log("Merge failed", prefix)
            return rc

    log(f"Pipeline complete for prefix: {prefix}", prefix)
    return 0


# =============================================================================
# CROSS-MODEL RUBRIC
# =============================================================================

def cmd_cross_rubric(args: argparse.Namespace) -> int:
    """Run cross-model rubric evaluation for better accuracy."""
    import subprocess

    prefix = args.prefix or ""
    traces = args.traces or []

    if not traces:
        # Find baseline traces
        import glob
        patterns = [
            f"{TRACES_DIR}/corebench_hard_hal_generalist_agent*_UPLOAD.json",
            f"{TRACES_DIR}/*baseline*.json",
        ]
        for pattern in patterns:
            traces.extend(glob.glob(pattern))
        traces = sorted(set(traces))[:4]  # Limit to 4

    if not traces:
        log("No traces found. Specify --traces or ensure baseline traces exist.", prefix)
        return 1

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "cross_model_rubric.py"),
        "--traces", *traces,
        "--model", args.model or "gpt-5.2",
        "--rubrics-dir", str(RUBRICS_DIR),
        "--output-dir", str(REPO_ROOT / "rubrics_output"),
    ]

    if prefix:
        cmd.extend(["--prefix", prefix])
    if args.failed_only:
        cmd.append("--failed-only")
    if args.max_tasks:
        cmd.extend(["--max-tasks", str(args.max_tasks)])
    if args.summary_only:
        cmd.append("--summary-only")
    if args.evaluate_model:
        cmd.extend(["--evaluate-model", args.evaluate_model])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])

    log(f"Running cross-model rubric evaluation", prefix)
    log(f"Traces: {len(traces)}", prefix)

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# =============================================================================
# STATUS
# =============================================================================

def cmd_status(args: argparse.Namespace) -> int:
    """Show status of traces and rubric evaluations."""
    prefix = args.prefix or ""

    log(f"Pipeline Status (prefix: '{prefix}' or all)", prefix)

    # Find traces
    if prefix:
        trace_pattern = f"{TRACES_DIR}/{prefix}*.json"
    else:
        trace_pattern = f"{TRACES_DIR}/*UPLOAD.json"

    traces = sorted(glob.glob(trace_pattern))
    log(f"Traces found: {len(traces)}", prefix)

    for trace_path in traces[-10:]:  # Show last 10
        p = Path(trace_path)
        data = json.loads(p.read_text())
        results = data.get("results", {})
        raw_logging = data.get("raw_logging_results", [])
        log(f"  {p.name}: {len(results.get('successful_tasks', []))} ok, "
            f"{len(results.get('failed_tasks', []))} fail, "
            f"{len(raw_logging)} log entries", prefix)

    # Find rubric outputs
    rubric_dirs = list((REPO_ROOT / "rubrics_output").glob("*/"))
    if prefix:
        rubric_dirs.extend(list((REPO_ROOT / f"rubrics_output_{prefix.rstrip('_')}").glob("*/")))

    for rd in rubric_dirs:
        csvs = list(rd.glob("*.csv"))
        if csvs:
            log(f"Rubric outputs in {rd.name}: {len(csvs)} files", prefix)

    # Find fixes
    fix_dirs = list((FIXES_DIR / DEFAULT_BENCHMARK).glob("capsule-*"))
    log(f"Fix packages: {len(fix_dirs)}", prefix)

    return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified HAL Agent Debug Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status
  python scripts/pipeline.py status --prefix orange

  # Run rubric evaluation
  python scripts/pipeline.py rubric --prefix orange --trace-file traces/baseline.json

  # Extract Weave traces
  python scripts/pipeline.py extract --prefix orange --project hal-agent-debug

  # Merge traces
  python scripts/pipeline.py merge --prefix orange

  # Apply fixes
  python scripts/pipeline.py fix --prefix orange --docker

  # Full pipeline
  python scripts/pipeline.py full --prefix orange --trace-file traces/baseline.json --project hal-agent-debug
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common(p):
        p.add_argument("--prefix", help="Prefix for this iteration (e.g., orange_, mango_)")

    # Status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    add_common(p_status)

    # Rubric
    p_rubric = subparsers.add_parser("rubric", help="Evaluate traces against rubrics")
    add_common(p_rubric)
    p_rubric.add_argument("--trace-file", required=True, help="Trace JSON file to evaluate")
    p_rubric.add_argument("--rubrics-dir", default=str(RUBRICS_DIR), help="Directory with rubric files")
    p_rubric.add_argument("--output-dir", default=str(REPO_ROOT / "rubrics_output"), help="Output directory")
    p_rubric.add_argument("--model", default="gpt-5.2", help="Model for rubric evaluation")
    p_rubric.add_argument("--failed-only", action="store_true", help="Only evaluate failed tasks")
    p_rubric.add_argument("--max-tasks", type=int, help="Limit number of tasks")
    p_rubric.add_argument("--base-url", help="OpenAI API base URL")
    p_rubric.add_argument("--api-key", default="dummy", help="OpenAI API key")

    # Extract
    p_extract = subparsers.add_parser("extract", help="Extract traces from Weave")
    add_common(p_extract)
    p_extract.add_argument("--project", required=True, help="Weave project name")
    p_extract.add_argument("--include-costs", action="store_true", default=True)
    p_extract.add_argument("--merge-input", action="append", help="Additional trace files to merge")

    # Merge
    p_merge = subparsers.add_parser("merge", help="Merge individual traces")
    add_common(p_merge)
    p_merge.add_argument("--input-pattern", action="append", help="Glob pattern for input traces")
    p_merge.add_argument("--output", type=Path, help="Output file path")
    p_merge.add_argument("--run-id", help="Override run_id in merged trace")

    # Fix
    p_fix = subparsers.add_parser("fix", help="Apply fixes and re-run evaluation")
    add_common(p_fix)
    p_fix.add_argument("--fixes-root", type=Path, help="Directory with fix packages")
    p_fix.add_argument("--agent-dir", type=Path, help="HAL agent directory")
    p_fix.add_argument("--agent-args", help="Agent args JSON file")
    p_fix.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark name")
    p_fix.add_argument("--docker", action="store_true", help="Run in Docker")
    p_fix.add_argument("--task-id", action="append", help="Specific task IDs to fix")
    p_fix.add_argument("--rubric-model", help="Model for rubric re-evaluation")
    p_fix.add_argument("--skip-rubrics", action="store_true", help="Skip rubric evaluation")

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Analyze failures and generate fixes")
    add_common(p_inspect)
    p_inspect.add_argument("--trace-file", required=True, help="Trace file to analyze")
    p_inspect.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark name")
    p_inspect.add_argument("--dry-run", action="store_true", help="Don't write fix files")
    p_inspect.add_argument("--task-id", action="append", help="Specific task IDs")

    # Cross-model rubric (recommended)
    p_cross = subparsers.add_parser("cross-rubric", help="Cross-model rubric evaluation (recommended)")
    add_common(p_cross)
    p_cross.add_argument("--traces", nargs="+", help="Trace files to analyze (auto-detects if not specified)")
    p_cross.add_argument("--model", default="gpt-5.2", help="Model for rubric evaluation")
    p_cross.add_argument("--failed-only", action="store_true", help="Only evaluate failed tasks")
    p_cross.add_argument("--max-tasks", type=int, help="Limit number of tasks")
    p_cross.add_argument("--summary-only", action="store_true", help="Only print cross-model summary")
    p_cross.add_argument("--evaluate-model", help="Only evaluate failures from this model")
    p_cross.add_argument("--base-url", help="OpenAI API base URL")

    # Full pipeline
    p_full = subparsers.add_parser("full", help="Run full pipeline")
    add_common(p_full)
    p_full.add_argument("--trace-file", help="Initial trace file")
    p_full.add_argument("--project", help="Weave project for extraction")
    p_full.add_argument("--model", default="gpt-5.2", help="Model for rubric evaluation")
    p_full.add_argument("--max-tasks", type=int, help="Limit tasks for rubric")
    p_full.add_argument("--base-url", help="OpenAI API base URL")
    p_full.add_argument("--api-key", default="dummy", help="OpenAI API key")
    p_full.add_argument("--run-fixes", action="store_true", help="Run fix application")
    p_full.add_argument("--skip-fix", action="store_true", help="Skip fix step")
    p_full.add_argument("--skip-extract", action="store_true", help="Skip Weave extraction")
    p_full.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    p_full.add_argument("--fixes-root", type=Path, help="Directory with fix packages")
    p_full.add_argument("--agent-dir", type=Path, help="HAL agent directory")
    p_full.add_argument("--agent-args", help="Agent args JSON file")
    p_full.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark name")
    p_full.add_argument("--docker", action="store_true", help="Run in Docker")
    p_full.add_argument("--task-id", action="append", help="Specific task IDs")
    p_full.add_argument("--rubric-model", help="Model for rubric re-evaluation")
    p_full.add_argument("--skip-rubrics", action="store_true", help="Skip rubric evaluation in fix")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "status": cmd_status,
        "rubric": cmd_rubric,
        "cross-rubric": cmd_cross_rubric,
        "extract": cmd_extract,
        "merge": cmd_merge,
        "fix": cmd_fix,
        "inspect": cmd_inspect,
        "full": cmd_full,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
