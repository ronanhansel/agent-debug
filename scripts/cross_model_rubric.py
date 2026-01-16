#!/usr/bin/env python3
"""
Cross-Model Rubric Evaluation

This script evaluates environmental barriers by considering ALL models' performance
on each task. Key insight: if ANY model succeeded at a task, failures by other
models are capability issues, not environmental barriers.

Two-phase approach:
1. Build task success map across all models
2. Evaluate with cross-model context

Usage:
    python scripts/cross_model_rubric.py \
        --traces traces/baseline_gpt41.json traces/baseline_o3.json traces/baseline_o4mini_high.json \
        --prefix orange \
        --model gpt-5.2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "traces"
RUBRICS_DIR = REPO_ROOT / "rubrics"


def log(msg: str, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)


@dataclass
class TaskSummary:
    """Summary of a task's performance across all models."""
    task_id: str
    models_succeeded: List[str] = field(default_factory=list)
    models_failed: List[str] = field(default_factory=list)
    error_patterns: Dict[str, List[str]] = field(default_factory=dict)  # model -> errors
    approaches_tried: Dict[str, List[str]] = field(default_factory=dict)  # model -> approaches

    @property
    def any_succeeded(self) -> bool:
        return len(self.models_succeeded) > 0

    @property
    def all_failed(self) -> bool:
        return len(self.models_succeeded) == 0 and len(self.models_failed) > 0

    def to_context_string(self) -> str:
        """Generate context string for rubric evaluation."""
        lines = [f"=== CROSS-MODEL SUMMARY FOR {self.task_id} ==="]

        if self.any_succeeded:
            lines.append(f"\n** IMPORTANT: {len(self.models_succeeded)} model(s) SUCCEEDED at this task **")
            lines.append(f"Successful models: {', '.join(self.models_succeeded)}")
            lines.append("This indicates the task IS SOLVABLE and has NO environmental barrier.")
            lines.append("Any failure by the evaluated model is a CAPABILITY ISSUE (score=0).")
        else:
            lines.append(f"\n** ALL {len(self.models_failed)} models FAILED at this task **")
            lines.append("This MAY indicate an environmental barrier, but requires careful analysis.")

        lines.append(f"\nModels that failed: {', '.join(self.models_failed)}")

        # Summarize error patterns
        if self.error_patterns:
            lines.append("\n--- Error Patterns by Model ---")
            for model, errors in self.error_patterns.items():
                if errors:
                    lines.append(f"\n{model}:")
                    for err in errors[:5]:  # Limit to 5 errors per model
                        lines.append(f"  - {err[:200]}")

        # Summarize approaches
        if self.approaches_tried:
            lines.append("\n--- Approaches Tried ---")
            for model, approaches in self.approaches_tried.items():
                if approaches:
                    lines.append(f"\n{model}: {', '.join(approaches[:5])}")

        return "\n".join(lines)


@dataclass
class ModelTrace:
    """Parsed trace for a single model."""
    model_name: str
    trace_path: Path
    successful_tasks: Set[str]
    failed_tasks: Set[str]
    task_conversations: Dict[str, List[Dict]]  # task_id -> conversation entries
    task_errors: Dict[str, List[str]]  # task_id -> error messages


def extract_model_name(trace_data: Dict[str, Any], trace_path: Path) -> str:
    """Extract model name from trace config or filename."""
    config = trace_data.get("config", {})
    agent_args = config.get("agent_args", {})

    model_name = agent_args.get("model_name", "")
    if model_name:
        # Clean up model name
        model_name = model_name.replace("openai/", "").replace("-2025", "").replace("_", "-")
        return model_name

    # Fall back to filename parsing
    name = trace_path.stem
    for pattern in ["gpt-4_1", "gpt-4.1", "gpt41", "o3", "o4-mini", "gpt-5"]:
        if pattern in name.lower():
            return pattern
    return name[:30]


def extract_errors_from_conversation(entries: List[Dict]) -> List[str]:
    """Extract error patterns from conversation entries."""
    errors = []
    error_patterns = [
        r"Error:.*",
        r"ModuleNotFoundError:.*",
        r"FileNotFoundError:.*",
        r"PermissionError:.*",
        r"Exit Code: [1-9]\d*",
        r"not found",
        r"Permission denied",
        r"cannot access",
        r"No such file",
        r"ImportError:.*",
        r"CondaToSNonInteractiveError.*",
    ]

    for entry in entries:
        # Check output
        output = entry.get("output", {})
        choices = output.get("choices", [])
        for choice in choices:
            content = choice.get("message", {}).get("content", "")
            if isinstance(content, str):
                for pattern in error_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    errors.extend(matches[:3])  # Limit per pattern

    # Dedupe while preserving order
    seen = set()
    unique_errors = []
    for err in errors:
        err_key = err[:100].lower()
        if err_key not in seen:
            seen.add(err_key)
            unique_errors.append(err)

    return unique_errors[:10]  # Limit total errors


def extract_approaches_from_conversation(entries: List[Dict]) -> List[str]:
    """Extract high-level approaches tried from conversation."""
    approaches = set()
    approach_indicators = {
        "pip install": "pip_install",
        "conda install": "conda_install",
        "apt-get": "apt_install",
        "Rscript": "r_execution",
        "python": "python_execution",
        "jupyter": "notebook_execution",
        "nbconvert": "notebook_conversion",
        "rmarkdown::render": "rmarkdown_render",
        "symlink": "symlink_creation",
        "mkdir": "directory_creation",
        "chmod": "permission_change",
        "wget": "download",
        "curl": "download",
        "git clone": "git_clone",
    }

    for entry in entries:
        output = entry.get("output", {})
        choices = output.get("choices", [])
        for choice in choices:
            content = choice.get("message", {}).get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                for indicator, approach in approach_indicators.items():
                    if indicator.lower() in content_lower:
                        approaches.add(approach)

    return list(approaches)


def load_model_trace(trace_path: Path) -> ModelTrace:
    """Load and parse a single model trace."""
    data = json.loads(trace_path.read_text())

    model_name = extract_model_name(data, trace_path)
    results = data.get("results", {})
    successful_tasks = set(results.get("successful_tasks", []))
    failed_tasks = set(results.get("failed_tasks", []))

    # Group conversations by task
    raw_entries = data.get("raw_logging_results", [])
    task_conversations: Dict[str, List[Dict]] = defaultdict(list)

    for entry in raw_entries:
        task_id = (
            entry.get("attributes", {}).get("weave_task_id")
            or entry.get("weave_task_id")
            or "unknown"
        )
        task_conversations[task_id].append(entry)

    # Extract errors per task
    task_errors: Dict[str, List[str]] = {}
    for task_id, entries in task_conversations.items():
        task_errors[task_id] = extract_errors_from_conversation(entries)

    return ModelTrace(
        model_name=model_name,
        trace_path=trace_path,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        task_conversations=dict(task_conversations),
        task_errors=task_errors,
    )


def build_task_summaries(model_traces: List[ModelTrace]) -> Dict[str, TaskSummary]:
    """Build cross-model summaries for all tasks."""
    # Collect all task IDs
    all_tasks: Set[str] = set()
    for trace in model_traces:
        all_tasks.update(trace.successful_tasks)
        all_tasks.update(trace.failed_tasks)

    summaries: Dict[str, TaskSummary] = {}

    for task_id in all_tasks:
        summary = TaskSummary(task_id=task_id)

        for trace in model_traces:
            if task_id in trace.successful_tasks:
                summary.models_succeeded.append(trace.model_name)
            elif task_id in trace.failed_tasks:
                summary.models_failed.append(trace.model_name)

                # Add error patterns
                if task_id in trace.task_errors:
                    summary.error_patterns[trace.model_name] = trace.task_errors[task_id]

                # Add approaches tried
                if task_id in trace.task_conversations:
                    approaches = extract_approaches_from_conversation(
                        trace.task_conversations[task_id]
                    )
                    summary.approaches_tried[trace.model_name] = approaches

        summaries[task_id] = summary

    return summaries


def format_conversation(entries: List[Dict], max_chars: int = 50000) -> str:
    """Format conversation entries for rubric evaluation."""
    lines = []
    total_chars = 0

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
                line = f"[{role}]: {content[:1500]}"
                if total_chars + len(line) > max_chars:
                    break
                lines.append(line)
                total_chars += len(line)

        output = entry.get("output", {})
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
                line = f"[assistant]: {content[:1500]}"
                if total_chars + len(line) > max_chars:
                    break
                lines.append(line)
                total_chars += len(line)

        if total_chars > max_chars:
            break

    return "\n\n".join(lines[-100:])


def evaluate_with_cross_model_context(
    client: OpenAI,
    model: str,
    rubric_text: str,
    task_id: str,
    task_summary: TaskSummary,
    conversation: str,
    evaluated_model: str,
) -> Dict[str, Any]:
    """Evaluate a single task with cross-model context."""

    # Build the prompt with cross-model context
    cross_model_context = task_summary.to_context_string()

    prompt = f"""You are evaluating an agent transcript against the following rubric, WITH ADDITIONAL CROSS-MODEL CONTEXT.

RUBRIC:
{rubric_text}

{cross_model_context}

=== TRANSCRIPT FOR {evaluated_model} ON TASK {task_id} ===
{conversation}

=== EVALUATION INSTRUCTIONS ===

Based on the rubric criteria AND the cross-model context, evaluate this transcript.

CRITICAL RULE: If ANY model succeeded at this task (see cross-model summary above), then this task has NO environmental barrier. The failure MUST be scored as 0 (capability issue).

Only score 1 (environmental barrier) if:
1. ALL models failed at this task, AND
2. The failures show evidence of an IMPASSABLE infrastructure defect that no agent could overcome

Provide your response as a JSON object with:
- "score": 0 or 1
- "explanation": detailed explanation citing evidence
- "cross_model_reasoning": how the cross-model context influenced your decision

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
        return {
            "task_id": task_id,
            "model": evaluated_model,
            "score": float(result.get("score", 0)),
            "explanation": result.get("explanation", ""),
            "cross_model_reasoning": result.get("cross_model_reasoning", ""),
            "any_model_succeeded": task_summary.any_succeeded,
            "models_succeeded": task_summary.models_succeeded,
            "models_failed": task_summary.models_failed,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "model": evaluated_model,
            "score": 0,
            "explanation": "",
            "cross_model_reasoning": "",
            "error": str(e),
            "any_model_succeeded": task_summary.any_succeeded,
            "models_succeeded": task_summary.models_succeeded,
            "models_failed": task_summary.models_failed,
        }


def main():
    parser = argparse.ArgumentParser(description="Cross-model rubric evaluation")
    parser.add_argument("--traces", nargs="+", required=True, help="Trace files to analyze")
    parser.add_argument("--prefix", default="", help="Output prefix")
    parser.add_argument("--model", default="gpt-5.2", help="Model for rubric evaluation")
    parser.add_argument("--rubrics-dir", default=str(RUBRICS_DIR), help="Rubrics directory")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "rubrics_output"), help="Output directory")
    parser.add_argument("--base-url", help="OpenAI API base URL")
    parser.add_argument("--api-key", default="dummy", help="OpenAI API key")
    parser.add_argument("--evaluate-model", help="Only evaluate failures from this model")
    parser.add_argument("--max-tasks", type=int, help="Limit number of tasks")
    parser.add_argument("--failed-only", action="store_true", help="Only evaluate failed tasks")
    parser.add_argument("--summary-only", action="store_true", help="Only print cross-model summary")
    args = parser.parse_args()

    prefix = args.prefix

    # Load rubric
    rubric_files = list(Path(args.rubrics_dir).glob("*.txt"))
    if not rubric_files:
        log(f"No rubric files found in {args.rubrics_dir}", prefix)
        return 1
    rubric_text = rubric_files[0].read_text()
    rubric_name = rubric_files[0].stem
    log(f"Loaded rubric: {rubric_name}", prefix)

    # Load all model traces
    log(f"Loading {len(args.traces)} model traces...", prefix)
    model_traces: List[ModelTrace] = []
    for trace_path in args.traces:
        path = Path(trace_path)
        if not path.exists():
            log(f"Warning: Trace not found: {path}", prefix)
            continue
        trace = load_model_trace(path)
        log(f"  {trace.model_name}: {len(trace.successful_tasks)} success, {len(trace.failed_tasks)} failed, "
            f"{len(trace.task_conversations)} conversations", prefix)
        model_traces.append(trace)

    if not model_traces:
        log("No valid traces loaded", prefix)
        return 1

    # Build cross-model summaries
    log("Building cross-model task summaries...", prefix)
    task_summaries = build_task_summaries(model_traces)

    # Print summary statistics
    any_succeeded_count = sum(1 for s in task_summaries.values() if s.any_succeeded)
    all_failed_count = sum(1 for s in task_summaries.values() if s.all_failed)
    log(f"Task summary: {len(task_summaries)} total tasks", prefix)
    log(f"  - {any_succeeded_count} tasks solved by at least one model (NOT env barriers)", prefix)
    log(f"  - {all_failed_count} tasks failed by ALL models (potential env barriers)", prefix)

    if args.summary_only:
        log("\n=== POTENTIAL ENVIRONMENTAL BARRIERS (all models failed) ===", prefix)
        for task_id, summary in sorted(task_summaries.items()):
            if summary.all_failed:
                log(f"\n{task_id}:", prefix)
                for model, errors in summary.error_patterns.items():
                    if errors:
                        log(f"  {model}: {errors[0][:100]}...", prefix)
        return 0

    # Setup OpenAI client
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    # Determine which model(s) to evaluate
    if args.evaluate_model:
        evaluate_traces = [t for t in model_traces if args.evaluate_model in t.model_name]
    else:
        evaluate_traces = model_traces

    # Collect all results
    all_results = []

    for trace in evaluate_traces:
        log(f"\nEvaluating model: {trace.model_name}", prefix)

        # Get tasks to evaluate
        if args.failed_only:
            task_ids = list(trace.failed_tasks)
        else:
            task_ids = list(trace.failed_tasks | trace.successful_tasks)

        if args.max_tasks:
            task_ids = task_ids[:args.max_tasks]

        for i, task_id in enumerate(task_ids):
            log(f"  [{i+1}/{len(task_ids)}] {task_id}", prefix)

            # Get cross-model summary
            summary = task_summaries.get(task_id)
            if not summary:
                log(f"    No summary found, skipping", prefix)
                continue

            # Quick decision: if any model succeeded, this is NOT an env barrier
            if summary.any_succeeded and task_id in trace.failed_tasks:
                log(f"    Quick decision: score=0 (other models succeeded)", prefix)
                result = {
                    "task_id": task_id,
                    "model": trace.model_name,
                    "score": 0,
                    "explanation": f"Not an environmental barrier: {', '.join(summary.models_succeeded)} succeeded at this task.",
                    "cross_model_reasoning": "Task solvable by other models, so failure is capability issue.",
                    "any_model_succeeded": True,
                    "models_succeeded": summary.models_succeeded,
                    "models_failed": summary.models_failed,
                    "quick_decision": True,
                }
                all_results.append(result)
                continue

            # Full evaluation needed (all models failed or checking successful task)
            conversation = ""
            if task_id in trace.task_conversations:
                conversation = format_conversation(trace.task_conversations[task_id])

            if not conversation:
                log(f"    No conversation found, skipping", prefix)
                continue

            result = evaluate_with_cross_model_context(
                client=client,
                model=args.model,
                rubric_text=rubric_text,
                task_id=task_id,
                task_summary=summary,
                conversation=conversation,
                evaluated_model=trace.model_name,
            )

            log(f"    Score: {result['score']}", prefix)
            all_results.append(result)

    # Write results
    output_dir = Path(args.output_dir) / f"{rubric_name}_cross_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_label = f"{prefix}cross_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{output_label}.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "model", "criteria", "grade", "any_model_succeeded",
            "models_succeeded", "models_failed", "explanation", "cross_model_reasoning"
        ])

        for r in all_results:
            writer.writerow([
                r["task_id"],
                r["model"],
                rubric_name,
                f"{r['score']:.2f}",
                r.get("any_model_succeeded", False),
                ";".join(r.get("models_succeeded", [])),
                ";".join(r.get("models_failed", [])),
                r.get("explanation", ""),
                r.get("cross_model_reasoning", ""),
            ])

    # Summary
    env_barrier_count = sum(1 for r in all_results if r["score"] >= 1)
    quick_decisions = sum(1 for r in all_results if r.get("quick_decision"))

    log(f"\n=== SUMMARY ===", prefix)
    log(f"Total evaluated: {len(all_results)}", prefix)
    log(f"Environmental barriers: {env_barrier_count}", prefix)
    log(f"Capability issues: {len(all_results) - env_barrier_count}", prefix)
    log(f"Quick decisions (other model succeeded): {quick_decisions}", prefix)
    log(f"Results written to: {output_path}", prefix)

    return 0


if __name__ == "__main__":
    exit(main())
