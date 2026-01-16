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
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

# Add rubric_evaluator to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Try to use docent via rubric_evaluator (preferred)
USE_DOCENT = False
try:
    from rubric_evaluator.cli import (
        _ensure_llm_cache_path,
        build_docent_agent_runs,
        evaluate_environmental_barrier,
        build_rubric_from_definition,
        resolve_rubric_model_option,
        RubricDefinition,
        TaskConversation,
        TraceMessage,
        extract_task_messages,
        group_entries_by_task,
    )
    _ensure_llm_cache_path()
    USE_DOCENT = True
except ImportError as e:
    print(f"Note: Docent integration not available ({e}). Using OpenAI client directly.")

from openai import OpenAI

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Thread-safe logging
_log_lock = Lock()

TRACES_DIR = REPO_ROOT / "traces"
RUBRICS_DIR = REPO_ROOT / "rubrics"
RUBRIC_TEMPLATES_DIR = REPO_ROOT / "rubric_templates"
CACHE_DIR = REPO_ROOT / "rubrics_cache"


# ──────────────────────────────────────────────────────────────────────────────
# Retry with backoff (from docent pattern)
# ──────────────────────────────────────────────────────────────────────────────

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 2.0, *args, **kwargs):
    """Retry a function with exponential backoff on connection/timeout errors."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout, HTTPError, Exception) as e:
            last_exception = e
            # Check if it's a retryable error
            error_str = str(e).lower()
            is_retryable = any(x in error_str for x in [
                'connection', 'timeout', 'rate limit', '429', '500', '502', '503', '504',
                'overloaded', 'capacity', 'busy', 'unavailable'
            ])
            if not is_retryable:
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            with _log_lock:
                print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise last_exception


# ──────────────────────────────────────────────────────────────────────────────
# Result caching
# ──────────────────────────────────────────────────────────────────────────────

class EvalCache:
    """Cache evaluation results to avoid re-running on restart."""

    def __init__(self, cache_dir: Path, prefix: str, rubric_name: str):
        self.cache_dir = cache_dir / f"{prefix}{rubric_name}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _cache_key(self, task_id: str, model: str) -> str:
        """Generate cache filename for a task/model combo."""
        key = f"{task_id}_{model}".replace("/", "_").replace(" ", "_")
        return hashlib.md5(key.encode()).hexdigest()[:16] + ".json"

    def get(self, task_id: str, model: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists."""
        cache_file = self.cache_dir / self._cache_key(task_id, model)
        if cache_file.exists():
            try:
                with cache_file.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set(self, task_id: str, model: str, result: Dict[str, Any]) -> None:
        """Cache a result."""
        cache_file = self.cache_dir / self._cache_key(task_id, model)
        with self._lock:
            try:
                with cache_file.open("w") as f:
                    json.dump(result, f, indent=2)
            except IOError:
                pass

    def get_cached_count(self) -> int:
        """Return number of cached results."""
        return len(list(self.cache_dir.glob("*.json")))


def log(msg: str, prefix: str = "") -> None:
    with _log_lock:
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


def format_conversation(entries: List[Dict], max_chars: int = 500000) -> str:
    """Format conversation entries for rubric evaluation.

    Args:
        entries: List of conversation entries from trace
        max_chars: Maximum total characters (default 500K, ~125K tokens)

    Strategy: Always include full task prompt, then prioritize HIGH-SIGNAL content
    (errors, exceptions, final attempts) over low-signal content.
    """
    # High-signal patterns that indicate important debugging info
    HIGH_SIGNAL_PATTERNS = [
        r'error', r'exception', r'traceback', r'failed', r'failure',
        r'nameerror', r'syntaxerror', r'typeerror', r'valueerror', r'keyerror',
        r'importerror', r'modulenotfounderror', r'attributeerror', r'indexerror',
        r'permission denied', r'not found', r'not defined', r'cannot', r'unable',
        r'exit code', r'returned non-zero', r'timed out', r'timeout',
        r'assert', r'mismatch', r'unexpected', r'invalid', r'malformed',
        r'convergence', r'bracket', r'overflow', r'underflow', r'nan', r'inf',
        r'shape', r'dimension', r'broadcast', r'dtype',
    ]
    high_signal_re = re.compile('|'.join(HIGH_SIGNAL_PATTERNS), re.IGNORECASE)

    lines = []
    task_prompt = None

    # First pass: extract full task prompt (first user message with substantial content)
    for entry in entries[:1]:
        messages = entry.get("inputs", {}).get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            if role == "user" and content and len(content) > 500:
                task_prompt = f"[TASK PROMPT]: {content}"
                break

    if task_prompt:
        lines.append(task_prompt)

    # Second pass: collect all conversation turns with signal scoring
    conversation_turns = []  # List of (content, is_high_signal, turn_index)
    turn_idx = 0

    for entry in entries:
        messages = entry.get("inputs", {}).get("messages", [])

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            if content:
                # Skip the task prompt we already captured
                if task_prompt and content in task_prompt:
                    continue
                formatted = f"[{role}]: {content}"
                is_high_signal = bool(high_signal_re.search(content))
                conversation_turns.append((formatted, is_high_signal, turn_idx))
                turn_idx += 1

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
                formatted = f"[assistant]: {content}"
                is_high_signal = bool(high_signal_re.search(content))
                conversation_turns.append((formatted, is_high_signal, turn_idx))
                turn_idx += 1

    # Calculate remaining budget after task prompt
    task_prompt_len = len(task_prompt) if task_prompt else 0
    remaining_budget = max_chars - task_prompt_len - 100

    # Check if everything fits
    total_len = sum(len(t[0]) for t in conversation_turns)

    if total_len <= remaining_budget:
        # Everything fits - include all
        lines.extend([t[0] for t in conversation_turns])
    else:
        # Need to be selective - prioritize:
        # 1. First 3 turns (initial approach)
        # 2. Last 5 turns (final state)
        # 3. All high-signal turns (errors, exceptions)
        # 4. Fill remaining budget with other turns

        n = len(conversation_turns)
        must_include = set()

        # Always include first 3 and last 5
        for i in range(min(3, n)):
            must_include.add(i)
        for i in range(max(0, n - 5), n):
            must_include.add(i)

        # Always include high-signal turns
        for i, (content, is_high_signal, _) in enumerate(conversation_turns):
            if is_high_signal:
                must_include.add(i)

        # Build result with must-include turns
        selected_indices = sorted(must_include)
        selected = []
        current_len = 0
        last_included = -1

        for i in selected_indices:
            content = conversation_turns[i][0]

            # Add gap marker if we skipped turns
            if last_included >= 0 and i > last_included + 1:
                gap = i - last_included - 1
                gap_marker = f"\n[... {gap} turns omitted ...]\n"
                selected.append(gap_marker)
                current_len += len(gap_marker)

            # Check if we have budget
            if current_len + len(content) <= remaining_budget:
                selected.append(content)
                current_len += len(content)
                last_included = i
            else:
                # Budget exhausted - add truncation marker and stop
                selected.append("\n[... TRUNCATED due to length ...]\n")
                break

        lines.extend(selected)

    return "\n\n".join(lines)


def evaluate_with_cross_model_context(
    client: OpenAI,
    model: str,
    rubric_text: str,
    task_id: str,
    task_summary: TaskSummary,
    conversation: str,
    evaluated_model: str,
    successful_conversation: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single task with cross-model context."""

    # Build the prompt with cross-model context
    cross_model_context = task_summary.to_context_string()

    # Add successful model's approach if available
    success_context = ""
    if successful_conversation and task_summary.any_succeeded:
        success_context = f"""
=== HOW A SUCCESSFUL MODEL APPROACHED THIS TASK ===
One of the successful models ({task_summary.models_succeeded[0]}) used this approach:
{successful_conversation[:8000]}

IMPORTANT: Analyze whether this successful approach represents:
A) A LEGITIMATE alternative path that any agent could use (suggests capability issue)
B) A lucky workaround or circumvention that wouldn't reliably work (might still be env barrier)
C) Random variation/non-deterministic success (inconclusive)
"""

    prompt = f"""You are evaluating an agent transcript against the following rubric, WITH CROSS-MODEL CONTEXT.

RUBRIC:
{rubric_text}

{cross_model_context}
{success_context}

=== TRANSCRIPT FOR {evaluated_model} ON TASK {task_id} ===
{conversation}

=== NUANCED EVALUATION INSTRUCTIONS ===

You must make a CAREFUL determination about environmental barriers vs capability issues.

**IF ANOTHER MODEL SUCCEEDED:**
Do NOT automatically assume score=0. Instead, analyze:
1. HOW did the successful model solve it? What specific approach/commands did it use?
2. Is this approach GENERALLY AVAILABLE to other agents? (legitimate alternative path)
3. Or was it LUCK/CIRCUMVENTION? (random success, timing, non-deterministic behavior)
4. Could a well-designed agent reasonably discover and use this approach?

If the success was through a legitimate, reproducible approach → score=0 (capability issue)
If the success appears to be luck/circumvention/non-reproducible → still might be env barrier

**IF ALL MODELS FAILED:**
Do NOT automatically assume env barrier. Instead, analyze:
1. What approaches did the models try?
2. Are there UNTRIED approaches that could plausibly work?
3. Is the failure due to a FUNDAMENTAL infrastructure limitation (no R installed, permissions blocked)?
4. Or is it due to models not being clever enough to find a working approach?

Only score=1 if there is truly NO viable path regardless of agent skill.

**SCORING CRITERIA:**
- score=1 (Environmental Barrier): Infrastructure makes task MECHANICALLY IMPOSSIBLE
  - Missing runtime (R/Python) that cannot be installed
  - Permission denied on required operations with no workaround
  - Missing benchmark data that should have been provided
  - Sandbox restrictions with no alternative path

- score=0 (Capability Issue): Agent COULD succeed with better approach
  - A working approach exists (even if models didn't find it)
  - Package installation possible but agent used wrong method
  - File exists but agent looked in wrong place
  - Tool limitations have workarounds the agent didn't try

Provide your response as a JSON object with:
- "score": 0 or 1
- "explanation": detailed explanation citing specific evidence
- "cross_model_reasoning": how you weighed the cross-model evidence
- "success_analysis": (if applicable) analysis of how/why another model succeeded
- "alternative_paths": what approaches could potentially work but weren't tried

Respond with ONLY the JSON object, no other text."""

    def _make_api_call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=15000,
        )

    try:
        # Use retry with backoff for resilience
        response = retry_with_backoff(_make_api_call, max_retries=3, base_delay=2.0)
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
    parser.add_argument("--rubric", help="Path to rubric template file (e.g., rubric_templates/scicode.txt)")
    parser.add_argument("--rubrics-dir", default=str(RUBRICS_DIR), help="Rubrics directory (legacy, use --rubric instead)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "rubrics_output"), help="Output directory")
    parser.add_argument("--base-url", help="OpenAI API base URL")
    parser.add_argument("--api-key", default="dummy", help="OpenAI API key")
    parser.add_argument("--evaluate-model", help="Only evaluate failures from this model")
    parser.add_argument("--max-tasks", type=int, help="Limit number of tasks")
    parser.add_argument("--parallel", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--max-context", type=int, default=500000, help="Max characters per conversation (default: 500000, ~125K tokens)")
    parser.add_argument("--failed-only", action="store_true", help="Only evaluate failed tasks")
    parser.add_argument("--summary-only", action="store_true", help="Only print cross-model summary")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (re-evaluate all tasks)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    args = parser.parse_args()

    prefix = args.prefix

    # Load rubric - prefer --rubric argument, fall back to --rubrics-dir
    if args.rubric:
        rubric_path = Path(args.rubric)
        if not rubric_path.exists():
            # Try relative to repo root
            rubric_path = REPO_ROOT / args.rubric
        if not rubric_path.exists():
            log(f"Rubric file not found: {args.rubric}", prefix)
            return 1
        rubric_text = rubric_path.read_text()
        rubric_name = rubric_path.stem
        log(f"Loaded rubric: {rubric_name} (from {rubric_path})", prefix)
    else:
        # Legacy: search in rubrics directory
        rubric_files = list(Path(args.rubrics_dir).glob("*.txt"))
        if not rubric_files:
            log(f"No rubric files found in {args.rubrics_dir}", prefix)
            log(f"Tip: Use --rubric to specify a rubric template, e.g.:", prefix)
            log(f"  --rubric rubric_templates/corebench.txt", prefix)
            log(f"  --rubric rubric_templates/scicode.txt", prefix)
            return 1
        rubric_text = rubric_files[0].read_text()
        rubric_name = rubric_files[0].stem
        log(f"Loaded rubric: {rubric_name}", prefix)

    log(f"Max context per task: {args.max_context:,} chars (~{args.max_context // 4:,} tokens)", prefix)

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

    # Build a map to find successful model's conversation for a task
    def get_successful_conversation(task_id: str) -> Optional[str]:
        """Get conversation from a model that succeeded at this task."""
        summary = task_summaries.get(task_id)
        if not summary or not summary.any_succeeded:
            return None

        # Find a trace that succeeded
        for t in model_traces:
            if task_id in t.successful_tasks and task_id in t.task_conversations:
                return format_conversation(t.task_conversations[task_id], max_chars=args.max_context)
        return None

    # Prepare evaluation tasks
    eval_tasks: List[Tuple[str, ModelTrace, TaskSummary, str, Optional[str]]] = []

    for trace in evaluate_traces:
        log(f"\nPreparing tasks for model: {trace.model_name}", prefix)

        # Get tasks to evaluate
        if args.failed_only:
            task_ids = list(trace.failed_tasks)
        else:
            task_ids = list(trace.failed_tasks | trace.successful_tasks)

        if args.max_tasks:
            task_ids = task_ids[:args.max_tasks]

        for task_id in task_ids:
            # Get cross-model summary
            summary = task_summaries.get(task_id)
            if not summary:
                continue

            # Get the evaluated model's conversation
            conversation = ""
            if task_id in trace.task_conversations:
                conversation = format_conversation(trace.task_conversations[task_id], max_chars=args.max_context)

            if not conversation:
                continue

            # Get successful model's conversation for comparison (if any model succeeded)
            successful_conv = None
            if summary.any_succeeded and task_id in trace.failed_tasks:
                successful_conv = get_successful_conversation(task_id)

            eval_tasks.append((task_id, trace, summary, conversation, successful_conv))

    log(f"\nEvaluating {len(eval_tasks)} tasks with {args.parallel} parallel workers", prefix)

    # Worker function for parallel evaluation
    def evaluate_task(task_tuple: Tuple[str, ModelTrace, TaskSummary, str, Optional[str]], idx: int) -> Dict[str, Any]:
        task_id, trace, summary, conversation, successful_conv = task_tuple
        log(f"  [{idx+1}/{len(eval_tasks)}] {task_id} ({trace.model_name})", prefix)

        result = evaluate_with_cross_model_context(
            client=client,
            model=args.model,
            rubric_text=rubric_text,
            task_id=task_id,
            task_summary=summary,
            conversation=conversation,
            evaluated_model=trace.model_name,
            successful_conversation=successful_conv,
        )

        log(f"  [{idx+1}/{len(eval_tasks)}] {task_id} → Score: {result['score']}", prefix)

        # Verbose output: print grade and reasoning
        if args.failed_only:
            log(f"\n{'='*80}", prefix)
            log(f"TASK: {task_id} | MODEL: {trace.model_name}", prefix)
            log(f"GRADE: {result['score']}", prefix)
            log(f"ANY MODEL SUCCEEDED: {result.get('any_model_succeeded', False)}", prefix)
            if result.get('models_succeeded'):
                log(f"MODELS SUCCEEDED: {', '.join(result.get('models_succeeded', []))}", prefix)
            log(f"\nEXPLANATION:\n{result.get('explanation', 'N/A')}", prefix)
            log(f"\nCROSS-MODEL REASONING:\n{result.get('cross_model_reasoning', 'N/A')}", prefix)
            if result.get('error'):
                log(f"\nERROR: {result.get('error')}", prefix)
            log(f"{'='*80}\n", prefix)

        return result

    # Run evaluations in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(evaluate_task, task, i): i
            for i, task in enumerate(eval_tasks)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                idx = futures[future]
                log(f"  [ERROR] Task {idx} failed: {e}", prefix)

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
