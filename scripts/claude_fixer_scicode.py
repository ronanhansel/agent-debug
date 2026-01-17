#!/usr/bin/env python3
"""
Claude Code CLI-based SciCode Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
in SciCode benchmark tasks WITHOUT nerfing the scientific problems.

This script:
1. Gathers context about the task (rubric results, model conversations, task details)
2. Invokes Claude Code CLI with a detailed prompt
3. Claude Code analyzes and creates targeted fixes

Key principle: Fix INTRINSIC FORMATION ERRORS only, never simplify the science.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "traces"
FIXES_DIR = REPO_ROOT / "fixes"

# Thread-safe progress tracking
_progress_lock = Lock()
_completed_count = 0
_total_count = 0


def log(msg: str, prefix: str = "") -> None:
    with _progress_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        tag = f"[{prefix}] " if prefix else ""
        print(f"[{ts}] {tag}{msg}", flush=True)


def log_progress(task_id: str, status: str, prefix: str = "") -> None:
    """Thread-safe progress logging for parallel mode."""
    global _completed_count
    with _progress_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        if status == "completed":
            _completed_count += 1
            print(f"[{ts}] ✓ {task_id} DONE ({_completed_count}/{_total_count})", flush=True)
        elif status == "started":
            print(f"[{ts}] → {task_id} STARTED ({_completed_count}/{_total_count} done)", flush=True)
        elif status == "skipped":
            _completed_count += 1
            print(f"[{ts}] ⊘ {task_id} SKIPPED (already has fix) ({_completed_count}/{_total_count})", flush=True)
        elif status == "failed":
            _completed_count += 1
            print(f"[{ts}] ✗ {task_id} FAILED ({_completed_count}/{_total_count})", flush=True)


def has_existing_fix(benchmark: str, task_id: str) -> bool:
    """Check if a task already has a fix."""
    fix_dir = FIXES_DIR / benchmark / task_id
    if not fix_dir.exists():
        return False
    fix_files = ["env_override.json", "input_override.json", "README.md"]
    return any((fix_dir / f).exists() for f in fix_files)


def load_rubric_results(rubric_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load rubric evaluation results by task_id."""
    results = {}
    with rubric_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "")
            results[task_id] = {
                "score": float(row.get("grade", 0)),
                "explanation": row.get("explanation", ""),
                "model_run": row.get("model_run", ""),
            }
    return results


def load_judge_verdicts(verdict_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load judge verdicts by task_id."""
    results = {}
    if not verdict_csv.exists():
        return results
    with verdict_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "")
            results[task_id] = {
                "final_grade": float(row.get("final_grade", 0)),
                "satisfies_rubric": row.get("satisfies_rubric", "0") == "1",
                "reasoning": row.get("reasoning", ""),
                "num_evaluations": int(row.get("num_evaluations", 0)),
                "model_runs": row.get("model_runs", "").split(";"),
            }
    return results


def load_task_conversations(trace_files: List[Path], task_id: str) -> Dict[str, str]:
    """Load conversations for a task from multiple trace files."""
    conversations = {}

    for trace_path in trace_files:
        if not trace_path.exists():
            continue

        try:
            data = json.loads(trace_path.read_text())
        except:
            continue

        # Extract model name
        model_name = data.get("config", {}).get("agent_args", {}).get("model_name", trace_path.stem[:30])
        model_name = model_name.replace("openai/", "").replace("-2025", "")

        # Find conversation for this task
        raw_entries = data.get("raw_logging_results", [])
        task_entries = []

        for entry in raw_entries:
            entry_task = (
                entry.get("attributes", {}).get("weave_task_id")
                or entry.get("weave_task_id")
                or ""
            )
            if entry_task == task_id:
                task_entries.append(entry)

        if task_entries:
            # Format conversation
            lines = []
            for entry in task_entries[:30]:  # Limit entries
                messages = entry.get("inputs", {}).get("messages", [])
                for msg in messages[-3:]:  # Last 3 messages per entry
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    if content:
                        lines.append(f"[{role}]: {content[:2000]}")

                # Assistant output
                output = entry.get("output", {})
                choices = output.get("choices", [])
                if choices:
                    out_content = choices[0].get("message", {}).get("content", "")
                    if isinstance(out_content, list):
                        out_content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in out_content
                        )
                    if out_content:
                        lines.append(f"[assistant]: {out_content[:2000]}")

            conversations[model_name] = "\n\n".join(lines[-50:])  # Last 50 exchanges

    return conversations


def load_all_rubric_evaluations(rubric_dir: Path, task_id: str) -> List[Dict[str, Any]]:
    """Load all rubric evaluations for a task from all CSV files in the rubric directory."""
    evaluations = []
    for csv_file in rubric_dir.glob("*.csv"):
        with csv_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("task_id") == task_id:
                    evaluations.append({
                        "source": csv_file.stem,
                        "grade": row.get("grade", ""),
                        "explanation": row.get("explanation", ""),
                        "model_run": row.get("model_run", ""),
                    })
    return evaluations


def build_claude_prompt_single(
    task_id: str,
    evaluations: List[Dict[str, Any]],
    judge_verdict: Optional[Dict[str, Any]],
    conversations: Dict[str, str],
) -> str:
    """Build prompt section for a single task."""

    # Format evaluations section
    eval_sections = []
    for i, ev in enumerate(evaluations, 1):
        eval_sections.append(f"""
### Evaluation {i} (from {ev.get('model_run', 'unknown')})
- **Grade**: {ev.get('grade', 'N/A')} (1.0 = IFE detected, 0.0 = no IFE)
- **Explanation**: {ev.get('explanation', 'N/A')[:2000]}
""")
    evaluations_text = "\n".join(eval_sections) if eval_sections else "No evaluations available."

    # Format judge verdict
    judge_text = "No judge verdict available."
    if judge_verdict:
        judge_text = f"""
**Final Grade**: {judge_verdict.get('final_grade', 'N/A')}
**Satisfies Rubric (has IFE)**: {judge_verdict.get('satisfies_rubric', False)}
**Judge Reasoning**: {judge_verdict.get('reasoning', 'N/A')}
"""

    # Format conversations (abbreviated)
    conv_sections = []
    for model, conv in list(conversations.items())[:3]:  # Max 3 models per task
        conv_sections.append(f"#### {model}\n```\n{conv[:5000]}\n```")
    conversations_text = "\n".join(conv_sections) if conv_sections else "No logs."

    return f"""
---
## TASK: {task_id}
---

### Rubric Evaluations
{evaluations_text}

### Judge Verdict
{judge_text}

### Model Conversation Logs (abbreviated)
{conversations_text}
"""


def build_claude_prompt_batch(
    tasks_data: List[Dict[str, Any]],
    benchmark: str,
) -> str:
    """Build the prompt for Claude Code CLI with multiple tasks."""

    task_ids = [t['task_id'] for t in tasks_data]
    task_sections = []
    for t in tasks_data:
        section = build_claude_prompt_single(
            task_id=t['task_id'],
            evaluations=t['evaluations'],
            judge_verdict=t['judge_verdict'],
            conversations=t['conversations'],
        )
        task_sections.append(section)

    tasks_text = "\n".join(task_sections)

    prompt = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in SciCode benchmark tasks.

**YOU HAVE {len(task_ids)} TASKS TO PROCESS: {", ".join(task_ids)}**

Process each task sequentially, creating fixes as needed. You only need to read the HAL harness files ONCE at the start.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the scientific problem easier
2. **PRESERVE SCIENTIFIC RIGOR** - The task should remain as challenging as intended
3. **NO NERFING** - Do not simplify scientific concepts, give hints, reduce precision, or pre-compute results
4. **VALID FIXES**: Dependency constraints, ambiguous signatures, parsing issues, inconsistent instructions
5. **INVALID FIXES**: Solution hints, simplified physics/math, pre-importing modules

## HAL HARNESS STRUCTURE - READ ONCE AT START

**FIRST, read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/scicode.py` - Main benchmark class
- `hal-harness/hal/benchmarks/SciCode/` - Evaluation utilities

**How Evaluation Works:**
1. Agent produces code for each sub-step (e.g., task 11 has 11.1, 11.2, etc.)
2. HAL writes code to temp files, appending test cases from HuggingFace dataset
3. Test cases compare agent output against stored targets
4. Task passes if ALL sub-steps pass ALL test cases

**To inspect a specific task, run:**
```python
from datasets import load_dataset
dataset = load_dataset("SciCode1/SciCode", split="test")
task = [t for t in dataset if t['problem_id'] == 'TASK_ID'][0]
print(task['sub_steps'])  # See sub-steps, test_cases, required_dependencies
```

**Common IFE Sources:**
- `required_dependencies` whitelist missing necessary modules (e.g., scipy.constants)
- Test assertions too strict (floating point tolerance)
- Ambiguous function signatures between steps

**Additional Context (if needed):**
- Trace files with full agent conversations are in `traces/` directory
- Trace file names match the rubric CSV names (e.g., `scicode_hal_generalist_agent_gpt4120250414_*.json`)
- You can read these for detailed agent interactions if the rubric explanations aren't sufficient

## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/{benchmark}/TASK_ID/`
- `dependency_override.json` - Additional allowed dependencies
- `instruction_override.json` - Clarified instructions
- `evaluation_override.json` - Evaluation harness fixes
- `README.md` - Explanation of the fix

If NO fix needed (capability issue, not IFE), create README.md explaining why.

## TASKS TO PROCESS

{tasks_text}

## BEGIN

1. First, read `hal-harness/hal/benchmarks/scicode.py` to understand evaluation
2. For each task above:
   a. Load task from HuggingFace to see `required_dependencies` and `test_cases`
   b. Analyze the rubric evaluations and conversation logs
   c. Determine if IFE exists or if it's a capability issue
   d. Create appropriate fix (or document why no fix needed)
3. Create fixes in `fixes/{benchmark}/TASK_ID/` directories

Remember: Make evaluation FAIR, not EASY. Science must remain challenging.
'''

    return prompt


def format_stream_json(line: str, task_id: str) -> None:
    """Format and print a JSON stream line nicely."""
    try:
        data = json.loads(line)
        msg_type = data.get("type", "unknown")

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"

        ts = datetime.now().strftime("%H:%M:%S")

        if msg_type == "assistant":
            content = data.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        if "file_path" in tool_input:
                            path = tool_input.get("file_path", "")
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {path}")
                        else:
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {json.dumps(tool_input)[:200]}")
            elif isinstance(content, str) and content:
                print(f"{DIM}[{ts}]{RESET} {CYAN}[ASSISTANT]{RESET} {content[:500]}")

        elif msg_type == "user":
            content = data.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "")[:8]
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            preview = result_content[:300].replace("\n", " ")
                            print(f"{DIM}[{ts}]{RESET} {GREEN}[RESULT {tool_id}]{RESET} {preview}...")

        elif msg_type == "result":
            cost = data.get("cost_usd", 0)
            duration = data.get("duration_ms", 0) / 1000
            print(f"\n{BOLD}{GREEN}[COMPLETED]{RESET} Task: {task_id}")
            print(f"  Cost: ${cost:.4f} | Duration: {duration:.1f}s")

        elif msg_type == "error":
            error = data.get("error", {})
            print(f"{DIM}[{ts}]{RESET} {RED}[ERROR]{RESET} {error.get('message', str(error))}")

        elif msg_type == "system":
            msg = data.get("message", "")
            if msg:
                print(f"{DIM}[{ts}]{RESET} {BLUE}[SYSTEM]{RESET} {msg}")

    except json.JSONDecodeError:
        if line.strip():
            print(line.strip())


def run_claude_code(
    prompt: str,
    task_id: str,
    working_dir: Path,
    fix_dir: Path,
    quiet: bool = False,
) -> int:
    """Run Claude Code CLI with the given prompt."""

    base_cmd = [
        "claude",
        "--dangerously-skip-permissions",
    ]

    if not quiet:
        base_cmd.extend(["--verbose", "--output-format", "stream-json"])
    else:
        base_cmd.extend(["--output-format", "json"])

    base_cmd.extend(["-p", prompt])

    log_path = fix_dir / "claude_session.jsonl"

    if not quiet:
        log(f"Running Claude Code CLI for {task_id}...")
        log(f"Working directory: {working_dir}")
        print(f"\n{'='*60}")
        print(f"CLAUDE CODE SESSION: {task_id}")
        print(f"{'='*60}\n")

        process = subprocess.Popen(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            with log_path.open("w") as log_file:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    format_stream_json(line, task_id)
        except KeyboardInterrupt:
            process.terminate()
            log(f"Interrupted by user")
            return 130

        process.wait()
        log(f"Session log saved to: {log_path}")
        return process.returncode

    else:
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            capture_output=True,
            text=True,
        )

        with log_path.open("w") as log_file:
            log_file.write(result.stdout)
            if result.stderr:
                log_file.write(f"\n--- STDERR ---\n{result.stderr}")

        return result.returncode


def process_task_batch(
    task_ids: List[str],
    rubric_dir: Path,
    judge_verdicts: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    batch_id: int = 0,
    quiet: bool = False,
) -> List[Tuple[str, bool, str]]:
    """Process a batch of tasks in a single Claude session. Returns list of (task_id, success, message)."""

    try:
        # Gather data for all tasks in batch
        tasks_data = []
        for task_id in task_ids:
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations(trace_files, task_id)

            tasks_data.append({
                'task_id': task_id,
                'evaluations': evaluations,
                'judge_verdict': judge_verdict,
                'conversations': conversations,
            })

            # Create fix directory for each task
            fix_dir = FIXES_DIR / benchmark / task_id
            fix_dir.mkdir(parents=True, exist_ok=True)

        # Build batch prompt
        prompt = build_claude_prompt_batch(
            tasks_data=tasks_data,
            benchmark=benchmark,
        )

        # Save prompt to first task's directory
        batch_fix_dir = FIXES_DIR / benchmark / task_ids[0]
        (batch_fix_dir / "claude_prompt_batch.txt").write_text(prompt)

        if not quiet:
            log(f"Batch {batch_id}: Starting {len(task_ids)} tasks: {', '.join(task_ids)}")

        # Run Claude Code for entire batch
        exit_code = run_claude_code(
            prompt=prompt,
            task_id=f"batch_{batch_id}_{'-'.join(task_ids[:3])}",
            working_dir=REPO_ROOT,
            fix_dir=batch_fix_dir,
            quiet=quiet,
        )

        if exit_code == 0:
            if not quiet:
                log(f"Batch {batch_id}: Completed all {len(task_ids)} tasks")
            return [(tid, True, "Batch completed successfully") for tid in task_ids]
        else:
            if not quiet:
                log(f"Batch {batch_id}: Failed with exit code {exit_code}")
            return [(tid, False, f"Batch failed with code {exit_code}") for tid in task_ids]

    except Exception as e:
        log(f"Batch {batch_id}: Exception - {e}")
        return [(tid, False, str(e)) for tid in task_ids]


def process_single_task(
    task_id: str,
    rubric_dir: Path,
    judge_verdicts: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    quiet: bool = False,
) -> Tuple[str, bool, str]:
    """Process a single task (legacy, wraps batch function)."""
    results = process_task_batch(
        task_ids=[task_id],
        rubric_dir=rubric_dir,
        judge_verdicts=judge_verdicts,
        trace_files=trace_files,
        benchmark=benchmark,
        batch_id=0,
        quiet=quiet,
    )
    return results[0] if results else (task_id, False, "No result")


def main():
    parser = argparse.ArgumentParser(
        description="Use Claude Code to diagnose and fix SciCode IFEs"
    )

    parser.add_argument(
        "--rubric-dir",
        type=str,
        required=True,
        help="Directory containing rubric CSV outputs (e.g., rubrics_output/scicode)",
    )
    parser.add_argument(
        "--judge-csv",
        type=str,
        help="Path to judge verdict CSV (optional)",
    )
    parser.add_argument(
        "--trace-files",
        type=str,
        nargs="+",
        default=[],
        help="Trace files to extract conversations from (optional, provides additional context)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="scicode",
        help="Benchmark name (default: scicode)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="Specific task IDs to process (default: all with IFEs)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--min-grade",
        type=float,
        default=0.5,
        help="Minimum rubric grade to consider as IFE (default: 0.5)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have fixes",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel Claude Code sessions (default: 1)",
    )
    parser.add_argument(
        "--tasks-per-batch",
        type=int,
        default=5,
        help="Number of tasks per Claude session (default: 5). Each Claude instance processes this many tasks sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without running Claude Code",
    )

    args = parser.parse_args()

    # Resolve paths
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    trace_files = [Path(f) if Path(f).is_absolute() else REPO_ROOT / f for f in args.trace_files] if args.trace_files else []
    if trace_files:
        log(f"Using {len(trace_files)} trace files for conversation context")
    else:
        log("No trace files provided - using rubric evaluations only")

    # Load judge verdicts if provided
    judge_verdicts = {}
    if args.judge_csv:
        judge_path = Path(args.judge_csv)
        if not judge_path.is_absolute():
            judge_path = REPO_ROOT / judge_path
        judge_verdicts = load_judge_verdicts(judge_path)
        log(f"Loaded {len(judge_verdicts)} judge verdicts")

    # Determine tasks to process
    if args.task_ids:
        task_ids = args.task_ids
    else:
        # Find all tasks with IFEs based on rubric grades
        task_ids = set()
        for csv_file in rubric_dir.glob("*.csv"):
            with csv_file.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        grade = float(row.get("grade", 0) or 0)
                    except (ValueError, TypeError):
                        grade = 0
                    if grade >= args.min_grade:
                        task_ids.add(row.get("task_id", ""))
        task_ids = sorted(task_ids)
        log(f"Found {len(task_ids)} tasks with grade >= {args.min_grade}")

    # Filter existing fixes
    if args.skip_existing:
        original_count = len(task_ids)
        task_ids = [t for t in task_ids if not has_existing_fix(args.benchmark, t)]
        log(f"Skipped {original_count - len(task_ids)} tasks with existing fixes")

    # Limit tasks
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]

    log(f"Processing {len(task_ids)} tasks")

    # Create batches
    tasks_per_batch = args.tasks_per_batch
    batches = [task_ids[i:i + tasks_per_batch] for i in range(0, len(task_ids), tasks_per_batch)]
    num_batches = len(batches)

    log(f"Created {num_batches} batches of up to {tasks_per_batch} tasks each")

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Previewing batch prompts")
        print("="*60)

        # Preview first batch
        preview_batch = batches[0] if batches else []
        tasks_data = []
        for task_id in preview_batch:
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations(trace_files, task_id)
            tasks_data.append({
                'task_id': task_id,
                'evaluations': evaluations,
                'judge_verdict': judge_verdict,
                'conversations': conversations,
            })

        prompt = build_claude_prompt_batch(
            tasks_data=tasks_data,
            benchmark=args.benchmark,
        )

        print(f"\n{'='*60}")
        print(f"BATCH 1 of {num_batches}")
        print(f"Tasks: {', '.join(preview_batch)}")
        print(f"Prompt length: {len(prompt)} chars")
        print("="*60)
        print(prompt[:5000] + "..." if len(prompt) > 5000 else prompt)

        print(f"\n{'='*60}")
        print(f"DRY RUN SUMMARY")
        print(f"  Total tasks: {len(task_ids)}")
        print(f"  Batches: {num_batches}")
        print(f"  Tasks per batch: {tasks_per_batch}")
        print(f"  Parallel sessions: {args.parallel}")
        print(f"  Total Claude sessions: {num_batches}")
        print("="*60)
        return

    # Process batches
    global _total_count
    _total_count = len(task_ids)

    if args.parallel > 1:
        log(f"Running {args.parallel} parallel Claude sessions, {tasks_per_batch} tasks each")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_task_batch,
                    batch,
                    rubric_dir,
                    judge_verdicts,
                    trace_files,
                    args.benchmark,
                    batch_idx,
                    quiet=True,
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            all_results = []
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Summary
        successful = sum(1 for _, success, _ in all_results if success)
        log(f"\nCompleted: {successful}/{len(all_results)} tasks successful")
    else:
        # Sequential batch processing
        for batch_idx, batch in enumerate(batches):
            log(f"\n{'='*60}")
            log(f"BATCH {batch_idx + 1}/{num_batches}")
            log(f"Tasks: {', '.join(batch)}")
            log(f"{'='*60}")

            process_task_batch(
                task_ids=batch,
                rubric_dir=rubric_dir,
                judge_verdicts=judge_verdicts,
                trace_files=trace_files,
                benchmark=args.benchmark,
                batch_id=batch_idx,
                quiet=False,
            )


if __name__ == "__main__":
    main()
