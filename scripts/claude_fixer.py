#!/usr/bin/env python3
"""
Claude Code CLI-based Item Fixer

Uses Claude Code CLI (claude -p) to intelligently diagnose and fix environmental
barriers in CoreBench tasks WITHOUT nerfing the questions or making them easier.

This script:
1. Gathers context about the task (rubric results, model conversations, capsule details)
2. Invokes Claude Code CLI with a detailed prompt
3. Claude Code analyzes and creates targeted fixes

Key principle: Fix ENVIRONMENTAL BARRIERS only, never change the core challenge.
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
    # Check for actual fix files (not just the prompt)
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
                "cross_model_reasoning": row.get("cross_model_reasoning", ""),
                "models_succeeded": row.get("models_succeeded", "").split(";") if row.get("models_succeeded") else [],
                "models_failed": row.get("models_failed", "").split(";") if row.get("models_failed") else [],
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


def get_task_details(trace_files: List[Path], task_id: str) -> Dict[str, Any]:
    """Extract task details from trace files."""
    for trace_path in trace_files:
        if not trace_path.exists():
            continue
        try:
            data = json.loads(trace_path.read_text())
            # Check if this trace has info about the task
            if task_id in data.get("results", {}).get("failed_tasks", []):
                return {
                    "benchmark": data.get("config", {}).get("benchmark", "corebench_hard"),
                    "agent": data.get("config", {}).get("agent_name", "unknown"),
                }
            if task_id in data.get("results", {}).get("successful_tasks", []):
                return {
                    "benchmark": data.get("config", {}).get("benchmark", "corebench_hard"),
                    "agent": data.get("config", {}).get("agent_name", "unknown"),
                }
        except:
            continue
    return {"benchmark": "corebench_hard", "agent": "unknown"}


def build_claude_prompt(
    task_id: str,
    rubric_result: Dict[str, Any],
    conversations: Dict[str, str],
    task_details: Dict[str, Any],
    benchmark: str,
) -> str:
    """Build the prompt for Claude Code CLI."""

    # Format conversations section
    conv_sections = []
    for model, conv in conversations.items():
        conv_sections.append(f"### Model: {model}\n```\n{conv[:15000]}\n```")
    conversations_text = "\n\n".join(conv_sections) if conv_sections else "No conversation logs available."

    # Format rubric result
    rubric_text = f"""
Score: {rubric_result.get('score', 'N/A')} (1.0 = Environmental Barrier, 0.0 = Capability Issue)
Models that succeeded: {', '.join(rubric_result.get('models_succeeded', [])) or 'None'}
Models that failed: {', '.join(rubric_result.get('models_failed', [])) or 'None'}

Rubric Explanation:
{rubric_result.get('explanation', 'N/A')}

Cross-Model Reasoning:
{rubric_result.get('cross_model_reasoning', 'N/A')}
"""

    prompt = f'''You are diagnosing and fixing an environmental barrier in a CoreBench evaluation task.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX ENVIRONMENTAL BARRIERS ONLY** - Do NOT make the question easier or change its core challenge
2. **PRESERVE DIFFICULTY** - The task should remain as difficult as originally intended
3. **NO NERFING** - Do not:
   - Simplify the question
   - Give hints about the answer
   - Reduce computational requirements
   - Remove necessary steps
   - Pre-compute results for the agent
4. **VALID FIXES INCLUDE**:
   - Clarifying ambiguous instructions (without giving away answers)
   - Fixing missing environment dependencies that SHOULD be present
   - Correcting typos or unclear wording in task description
   - Ensuring required data files are accessible
   - Fixing permission issues that shouldn't exist
5. **INVALID FIXES** (DO NOT DO):
   - Adding answer hints
   - Pre-installing packages the agent should install itself
   - Simplifying the computational task
   - Reducing the scope of what needs to be done

## TASK INFORMATION

**Task ID**: {task_id}
**Benchmark**: {benchmark}

## RUBRIC EVALUATION RESULTS

{rubric_text}

## MODEL CONVERSATION LOGS

The following shows how different models attempted this task:

{conversations_text}

## YOUR MISSION

1. **DIAGNOSE**: Analyze the rubric results and conversation logs to understand:
   - What environmental barrier(s) exist?
   - Why did the models fail?
   - Is this truly an environmental issue or a capability issue?

2. **INVESTIGATE**:
   - Look at the capsule/task files if needed
   - Check what the task is actually asking
   - Understand what the correct environment setup should be

3. **DETERMINE FIX TYPE**:

   **Type A - Environment Fix** (env_override.json):
   - Missing system packages that SHOULD be pre-installed
   - Missing R runtime for R-based tasks
   - Broken conda environment
   - Permission issues

   **Type B - Prompt/Input Fix** (input_override.json):
   - Ambiguous or unclear task instructions
   - Missing context that agents need
   - Typos or errors in the question
   - Misleading wording

   **Type C - No Fix Needed**:
   - The failure is actually a capability issue (agent should have been able to solve it)
   - The task is working as intended

4. **CREATE FIX** (if needed):
   - Create the fix in: fixes/{benchmark}/{task_id}/
   - For env fixes: env_override.json with HAL_CONDA_PACKAGES, etc.
   - For input fixes: input_override.json with clarified instructions
   - Write a README.md explaining what was fixed and why

5. **DOCUMENT**:
   - Explain your diagnosis
   - Justify why this fix is appropriate
   - Confirm the fix does NOT nerf the question

## DIRECTORY STRUCTURE

```
fixes/
└── {benchmark}/
    └── {task_id}/
        ├── env_override.json      # Environment variables/packages
        ├── input_override.json    # Modified task instructions (if needed)
        └── README.md              # Explanation of the fix
```

## ENV_OVERRIDE.JSON FORMAT

```json
{{
  "HAL_CONDA_CHANNELS": "conda-forge",
  "HAL_CONDA_PACKAGES": "r-base r-essentials",
  "HAL_PIP_PACKAGES": "package1 package2",
  "HAL_APT_PACKAGES": "libfoo-dev"
}}
```

## INPUT_OVERRIDE.JSON FORMAT

```json
{{
  "clarifications": [
    "The output file should be named exactly 'results.csv'",
    "Use Python 3.10+ for this task"
  ],
  "corrected_instructions": "Optional: full corrected task description if needed"
}}
```

## BEGIN ANALYSIS

Start by reading the task files and understanding what this task requires.
Then diagnose the issue and create an appropriate fix (or determine no fix is needed).

Remember: Your goal is to make the evaluation FAIR, not EASY.
'''

    return prompt


def format_stream_json(line: str, task_id: str) -> None:
    """Format and print a JSON stream line nicely."""
    try:
        data = json.loads(line)
        msg_type = data.get("type", "unknown")

        # Color codes for terminal
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
            # Assistant message
            content = data.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")[:500]
                        print(f"{DIM}[{ts}]{RESET} {CYAN}[ASSISTANT]{RESET} {text}")
                    elif block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        # Format tool input nicely
                        if tool_name == "Bash":
                            cmd = tool_input.get("command", "")[:200]
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {cmd}")
                        elif tool_name in ("Read", "Glob", "Grep"):
                            path = tool_input.get("file_path", tool_input.get("pattern", tool_input.get("path", "")))
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {path}")
                        elif tool_name in ("Edit", "Write"):
                            path = tool_input.get("file_path", "")
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {path}")
                        else:
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {json.dumps(tool_input)[:200]}")
            elif isinstance(content, str) and content:
                print(f"{DIM}[{ts}]{RESET} {CYAN}[ASSISTANT]{RESET} {content[:500]}")

        elif msg_type == "user":
            # User/tool result
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
            # Final result
            cost = data.get("cost_usd", 0)
            duration = data.get("duration_ms", 0) / 1000
            print(f"\n{BOLD}{GREEN}[COMPLETED]{RESET} Task: {task_id}")
            print(f"  Cost: ${cost:.4f} | Duration: {duration:.1f}s")
            if data.get("result"):
                print(f"  Result: {str(data.get('result'))[:200]}")

        elif msg_type == "error":
            error = data.get("error", {})
            print(f"{DIM}[{ts}]{RESET} {RED}[ERROR]{RESET} {error.get('message', str(error))}")

        elif msg_type == "system":
            # System messages
            msg = data.get("message", "")
            if msg:
                print(f"{DIM}[{ts}]{RESET} {BLUE}[SYSTEM]{RESET} {msg}")

    except json.JSONDecodeError:
        # Not JSON, print as-is
        if line.strip():
            print(line.strip())


def run_claude_code(
    prompt: str,
    task_id: str,
    working_dir: Path,
    fix_dir: Path,
    resume_session: Optional[str] = None,
    quiet: bool = False,
) -> int:
    """Run Claude Code CLI with the given prompt.

    Args:
        quiet: If True, minimal output (for parallel mode). If False, stream formatted JSON.
    """

    # Build command
    base_cmd = [
        "claude",
        "--dangerously-skip-permissions",
    ]

    if not quiet:
        # Verbose streaming mode
        base_cmd.extend(["--verbose", "--output-format", "stream-json"])
    else:
        # Quiet mode - just JSON output at the end
        base_cmd.extend(["--output-format", "json"])

    if resume_session:
        base_cmd.extend(["--resume", resume_session])

    base_cmd.extend(["-p", prompt])

    # Prepare log file for raw JSON stream
    log_path = fix_dir / "claude_session.jsonl"

    if not quiet:
        log(f"Running Claude Code CLI for {task_id}...")
        log(f"Working directory: {working_dir}")
        print(f"\n{'='*60}")
        print(f"CLAUDE CODE SESSION: {task_id}")
        print(f"{'='*60}\n")

        # Run Claude Code with streaming output
        process = subprocess.Popen(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream, format, and save output
        try:
            with log_path.open("w") as log_file:
                for line in process.stdout:
                    # Save raw JSON
                    log_file.write(line)
                    log_file.flush()
                    # Format and display
                    format_stream_json(line, task_id)
        except KeyboardInterrupt:
            process.terminate()
            log(f"Interrupted by user")
            return 130

        process.wait()
        log(f"Session log saved to: {log_path}")
        return process.returncode

    else:
        # Quiet mode - run without streaming, save output to file
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            capture_output=True,
            text=True,
        )

        # Save output to log file
        with log_path.open("w") as log_file:
            log_file.write(result.stdout)
            if result.stderr:
                log_file.write(f"\n--- STDERR ---\n{result.stderr}")

        return result.returncode


def process_single_task(
    task_id: str,
    rubric_results: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    quiet: bool = False,
    resume_session: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """Process a single task. Returns (task_id, success, message)."""

    try:
        # Get rubric result
        rubric_result = rubric_results.get(task_id, {})

        # Load conversations
        conversations = load_task_conversations(trace_files, task_id)

        # Get task details
        task_details = get_task_details(trace_files, task_id)

        # Build prompt
        prompt = build_claude_prompt(
            task_id=task_id,
            rubric_result=rubric_result,
            conversations=conversations,
            task_details=task_details,
            benchmark=benchmark,
        )

        # Create fixes directory
        fix_dir = FIXES_DIR / benchmark / task_id
        fix_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt for reference
        prompt_path = fix_dir / "claude_prompt.txt"
        prompt_path.write_text(prompt)

        # Run Claude Code
        rc = run_claude_code(
            prompt=prompt,
            task_id=task_id,
            working_dir=REPO_ROOT,
            fix_dir=fix_dir,
            resume_session=resume_session,
            quiet=quiet,
        )

        if rc == 0:
            return (task_id, True, "completed")
        else:
            return (task_id, False, f"exit code {rc}")

    except Exception as e:
        return (task_id, False, str(e))


def main():
    global _total_count, _completed_count

    parser = argparse.ArgumentParser(
        description="Use Claude Code CLI to diagnose and fix environmental barriers"
    )
    parser.add_argument(
        "--trace-files", nargs="+", required=True,
        help="Trace files for context"
    )
    parser.add_argument(
        "--rubric-csv", required=True,
        help="Rubric CSV with evaluation results"
    )
    parser.add_argument(
        "--task-id", action="append", dest="task_ids",
        help="Specific task ID(s) to fix"
    )
    parser.add_argument(
        "--benchmark", default="corebench_hard",
        help="Benchmark name"
    )
    parser.add_argument(
        "--env-barriers-only", action="store_true",
        help="Only process tasks marked as environmental barriers (score >= 1)"
    )
    parser.add_argument(
        "--capability-issues-only", action="store_true",
        help="Only process tasks marked as capability issues (score < 1)"
    )
    parser.add_argument(
        "--resume", dest="resume_session",
        help="Resume a previous Claude Code session"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt without running Claude Code"
    )
    parser.add_argument(
        "--prefix", default="",
        help="Prefix for logging"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel Claude sessions (default: 1 for sequential with full logging)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tasks that already have fixes in fixes/<benchmark>/<task_id>/"
    )
    args = parser.parse_args()

    prefix = args.prefix

    # Load rubric results
    rubric_path = Path(args.rubric_csv)
    if not rubric_path.exists():
        log(f"Rubric CSV not found: {rubric_path}", prefix)
        return 1

    rubric_results = load_rubric_results(rubric_path)
    log(f"Loaded {len(rubric_results)} rubric results", prefix)

    # Determine which tasks to process
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = list(rubric_results.keys())

    # Filter by score if requested
    if args.env_barriers_only:
        task_ids = [t for t in task_ids if rubric_results.get(t, {}).get("score", 0) >= 1]
        log(f"Filtered to {len(task_ids)} environmental barriers", prefix)
    elif args.capability_issues_only:
        task_ids = [t for t in task_ids if rubric_results.get(t, {}).get("score", 0) < 1]
        log(f"Filtered to {len(task_ids)} capability issues", prefix)

    # Skip existing fixes if requested
    skipped_tasks = []
    if args.skip_existing:
        original_count = len(task_ids)
        task_ids_filtered = []
        for tid in task_ids:
            if has_existing_fix(args.benchmark, tid):
                skipped_tasks.append(tid)
            else:
                task_ids_filtered.append(tid)
        task_ids = task_ids_filtered
        log(f"Skipping {len(skipped_tasks)} tasks with existing fixes, {len(task_ids)} remaining", prefix)

    if not task_ids:
        log("No tasks to process", prefix)
        return 0

    # Load trace files
    trace_files = [Path(f) for f in args.trace_files]
    log(f"Using {len(trace_files)} trace files for context", prefix)

    # Set total for progress tracking
    _total_count = len(task_ids) + len(skipped_tasks)
    _completed_count = len(skipped_tasks)  # Skipped tasks count as done

    # Dry run mode
    if args.dry_run:
        for i, task_id in enumerate(task_ids):
            rubric_result = rubric_results.get(task_id, {})
            conversations = load_task_conversations(trace_files, task_id)
            task_details = get_task_details(trace_files, task_id)
            prompt = build_claude_prompt(
                task_id=task_id,
                rubric_result=rubric_result,
                conversations=conversations,
                task_details=task_details,
                benchmark=args.benchmark,
            )
            log(f"[{i+1}/{len(task_ids)}] {task_id} - Prompt length: {len(prompt)} chars", prefix)
            if args.parallel == 1:  # Only show full prompt in sequential mode
                print("\n" + "="*60)
                print(f"PROMPT FOR {task_id}")
                print("="*60)
                print(prompt[:5000] + "..." if len(prompt) > 5000 else prompt)
        return 0

    # Parallel mode
    if args.parallel > 1:
        log(f"\nRunning {len(task_ids)} tasks with {args.parallel} parallel workers", prefix)
        log("=" * 60, prefix)

        successes = 0
        failures = 0

        def run_task(task_id: str) -> Tuple[str, bool, str]:
            log_progress(task_id, "started", prefix)
            result = process_single_task(
                task_id=task_id,
                rubric_results=rubric_results,
                trace_files=trace_files,
                benchmark=args.benchmark,
                quiet=True,  # Minimal logging in parallel mode
                resume_session=args.resume_session,
            )
            if result[1]:
                log_progress(task_id, "completed", prefix)
            else:
                log_progress(task_id, "failed", prefix)
            return result

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_task, tid): tid for tid in task_ids}

            for future in as_completed(futures):
                task_id, success, message = future.result()
                if success:
                    successes += 1
                else:
                    failures += 1
                    log(f"  {task_id} failed: {message}", prefix)

        log("=" * 60, prefix)
        log(f"SUMMARY: {successes} succeeded, {failures} failed, {len(skipped_tasks)} skipped", prefix)

    # Sequential mode (with full logging)
    else:
        log(f"\nProcessing {len(task_ids)} tasks sequentially (use --parallel N for parallel execution)", prefix)

        for i, task_id in enumerate(task_ids):
            log(f"\n[{i+1}/{len(task_ids)}] Processing {task_id}", prefix)

            # Get rubric result
            rubric_result = rubric_results.get(task_id, {})

            # Load conversations
            conversations = load_task_conversations(trace_files, task_id)
            log(f"  Loaded conversations from {len(conversations)} models", prefix)

            # Get task details
            task_details = get_task_details(trace_files, task_id)

            # Build prompt
            prompt = build_claude_prompt(
                task_id=task_id,
                rubric_result=rubric_result,
                conversations=conversations,
                task_details=task_details,
                benchmark=args.benchmark,
            )

            # Create fixes directory
            fix_dir = FIXES_DIR / args.benchmark / task_id
            fix_dir.mkdir(parents=True, exist_ok=True)

            # Save prompt for reference
            prompt_path = fix_dir / "claude_prompt.txt"
            prompt_path.write_text(prompt)
            log(f"  Saved prompt to {prompt_path}", prefix)

            # Run Claude Code with full streaming output
            rc = run_claude_code(
                prompt=prompt,
                task_id=task_id,
                working_dir=REPO_ROOT,
                fix_dir=fix_dir,
                resume_session=args.resume_session,
                quiet=False,  # Full logging in sequential mode
            )

            if rc != 0:
                log(f"  Claude Code exited with code {rc}", prefix)
            else:
                log(f"  Claude Code completed successfully", prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
