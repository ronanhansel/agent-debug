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


def build_claude_prompt(
    task_id: str,
    evaluations: List[Dict[str, Any]],
    judge_verdict: Optional[Dict[str, Any]],
    conversations: Dict[str, str],
    benchmark: str,
) -> str:
    """Build the prompt for Claude Code CLI."""

    # Format evaluations section
    eval_sections = []
    for i, ev in enumerate(evaluations, 1):
        eval_sections.append(f"""
### Evaluation {i} (from {ev.get('model_run', 'unknown')})
- **Grade**: {ev.get('grade', 'N/A')} (1.0 = IFE detected, 0.0 = no IFE)
- **Explanation**: {ev.get('explanation', 'N/A')[:3000]}
""")
    evaluations_text = "\n".join(eval_sections) if eval_sections else "No evaluations available."

    # Format judge verdict
    judge_text = "No judge verdict available."
    if judge_verdict:
        judge_text = f"""
**Final Grade**: {judge_verdict.get('final_grade', 'N/A')}
**Satisfies Rubric (has IFE)**: {judge_verdict.get('satisfies_rubric', False)}
**Judge Reasoning**: {judge_verdict.get('reasoning', 'N/A')}
**Number of Evaluations Considered**: {judge_verdict.get('num_evaluations', 0)}
"""

    # Format conversations section
    conv_sections = []
    for model, conv in conversations.items():
        conv_sections.append(f"### Model: {model}\n```\n{conv[:10000]}\n```")
    conversations_text = "\n\n".join(conv_sections) if conv_sections else "No conversation logs available."

    prompt = f'''You are diagnosing and fixing an Intrinsic Formation Error (IFE) in a SciCode benchmark task.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the scientific problem easier
2. **PRESERVE SCIENTIFIC RIGOR** - The task should remain as challenging as intended
3. **NO NERFING** - Do not:
   - Simplify the scientific concepts or equations
   - Give hints about the solution approach
   - Reduce numerical precision requirements
   - Pre-compute results for the agent
   - Remove required scientific understanding
4. **VALID FIXES INCLUDE**:
   - Fixing contradictory dependency constraints (e.g., whitelist too restrictive)
   - Clarifying ambiguous function signatures or expected outputs
   - Fixing parsing/regex issues in the evaluation harness
   - Correcting inconsistent instructions across problem steps
   - Ensuring test cases match the stated requirements
5. **INVALID FIXES** (DO NOT DO):
   - Adding solution hints
   - Simplifying the physics/math/chemistry
   - Pre-importing modules the agent should determine
   - Reducing the scope of scientific calculation

## TASK INFORMATION

**Task ID**: {task_id}
**Benchmark**: {benchmark}

## RUBRIC EVALUATIONS FROM MULTIPLE MODELS

The following evaluations analyzed traces from different AI models attempting this task.
Each evaluation grades whether an Intrinsic Formation Error exists (1.0 = definite IFE, 0.0 = no IFE).

{evaluations_text}

## JUDGE VERDICT (Aggregated Analysis)

{judge_text}

## MODEL CONVERSATION LOGS

The following shows how different models attempted this task:

{conversations_text}

## YOUR MISSION

1. **DIAGNOSE**: Analyze the evaluations and conversation logs to understand:
   - What specific IFE exists (contradictory constraints, parsing issues, etc.)?
   - Is there consensus across evaluations?
   - Is this truly an IFE or a capability issue?

2. **INVESTIGATE**:
   - Look at the SciCode task definition if available
   - Check the dependency constraints vs. required functionality
   - Understand what the correct setup should be

3. **DETERMINE FIX TYPE**:

   **Type A - Dependency Fix** (dependency_override.json):
   - Expand allowed dependencies to include required modules (e.g., scipy.constants)
   - Fix conda/pip package specifications

   **Type B - Instruction Fix** (instruction_override.json):
   - Clarify ambiguous function signatures
   - Fix inconsistent step requirements
   - Correct misleading wording

   **Type C - Evaluation Fix** (evaluation_override.json):
   - Fix parsing regex patterns
   - Adjust output format expectations
   - Correct test case assertions

   **Type D - No Fix Needed**:
   - The failure is actually a capability issue
   - The evaluations incorrectly identified an IFE

4. **CREATE FIX** (if needed):
   Create the fix in: fixes/{benchmark}/{task_id}/

   Write the appropriate override file and a README.md explaining:
   - What IFE was identified
   - What fix was applied
   - Why this fix is appropriate and doesn't nerf the task

5. **DOCUMENT**:
   - Explain your diagnosis
   - Justify why this fix is appropriate
   - Confirm the fix does NOT simplify the science

## DIRECTORY STRUCTURE

```
fixes/
└── {benchmark}/
    └── {task_id}/
        ├── dependency_override.json   # Additional allowed dependencies
        ├── instruction_override.json  # Clarified instructions
        ├── evaluation_override.json   # Evaluation harness fixes
        └── README.md                  # Explanation of the fix
```

## OVERRIDE FILE FORMATS

### dependency_override.json
```json
{{
  "additional_imports": ["scipy.constants", "scipy.special"],
  "pip_packages": ["sympy>=1.12"],
  "conda_packages": ["r-base"],
  "rationale": "scipy.constants needed for physical constants but was not in allowed list"
}}
```

### instruction_override.json
```json
{{
  "clarifications": [
    "The function should return values in SI units",
    "Use numpy arrays for vector operations"
  ],
  "corrected_signature": "def calculate_energy(n: int, l: int) -> float:",
  "rationale": "Original signature was inconsistent with the problem description"
}}
```

### evaluation_override.json
```json
{{
  "code_fence_pattern": "```(?:python|py)?\\\\s*\\\\n(.*?)\\\\n```",
  "output_tolerance": 1e-6,
  "rationale": "Original regex too strict, rejected valid code blocks"
}}
```

## BEGIN ANALYSIS

Start by analyzing the evaluations and conversation logs.
Then diagnose the specific IFE and create an appropriate fix (or determine no fix is needed).

Remember: Your goal is to make the evaluation FAIR, not EASY. The science must remain challenging.
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


def process_single_task(
    task_id: str,
    rubric_dir: Path,
    judge_verdicts: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    quiet: bool = False,
) -> Tuple[str, bool, str]:
    """Process a single task. Returns (task_id, success, message)."""

    try:
        # Load all evaluations for this task
        evaluations = load_all_rubric_evaluations(rubric_dir, task_id)

        # Get judge verdict
        judge_verdict = judge_verdicts.get(task_id)

        # Load conversations
        conversations = load_task_conversations(trace_files, task_id)

        # Create fix directory
        fix_dir = FIXES_DIR / benchmark / task_id
        fix_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt
        prompt = build_claude_prompt(
            task_id=task_id,
            evaluations=evaluations,
            judge_verdict=judge_verdict,
            conversations=conversations,
            benchmark=benchmark,
        )

        # Save prompt for reference
        (fix_dir / "claude_prompt.txt").write_text(prompt)

        if not quiet:
            log_progress(task_id, "started")

        # Run Claude Code
        exit_code = run_claude_code(
            prompt=prompt,
            task_id=task_id,
            working_dir=REPO_ROOT,
            fix_dir=fix_dir,
            quiet=quiet,
        )

        if exit_code == 0:
            if not quiet:
                log_progress(task_id, "completed")
            return (task_id, True, "Fix created successfully")
        else:
            if not quiet:
                log_progress(task_id, "failed")
            return (task_id, False, f"Claude Code exited with code {exit_code}")

    except Exception as e:
        if not quiet:
            log_progress(task_id, "failed")
        return (task_id, False, str(e))


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
        required=True,
        help="Trace files to extract conversations from",
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
        "--dry-run",
        action="store_true",
        help="Preview prompts without running Claude Code",
    )

    args = parser.parse_args()

    # Resolve paths
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    trace_files = [Path(f) if Path(f).is_absolute() else REPO_ROOT / f for f in args.trace_files]

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

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Previewing prompts")
        print("="*60)

        for task_id in task_ids[:3]:  # Preview first 3
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations(trace_files, task_id)

            prompt = build_claude_prompt(
                task_id=task_id,
                evaluations=evaluations,
                judge_verdict=judge_verdict,
                conversations=conversations,
                benchmark=args.benchmark,
            )

            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"Evaluations: {len(evaluations)}")
            print(f"Judge verdict: {'Yes' if judge_verdict else 'No'}")
            print(f"Conversations: {len(conversations)} models")
            print(f"Prompt length: {len(prompt)} chars")
            print("="*60)
            print(prompt[:3000] + "..." if len(prompt) > 3000 else prompt)

        print(f"\n{'='*60}")
        print(f"DRY RUN COMPLETE - Would process {len(task_ids)} tasks")
        return

    # Process tasks
    global _total_count
    _total_count = len(task_ids)

    if args.parallel > 1:
        log(f"Running {args.parallel} parallel sessions")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_single_task,
                    task_id,
                    rubric_dir,
                    judge_verdicts,
                    trace_files,
                    args.benchmark,
                    quiet=True,
                ): task_id
                for task_id in task_ids
            }

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # Summary
        successful = sum(1 for _, success, _ in results if success)
        log(f"\nCompleted: {successful}/{len(results)} tasks successful")
    else:
        for task_id in task_ids:
            process_single_task(
                task_id=task_id,
                rubric_dir=rubric_dir,
                judge_verdicts=judge_verdicts,
                trace_files=trace_files,
                benchmark=args.benchmark,
                quiet=False,
            )


if __name__ == "__main__":
    main()
