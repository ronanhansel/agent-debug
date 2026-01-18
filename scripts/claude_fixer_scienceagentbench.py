#!/usr/bin/env python3
"""
Claude Code CLI-based ScienceAgentBench Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
in ScienceAgentBench tasks WITHOUT nerfing the scientific problems.

This script:
1. Gathers context about the task (rubric results, model conversations, task details)
2. Invokes Claude Code CLI with a detailed prompt
3. Claude Code analyzes and creates targeted fixes

Key principle: Fix INTRINSIC FORMATION ERRORS only, never simplify the science.

Usage:
    # List tasks with IFEs detected
    python scripts/claude_fixer_scienceagentbench.py --list-ife-tasks

    # Fix a specific task
    python scripts/claude_fixer_scienceagentbench.py --task-id 74

    # Fix all tasks with Grade=1 in judge verdict
    python scripts/claude_fixer_scienceagentbench.py --all-ife

    # Batch mode (multiple tasks per Claude session)
    python scripts/claude_fixer_scienceagentbench.py --task-id 74 --task-id 43 --task-id 2 --batch

    # Skip existing fixes
    python scripts/claude_fixer_scienceagentbench.py --all-ife --skip-existing
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
RUBRICS_OUTPUT_DIR = REPO_ROOT / "rubrics_output" / "scienceagentbench"
JUDGE_OUTPUT = REPO_ROOT / "judge_output" / "scienceagentbench_verdict.csv"
BENCHMARK = "scienceagentbench"

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


def has_existing_fix(task_id: str) -> bool:
    """Check if a task already has a fix."""
    fix_dir = FIXES_DIR / BENCHMARK / task_id
    if not fix_dir.exists():
        return False
    fix_files = ["env_override.json", "evaluation_override.json", "README.md"]
    return any((fix_dir / f).exists() for f in fix_files)


def load_judge_verdicts() -> Dict[str, Dict[str, Any]]:
    """Load judge verdicts by task_id."""
    results = {}
    if not JUDGE_OUTPUT.exists():
        return results
    with JUDGE_OUTPUT.open() as f:
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


def load_all_rubric_evaluations(task_id: str) -> List[Dict[str, Any]]:
    """Load all rubric evaluations for a task from all CSV files in the rubric directory."""
    evaluations = []
    for csv_file in RUBRICS_OUTPUT_DIR.glob("*.csv"):
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


def load_task_conversations(task_id: str) -> Dict[str, str]:
    """Load conversations for a task from trace files."""
    conversations = {}

    for trace_path in TRACES_DIR.glob(f"{BENCHMARK}_*.json"):
        if not trace_path.exists():
            continue

        try:
            data = json.loads(trace_path.read_text())
        except:
            continue

        # Extract model name
        model_name = data.get("config", {}).get("agent_args", {}).get("model_name", trace_path.stem[:30])
        model_name = model_name.replace("openai/", "").replace("-2025", "")

        # Get eval results for this task
        raw_eval = data.get("raw_eval_results", {}).get("eval_result", {})
        task_result = raw_eval.get(task_id, {})

        if task_result:
            lines = []
            lines.append(f"Success Rate: {task_result.get('success_rate', 'N/A')}")
            lines.append(f"Valid Program: {task_result.get('valid_program', 'N/A')}")
            lines.append(f"CodeBERT Score: {task_result.get('codebert_score', 'N/A')}")
            log_info = task_result.get('log_info', '')
            if log_info:
                lines.append(f"Log Info: {str(log_info)[:2000]}")

            conversations[model_name] = "\n".join(lines)

    return conversations


def get_ife_tasks() -> List[str]:
    """Get list of task IDs with IFE detected (Grade=1)."""
    verdicts = load_judge_verdicts()
    return [task_id for task_id, v in verdicts.items() if v.get("final_grade", 0) >= 1.0]


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

    # Format conversations (model results)
    conv_sections = []
    for model, conv in list(conversations.items())[:4]:
        conv_sections.append(f"#### {model}\n```\n{conv[:3000]}\n```")
    conversations_text = "\n".join(conv_sections) if conv_sections else "No logs."

    return f"""
---
## TASK: {task_id}
---

### Rubric Evaluations
{evaluations_text}

### Judge Verdict
{judge_text}

### Model Execution Results
{conversations_text}
"""


def build_claude_prompt_batch(tasks_data: List[Dict[str, Any]]) -> str:
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

    prompt = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in ScienceAgentBench tasks.

**YOU HAVE {len(task_ids)} TASKS TO PROCESS: {", ".join(task_ids)}**

Process each task sequentially, creating fixes as needed. Be THOROUGH in your analysis.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the scientific problem easier
2. **PRESERVE SCIENTIFIC RIGOR** - The task should remain as challenging as intended
3. **NO NERFING** - Do not simplify scientific concepts, give hints, reduce precision, or pre-compute results
4. **VALID FIXES**: Environment packages, Docker config, evaluation tolerance, ambiguous instructions
5. **INVALID FIXES**: Solution hints, simplified science, pre-importing specialized modules

## SCIENCEAGENTBENCH HARNESS STRUCTURE

**First, read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/scienceagentbench.py` - Main benchmark class
- `hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench/` - Evaluation harness

**How Evaluation Works:**
1. Agent produces Python code for a scientific task
2. Code is executed in Docker container with `evaluation/harness.py`
3. Metrics collected: Valid Execution Rate, Success Rate (task-specific criteria), CodeBERTScore
4. For visualization tasks: GPT-4 compares generated figures to gold standard

**To inspect a specific task, run:**
```python
from datasets import load_dataset
ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
task = ds[int(TASK_ID) - 1]  # 1-indexed
print(task['task_inst'])  # Task instruction
print(task['dataset_folder_tree'])  # Data files available
```

## THOROUGH ERROR ANALYSIS CHECKLIST

For EACH task, systematically check ALL of these potential error sources:

### 1. Environment/Dependency Issues
- [ ] Missing Python packages in Docker container (oggm, mne, mastml, biopsykit)
- [ ] Package version conflicts between scientific libraries
- [ ] Missing system libraries (GDAL, MPI, etc.)
- [ ] Conda vs pip installation conflicts for scientific packages
- [ ] GPU/CUDA requirements for deep learning tasks
- [ ] Memory/timeout limits too restrictive for large datasets

### 2. Data/Input Issues
- [ ] Missing or corrupted data files in dataset
- [ ] Data format differs from task documentation
- [ ] Column names don't match task description
- [ ] Encoding issues with scientific data files
- [ ] File path mismatches between task description and actual paths

### 3. Task Specification Issues
- [ ] Ambiguous output format requirements
- [ ] Unclear success criteria (what makes output "correct"?)
- [ ] Missing domain knowledge in instructions (formulas, methods)
- [ ] Conflicting requirements between task steps
- [ ] Unstated assumptions from source scientific papers

### 4. Evaluation Script Issues
- [ ] Numerical tolerance too strict for scientific precision
- [ ] Format-sensitive comparison (whitespace, column ordering)
- [ ] Evaluation crashes on valid but alternative outputs
- [ ] Metrics don't match task description expectations
- [ ] GPT-4 judge subjectivity for figure/visualization tasks

### 5. Gold Program/Reference Issues
- [ ] Gold program has hardcoded paths
- [ ] Gold program uses unavailable or outdated libraries
- [ ] Multiple scientifically valid approaches rejected
- [ ] Gold program doesn't match task requirements exactly

### 6. Cross-Model Failure Patterns
- [ ] Same error across ALL models → likely IFE
- [ ] Valid output rejected by evaluation → evaluation issue
- [ ] Environment blocks ALL models identically → setup issue
- [ ] Figure evaluation fails for functionally equivalent plots

## KNOWN IFE PATTERNS (from trace analysis)

**Environment Issues (High Priority):**
- Task 74: `oggm` / `oggm.core.distribute_2d` (glacier modeling) not available
- Task 43: `mne` (neuroimaging) not available
- Task 2: `mastml.features` (materials science ML) not available
- Other domain-specific packages: `biopsykit`, specialized GIS libraries

**Figure Evaluation Issues:**
- GPT-4 judge penalizes functionally equivalent but stylistically different visualizations
- Color scheme differences cause failures despite correct scientific content
- Axis label formatting differences cause failures
- Tasks 8, 25, 28, 50, 59, 68, 84, 91, 93 have figure evaluation issues

**Key Statistics:**
- 67/102 tasks failed across ALL 4 models (GPT-4.1, O3, O4-mini-high, O4-mini-low)
- 39 tasks have runtime errors in 2+ models
- 13 tasks have figure comparison failures in 2+ models

## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/scienceagentbench/TASK_ID/`

**Environment Fixes** (`env_override.json`):
```json
{{
  "HAL_CONDA_CHANNELS": "conda-forge",
  "HAL_CONDA_PACKAGES": "mne oggm",
  "HAL_PIP_PACKAGES": "biopsykit mastml",
  "HAL_APT_PACKAGES": "libfoo-dev",
  "HAL_TIMEOUT_SECONDS": 600,
  "notes": "Justification for these environment changes"
}}
```

**Evaluation Fixes** (`evaluation_override.json`):
```json
{{
  "figure_tolerance": "relaxed",
  "numerical_tolerance": 1e-4,
  "accept_alternative_formats": true,
  "skip_style_check": true,
  "notes": "Why this adjustment is fair, not a nerf"
}}
```

**Instruction Clarifications** (`instruction_override.json`):
```json
{{
  "clarifications": [
    "Output file must be named exactly 'output.csv'",
    "Use specific library version X"
  ],
  "additional_context": "Any missing domain knowledge needed"
}}
```

**Documentation** (`README.md`):
- Root cause analysis of the IFE
- What fix was applied and why
- Why this preserves task difficulty
- Expected outcome after fix

If NO fix needed (capability issue, not IFE), create README.md explaining why.

## FIX RUNNER SCRIPT

After creating fixes, also check/update the fix runner script:
`scripts/run_scienceagentbench_fixes.py`

The fix runner must:
1. Load fixes from `fixes/scienceagentbench/<task_id>/`
2. Apply environment overrides before Docker evaluation
3. Inject instruction clarifications into task prompts
4. Adjust evaluation parameters as specified
5. Run HAL evaluation with fixes applied
6. Output new traces with configurable prefix

**Reference implementation**: See `scripts/run_scicode_fixes.py` for the pattern.

## TASKS TO PROCESS

{tasks_text}

## BEGIN - SYSTEMATIC APPROACH

For EACH task:

1. **Read the benchmark code** to understand evaluation pipeline
2. **Load the specific task** from HuggingFace dataset
3. **Analyze ALL error messages** from model execution logs
4. **Check EACH item** in the error analysis checklist above
5. **Cross-reference with other models** - same error = likely IFE
6. **Create fix OR document why no fix needed**
7. **Verify fix doesn't nerf the scientific problem**

After processing all tasks:
8. **Update fix runner script** if new fix types were used
9. **Test that fixes can be applied** without errors

Remember: Make evaluation FAIR, not EASY. Be THOROUGH in diagnosis.
Scientific rigor must be preserved - we fix infrastructure, not science.
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
        # Quiet mode
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            capture_output=True,
            text=True,
        )
        with log_path.open("w") as log_file:
            log_file.write(result.stdout)
        return result.returncode


def process_task(task_id: str, skip_existing: bool = False, quiet: bool = False) -> bool:
    """Process a single task."""
    if skip_existing and has_existing_fix(task_id):
        log_progress(task_id, "skipped")
        return True

    log_progress(task_id, "started")

    # Gather context
    evaluations = load_all_rubric_evaluations(task_id)
    verdicts = load_judge_verdicts()
    judge_verdict = verdicts.get(task_id)
    conversations = load_task_conversations(task_id)

    # Build prompt
    tasks_data = [{
        'task_id': task_id,
        'evaluations': evaluations,
        'judge_verdict': judge_verdict,
        'conversations': conversations,
    }]
    prompt = build_claude_prompt_batch(tasks_data)

    # Create fix directory
    fix_dir = FIXES_DIR / BENCHMARK / task_id
    fix_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt for reference
    (fix_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    # Run Claude Code
    result = run_claude_code(prompt, task_id, REPO_ROOT, fix_dir, quiet=quiet)

    if result == 0:
        log_progress(task_id, "completed")
        return True
    else:
        log_progress(task_id, "failed")
        return False


def process_batch(task_ids: List[str], skip_existing: bool = False) -> bool:
    """Process multiple tasks in a single Claude session."""

    # Filter tasks
    if skip_existing:
        task_ids = [t for t in task_ids if not has_existing_fix(t)]

    if not task_ids:
        log("No tasks to process (all have existing fixes)")
        return True

    log(f"Processing batch of {len(task_ids)} tasks: {', '.join(task_ids)}")

    # Gather context for all tasks
    tasks_data = []
    verdicts = load_judge_verdicts()

    for task_id in task_ids:
        evaluations = load_all_rubric_evaluations(task_id)
        judge_verdict = verdicts.get(task_id)
        conversations = load_task_conversations(task_id)

        tasks_data.append({
            'task_id': task_id,
            'evaluations': evaluations,
            'judge_verdict': judge_verdict,
            'conversations': conversations,
        })

    # Build combined prompt
    prompt = build_claude_prompt_batch(tasks_data)

    # Create fix directory for batch
    batch_id = "_".join(task_ids[:3]) + (f"_and_{len(task_ids)-3}_more" if len(task_ids) > 3 else "")
    fix_dir = FIXES_DIR / BENCHMARK / f"_batch_{batch_id}"
    fix_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt for reference
    (fix_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    # Run Claude Code
    result = run_claude_code(prompt, batch_id, REPO_ROOT, fix_dir)

    return result == 0


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code CLI-based ScienceAgentBench Fixer"
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Task ID(s) to process (can be repeated)",
    )
    parser.add_argument(
        "--all-ife",
        action="store_true",
        help="Process all tasks with IFE detected (Grade=1)",
    )
    parser.add_argument(
        "--list-ife-tasks",
        action="store_true",
        help="List tasks with IFE detected and exit",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have fixes",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all tasks in a single Claude session",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (no streaming)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel Claude sessions (default: 1)",
    )

    args = parser.parse_args()

    # List IFE tasks
    if args.list_ife_tasks:
        ife_tasks = get_ife_tasks()
        print(f"\nTasks with IFE detected ({len(ife_tasks)} total):")
        for task_id in sorted(ife_tasks, key=lambda x: int(x) if x.isdigit() else x):
            verdicts = load_judge_verdicts()
            v = verdicts.get(task_id, {})
            has_fix = "✓ has fix" if has_existing_fix(task_id) else ""
            print(f"  {task_id}: Grade={v.get('final_grade', 'N/A')} {has_fix}")
        return

    # Get task list
    if args.all_ife:
        task_ids = get_ife_tasks()
        log(f"Found {len(task_ids)} tasks with IFE detected")
    elif args.task_ids:
        task_ids = args.task_ids
    else:
        parser.error("Must specify --task-id, --all-ife, or --list-ife-tasks")
        return

    if not task_ids:
        log("No tasks to process")
        return

    global _total_count
    _total_count = len(task_ids)

    # Process tasks
    if args.batch:
        success = process_batch(task_ids, skip_existing=args.skip_existing)
        sys.exit(0 if success else 1)
    elif args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_task, task_id, args.skip_existing, args.quiet): task_id
                for task_id in task_ids
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    log(f"Error processing {task_id}: {e}")
    else:
        for task_id in task_ids:
            process_task(task_id, skip_existing=args.skip_existing, quiet=args.quiet)


if __name__ == "__main__":
    main()
