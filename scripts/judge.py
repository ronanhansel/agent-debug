#!/usr/bin/env python3
"""
Judge script for aggregating rubric evaluations across multiple model runs.

Reads rubric CSV outputs, groups by task_id, and uses an LLM to make a final
verdict on whether each task satisfies the rubric based on all evaluations.

Usage:
    python scripts/judge.py \
        --pattern "scicode_*" \
        --rubric-dir rubrics_output/scicode \
        --model openai:gpt-5.2 \
        --output judge_output/scicode_verdict.csv \
        -y

    # With reasoning model
    python scripts/judge.py \
        --pattern "*.csv" \
        --rubric-dir rubrics_output/scicode \
        --model openai:o3-mini \
        --reasoning-effort medium \
        -y
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


@dataclass
class TaskEvaluation:
    """A single evaluation result for a task from one model run."""
    task_id: str
    criteria: str
    grade: str
    correct: str
    explanation: str
    model_run: str
    source_file: str


@dataclass
class JudgeVerdict:
    """Final verdict from the judge for a task."""
    task_id: str
    final_grade: float
    satisfies_rubric: bool
    reasoning: str
    num_evaluations: int
    model_runs: list[str]


JUDGE_SYSTEM_PROMPT = """You are an expert judge analyzing rubric evaluations from multiple AI model runs.

Your task is to determine whether a specific task satisfies the rubric criteria based on evaluations from different models.

The rubric evaluates "formational errors" - issues with the evaluation setup, environment, or infrastructure that prevent fair assessment of an agent's capabilities. These are NOT about the agent's coding ability, but about external barriers.

For each task, you will see evaluations from multiple model runs. Each evaluation includes:
- The grade (0-1 scale, where higher means more formational errors detected)
- Whether it was marked correct
- The explanation of what issues were found

Your job is to:
1. Analyze all evaluations for consistency and validity
2. Consider the explanations and reasoning from each evaluation
3. Determine a final verdict on whether formational errors exist for this task
4. Provide your reasoning

Respond in JSON format:
{
    "final_grade": <float 0-1>,
    "satisfies_rubric": <boolean - true if formational errors exist>,
    "reasoning": "<your detailed reasoning>"
}
"""


def find_csv_files(rubric_dir: Path, pattern: str) -> list[Path]:
    """Find CSV files matching the pattern in the rubric directory."""
    csv_files = []
    for f in rubric_dir.glob("*.csv"):
        if fnmatch(f.name, pattern):
            csv_files.append(f)
    return sorted(csv_files)


def read_evaluations(csv_files: list[Path]) -> dict[str, list[TaskEvaluation]]:
    """Read all CSV files and group evaluations by task_id."""
    evaluations: dict[str, list[TaskEvaluation]] = defaultdict(list)

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_item = TaskEvaluation(
                    task_id=row.get("task_id", ""),
                    criteria=row.get("criteria", ""),
                    grade=row.get("grade", ""),
                    correct=row.get("correct", ""),
                    explanation=row.get("explanation", ""),
                    model_run=row.get("model_run", ""),
                    source_file=csv_file.name,
                )
                evaluations[eval_item.task_id].append(eval_item)

    return evaluations


def build_judge_prompt(task_id: str, evals: list[TaskEvaluation]) -> str:
    """Build the prompt for the judge LLM."""
    prompt_parts = [
        f"# Task ID: {task_id}\n",
        f"## Evaluations from {len(evals)} model run(s):\n",
    ]

    for i, ev in enumerate(evals, 1):
        prompt_parts.append(f"""
### Evaluation {i} (from {ev.model_run})
- **Source**: {ev.source_file}
- **Criteria**: {ev.criteria}
- **Grade**: {ev.grade}
- **Correct**: {ev.correct}
- **Explanation**: {ev.explanation}
""")

    prompt_parts.append("""
## Your Task
Based on the above evaluations, provide your final verdict on whether this task has formational errors (issues with the evaluation environment, not agent capability).

Respond in JSON format with: final_grade, satisfies_rubric, reasoning
""")

    return "\n".join(prompt_parts)


def parse_judge_response(response: str) -> dict:
    """Parse the JSON response from the judge LLM."""
    # Try to extract JSON from the response
    try:
        # First try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in response
    json_match = re.search(r'\{[^{}]*"final_grade"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON in code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Return default if parsing fails
    return {
        "final_grade": 0.0,
        "satisfies_rubric": False,
        "reasoning": f"Failed to parse response: {response[:500]}",
    }


def judge_task(
    client: OpenAI,
    task_id: str,
    evals: list[TaskEvaluation],
    model: str,
    reasoning_effort: str | None = None,
    retries: int = 3,
) -> JudgeVerdict:
    """Use LLM to judge a single task based on all its evaluations."""
    prompt = build_judge_prompt(task_id, evals)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Build request kwargs
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    # Add reasoning effort for reasoning models
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            parsed = parse_judge_response(content)

            return JudgeVerdict(
                task_id=task_id,
                final_grade=float(parsed.get("final_grade", 0.0)),
                satisfies_rubric=bool(parsed.get("satisfies_rubric", False)),
                reasoning=str(parsed.get("reasoning", "")),
                num_evaluations=len(evals),
                model_runs=[ev.model_run for ev in evals],
            )
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"    Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # Return error verdict if all retries failed
    return JudgeVerdict(
        task_id=task_id,
        final_grade=0.0,
        satisfies_rubric=False,
        reasoning=f"Failed after {retries} attempts: {last_error}",
        num_evaluations=len(evals),
        model_runs=[ev.model_run for ev in evals],
    )


def write_verdicts_csv(verdicts: list[JudgeVerdict], output_path: Path) -> None:
    """Write judge verdicts to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "final_grade",
            "satisfies_rubric",
            "num_evaluations",
            "model_runs",
            "reasoning",
        ])
        for v in verdicts:
            writer.writerow([
                v.task_id,
                f"{v.final_grade:.3f}",
                "1" if v.satisfies_rubric else "0",
                v.num_evaluations,
                ";".join(v.model_runs),
                v.reasoning,
            ])

    print(f"Wrote {len(verdicts)} verdicts to {output_path}")


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Parse provider:model string. Returns (provider, model_name)."""
    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
        return provider, model_name
    return "openai", model_str


def main():
    parser = argparse.ArgumentParser(
        description="Judge aggregates rubric evaluations and produces final verdicts."
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match CSV files (e.g., 'scicode_*', '*.csv')",
    )
    parser.add_argument(
        "--rubric-dir",
        type=str,
        required=True,
        help="Directory containing rubric CSV outputs (e.g., rubrics_output/scicode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model as provider:model (e.g., openai:gpt-5.2, openai:o3-mini)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV path (default: judge_output/<rubric_dir_name>_verdict.csv)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit number of tasks to judge",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per task on failure (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay in seconds between API calls (default: 0)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Resolve paths
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    if not rubric_dir.exists():
        print(f"Error: Rubric directory not found: {rubric_dir}")
        sys.exit(1)

    # Find CSV files
    csv_files = find_csv_files(rubric_dir, args.pattern)
    if not csv_files:
        print(f"Error: No CSV files matching '{args.pattern}' found in {rubric_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Read evaluations
    evaluations = read_evaluations(csv_files)
    print(f"\nLoaded evaluations for {len(evaluations)} unique tasks")

    # Limit tasks if requested
    task_ids = list(evaluations.keys())
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
        print(f"Limiting to {len(task_ids)} tasks")

    # Parse model
    provider, model_name = parse_model_string(args.model)
    print(f"\nUsing model: {provider}:{model_name}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")

    # Create OpenAI client (works with LiteLLM proxy via OPENAI_BASE_URL)
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "sk-1234")
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"API base URL: {base_url}")

    # Confirm
    if not args.yes:
        response = input(f"\nJudge {len(task_ids)} tasks? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        output_path = REPO_ROOT / "judge_output" / f"{rubric_dir.name}_verdict.csv"

    # Run judge
    verdicts = []
    for i, task_id in enumerate(task_ids, 1):
        evals = evaluations[task_id]
        print(f"\n[{i}/{len(task_ids)}] Judging task {task_id} ({len(evals)} evaluations)...")

        verdict = judge_task(
            client=client,
            task_id=task_id,
            evals=evals,
            model=model_name,
            reasoning_effort=args.reasoning_effort,
            retries=args.retries,
        )
        verdicts.append(verdict)

        status = "SATISFIES" if verdict.satisfies_rubric else "DOES NOT SATISFY"
        print(f"  -> {status} rubric (grade: {verdict.final_grade:.2f})")
        if verdict.reasoning:
            reasoning_preview = verdict.reasoning[:150].replace("\n", " ")
            print(f"     Reasoning: {reasoning_preview}...")

        # Delay between tasks
        if args.delay > 0 and i < len(task_ids):
            time.sleep(args.delay)

    # Write output
    write_verdicts_csv(verdicts, output_path)

    # Summary
    satisfies_count = sum(1 for v in verdicts if v.satisfies_rubric)
    print(f"\n{'='*60}")
    print(f"Summary: {satisfies_count}/{len(verdicts)} tasks satisfy the rubric")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
