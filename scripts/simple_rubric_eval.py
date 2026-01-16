#!/usr/bin/env python3
"""
Simple Rubric Evaluator - Uses OpenAI directly without docent dependencies.
This is a lightweight alternative to the docent-based rubric evaluator.
"""

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RubricResult:
    task_id: str
    score: float
    explanation: str
    error: Optional[str] = None


def load_rubric(rubrics_dir: Path) -> tuple[str, dict]:
    """Load rubric text and schema from directory."""
    rubric_files = list(rubrics_dir.glob("*.txt"))
    if not rubric_files:
        raise FileNotFoundError(f"No rubric .txt files found in {rubrics_dir}")

    rubric_path = rubric_files[0]  # Use first rubric found
    rubric_text = rubric_path.read_text()

    # Load schema if exists
    schema_path = rubric_path.with_suffix(".schema.json")
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
    else:
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "enum": [0, 1]},
                "explanation": {"type": "string"}
            },
            "required": ["score", "explanation"]
        }

    return rubric_text, schema


def load_trace(trace_path: Path) -> Dict[str, Any]:
    """Load trace file."""
    return json.loads(trace_path.read_text())


def extract_task_conversations(trace_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Extract conversations grouped by task_id."""
    raw_entries = trace_data.get("raw_logging_results", [])

    tasks: Dict[str, List[Dict]] = {}
    for entry in raw_entries:
        task_id = (
            entry.get("attributes", {}).get("weave_task_id")
            or entry.get("weave_task_id")
            or entry.get("inputs", {}).get("task_id")
            or "unknown"
        )
        tasks.setdefault(task_id, []).append(entry)

    return tasks


def format_conversation_for_rubric(entries: List[Dict]) -> str:
    """Format conversation entries into readable text for rubric evaluation."""
    lines = []

    for i, entry in enumerate(entries[:50]):  # Limit to 50 entries to avoid token limits
        # Extract messages from input
        messages = entry.get("inputs", {}).get("messages", [])
        for msg in messages[-5:]:  # Last 5 messages per entry
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if content:
                content = content[:1000]  # Truncate long content
                lines.append(f"[{role}]: {content}")

        # Extract output/response
        output = entry.get("output", {})
        if output:
            choices = output.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if content:
                    content = content[:1000]
                    lines.append(f"[assistant]: {content}")

    return "\n\n".join(lines[-100:])  # Keep last 100 formatted blocks


def evaluate_with_openai(
    client: OpenAI,
    model: str,
    rubric_text: str,
    conversation: str,
    task_id: str,
) -> RubricResult:
    """Evaluate a single conversation against the rubric using OpenAI."""

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

        # Try to parse JSON
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = re.sub(r"```(?:json)?\s*", "", result_text)
            result_text = result_text.rstrip("`")

        result = json.loads(result_text)

        return RubricResult(
            task_id=task_id,
            score=float(result.get("score", 0)),
            explanation=result.get("explanation", ""),
        )
    except json.JSONDecodeError as e:
        return RubricResult(
            task_id=task_id,
            score=0,
            explanation="",
            error=f"JSON parse error: {e}. Raw response: {result_text[:200]}"
        )
    except Exception as e:
        return RubricResult(
            task_id=task_id,
            score=0,
            explanation="",
            error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description="Simple rubric evaluator using OpenAI")
    parser.add_argument("--trace-file", required=True, help="Path to trace JSON file")
    parser.add_argument("--rubrics-dir", default="rubrics", help="Directory with rubric files")
    parser.add_argument("--output-dir", default="rubrics_output", help="Output directory for CSV")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model to use")
    parser.add_argument("--max-tasks", type=int, help="Limit number of tasks to evaluate")
    parser.add_argument("--failed-only", action="store_true", help="Only evaluate failed tasks")
    parser.add_argument("--base-url", default=None, help="OpenAI API base URL (for proxy)")
    parser.add_argument("--api-key", default="dummy", help="OpenAI API key")
    args = parser.parse_args()

    # Set up OpenAI client
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "dummy")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load rubric
    rubrics_dir = Path(args.rubrics_dir)
    rubric_text, schema = load_rubric(rubrics_dir)
    print(f"Loaded rubric from {rubrics_dir}")

    # Load trace
    trace_path = Path(args.trace_file)
    trace_data = load_trace(trace_path)
    print(f"Loaded trace from {trace_path}")

    # Get task info
    results_block = trace_data.get("results", {})
    failed_tasks = set(results_block.get("failed_tasks", []))
    successful_tasks = set(results_block.get("successful_tasks", []))

    print(f"Total failed tasks: {len(failed_tasks)}")
    print(f"Total successful tasks: {len(successful_tasks)}")

    # Extract conversations
    task_conversations = extract_task_conversations(trace_data)
    print(f"Extracted {len(task_conversations)} task conversations")

    # Filter to failed tasks if requested
    if args.failed_only:
        task_conversations = {
            k: v for k, v in task_conversations.items()
            if k in failed_tasks
        }
        print(f"Filtered to {len(task_conversations)} failed tasks")

    # Limit tasks if requested
    task_ids = list(task_conversations.keys())
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
        print(f"Limited to {len(task_ids)} tasks")

    # Evaluate each task
    results: List[RubricResult] = []
    for i, task_id in enumerate(task_ids):
        print(f"\nEvaluating task {i+1}/{len(task_ids)}: {task_id}")

        entries = task_conversations[task_id]
        conversation = format_conversation_for_rubric(entries)

        if not conversation.strip():
            print(f"  Skipping - no conversation content")
            continue

        result = evaluate_with_openai(
            client=client,
            model=args.model,
            rubric_text=rubric_text,
            conversation=conversation,
            task_id=task_id,
        )

        results.append(result)

        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Score: {result.score}")
            print(f"  Explanation: {result.explanation}")

    # Write results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_label = trace_path.stem
    rubric_name = "environmental_barrier"

    output_path = output_dir / rubric_name / f"{trace_label}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "criteria", "grade", "correct", "explanation", "model_run"])

        for result in results:
            correct = ""
            if result.task_id in successful_tasks:
                correct = "1"
            elif result.task_id in failed_tasks:
                correct = "0"

            writer.writerow([
                result.task_id,
                rubric_name,
                f"{result.score:.2f}",
                correct,
                result.explanation,
                trace_label,
            ])

    print(f"\nResults written to {output_path}")

    # Summary
    env_barrier_count = sum(1 for r in results if r.score >= 1)
    print(f"\nSummary:")
    print(f"  Total evaluated: {len(results)}")
    print(f"  Environmental barriers: {env_barrier_count}")
    print(f"  Capability issues: {len(results) - env_barrier_count}")


if __name__ == "__main__":
    main()
