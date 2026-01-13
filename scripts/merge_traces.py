#!/usr/bin/env python3
"""
Merge multiple HAL trace JSON files into a single trace that can be re-graded.

Example:
    python scripts/merge_traces.py \
        --input 'traces/capsule-*20260113*_UPLOAD.json' \
        --output traces/corebench_hard_hal_generalist_agentgpt41_20260113_FIXED.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge HAL trace JSON files.")
    parser.add_argument(
        "--input",
        dest="patterns",
        action="append",
        required=True,
        help="Glob pattern for trace files (can be repeated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file for the merged trace.",
    )
    parser.add_argument(
        "--run-id",
        help="Override the run_id stored in the merged trace (default: output filename stem).",
    )
    parser.add_argument(
        "--agent-name",
        help="Override the agent_name stored in the merged trace.",
    )
    parser.add_argument(
        "--date",
        help="Override the date stored in the merged trace (default: first trace date).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def expand_patterns(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matched = [Path(p).resolve() for p in glob.glob(pattern)]
        files.extend(sorted(matched))
    unique_files: List[Path] = []
    seen = set()
    for file_path in files:
        if file_path not in seen:
            unique_files.append(file_path)
            seen.add(file_path)
    return unique_files


def load_trace(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_traces(
    paths: List[Path],
    *,
    run_id: Optional[str],
    agent_name: Optional[str],
    run_date: Optional[str],
) -> Dict[str, Any]:
    first_trace = load_trace(paths[0])
    config = json.loads(json.dumps(first_trace.get("config", {})))
    if run_id:
        config["run_id"] = run_id
    else:
        config.setdefault("run_id", Path(paths[0]).stem)
    if agent_name:
        config["agent_name"] = agent_name
    if run_date:
        config["date"] = run_date
    config["source_traces"] = [Path(p).name for p in paths]

    merged_results: Dict[str, Any] = {
        "successful_tasks": [],
        "failed_tasks": [],
        "latencies": {},
    }
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_eval_results: Dict[str, Any] = {}
    raw_logging_results: List[Any] = []
    written_correct = vision_correct = 0
    written_total = vision_total = 0

    for path in paths:
        data = load_trace(path)
        results = data.get("results", {})
        merged_results["successful_tasks"].extend(results.get("successful_tasks") or [])
        merged_results["failed_tasks"].extend(results.get("failed_tasks") or [])
        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_results["latencies"].setdefault(
                        key,
                        {
                            "first_call_timestamp": value.get("first_call_timestamp"),
                            "last_call_timestamp": value.get("last_call_timestamp"),
                        },
                    )
                    first_ts = value.get("first_call_timestamp")
                    last_ts = value.get("last_call_timestamp")
                    if first_ts and (
                        not existing.get("first_call_timestamp")
                        or first_ts < existing["first_call_timestamp"]
                    ):
                        existing["first_call_timestamp"] = first_ts
                    if last_ts and (
                        not existing.get("last_call_timestamp")
                        or last_ts > existing["last_call_timestamp"]
                    ):
                        existing["last_call_timestamp"] = last_ts
                else:
                    merged_results["latencies"][key] = (
                        merged_results["latencies"].get(key, 0) + value
                    )
        source_logging = data.get("raw_logging_results") or results.get("raw_logging_results")
        if source_logging:
            raw_logging_results.extend(source_logging)
        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(
                model_name, {"input_tokens": 0, "output_tokens": 0}
            )
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

        capsule_eval = data.get("raw_eval_results") or {}
        for task_id, stats in capsule_eval.items():
            raw_eval_results[task_id] = stats
            written_correct += stats.get("correct_written_answers", 0)
            vision_correct += stats.get("correct_vision_answers", 0)
            written_total += stats.get("total_written_questions", 0)
            vision_total += stats.get("total_vision_questions", 0)

    total_tasks = len(merged_results["successful_tasks"]) + len(merged_results["failed_tasks"])
    merged_results["accuracy"] = (
        len(merged_results["successful_tasks"]) / total_tasks if total_tasks else 0.0
    )
    merged_results["written_accuracy"] = (
        written_correct / written_total if written_total else 0.0
    )
    merged_results["vision_accuracy"] = (
        vision_correct / vision_total if vision_total else 0.0
    )
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
    return merged


def main() -> None:
    args = parse_args()
    input_files = expand_patterns(args.patterns)
    if not input_files:
        raise SystemExit("No trace files matched the provided --input pattern(s).")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output file {args.output} already exists. Use --force to overwrite.")

    run_id = args.run_id or args.output.stem
    merged = merge_traces(
        input_files,
        run_id=run_id,
        agent_name=args.agent_name,
        run_date=args.date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged {len(input_files)} traces into {args.output}")


if __name__ == "__main__":
    main()
