#!/usr/bin/env python3
"""Filter a trace JSON down to one capsule task entry."""

import argparse
import json
from pathlib import Path


def build_filtered_trace(trace: dict, task_id: str) -> dict:
    """Return a minimal structure with just the requested task sections."""
    results = trace.get("results", {})
    latencies = results.get("latencies", {}).get(task_id)
    raw_eval = trace.get("raw_eval_results", {}).get(task_id)
    raw_logs = [
        entry
        for entry in trace.get("raw_logging_results", [])
        if entry.get("attributes", {}).get("weave_task_id") == task_id
    ]

    return {
        "task_id": task_id,
        "latencies": latencies,
        "raw_eval_results": raw_eval,
        "raw_logging_results": raw_logs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only a specific capsule task section from a trace JSON "
            "and write it back out as another trace file in the same directory."
        )
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to the original trace JSON file.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Capsule task id to extract, e.g. capsule-2345790.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output filename. "
            "Defaults to <trace_stem>__<task>.json next to the original trace."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace).expanduser().resolve()

    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    trace = json.loads(trace_path.read_text())
    filtered = build_filtered_trace(trace, args.task)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = trace_path.parent / output_path
    else:
        safe_task = args.task.replace("/", "_")
        output_path = trace_path.parent / f"{trace_path.stem}__{safe_task}.json"

    output_path.write_text(json.dumps(filtered, indent=2))
    print(f"Wrote filtered trace for {args.task} to {output_path}")


if __name__ == "__main__":
    main()
