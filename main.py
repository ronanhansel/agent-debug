from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure hal-harness modules are importable when running from repository root.
REPO_ROOT = Path(__file__).parent.resolve()
HAL_HARNESS_PATH = (REPO_ROOT / "hal-harness").resolve()
if str(HAL_HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HAL_HARNESS_PATH))

from rubric_evaluator import cli as rubric_cli  # type: ignore


def add_rubric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-file", help="Path to a specific trace JSON file.")
    parser.add_argument("--trace-dir", default="traces", help="Directory to scan for trace JSON files.")
    parser.add_argument("--max-tasks", type=int, help="Limit the number of tasks evaluated.")
    parser.add_argument("--rubric-model", help="Override the rubric model provider:deployment.")
    parser.add_argument("--json-mode", action="store_true", help="Request JSON mode completions.")
    parser.add_argument("--rubrics-dir", default="rubrics", help="Directory containing rubric definitions.")
    parser.add_argument("--output-dir", default="rubrics_output", help="Directory to write rubric CSV files.")
    parser.add_argument(
        "--output-mode",
        choices=["csv", "stdout"],
        default="csv",
        help="Select 'csv' to export files or 'stdout' to print results only.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Override reasoning_effort passed to the rubric model.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )


def add_debugger_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rubrics-csv", help="Path to a single rubric CSV.")
    parser.add_argument(
        "--rubrics-output-dir",
        default="../rubrics_output",
        help="Directory containing rubric CSV files (default: ../rubrics_output).",
    )
    parser.add_argument("--traces-dir", default="traces", help="Directory containing reference traces for rubric context.")
    parser.add_argument(
        "--agent-dir",
        default="hal-harness/agents/hal_generalist_agent",
        help="Path to the agent implementation directory.",
    )
    parser.add_argument("--agent-args", help="Path to a JSON file or raw JSON string with agent kwargs.")
    parser.add_argument("--agent-function", default="main.run", help="Agent entrypoint (default: main.run).")
    parser.add_argument("--benchmark-name", default="swebench_verified", help="Benchmark name override.")
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Limit processing to specific task IDs (repeatable).",
    )
    parser.add_argument(
        "--debug-mode",
        choices=["inspect", "run"],
        default="inspect",
        help="Inspector produces guidance; runner applies fix packages.",
    )
    parser.add_argument(
        "--trace-output-dir",
        default="traces/debug_runs",
        help="Where rerun traces should be written when debug-mode=run.",
    )
    parser.add_argument(
        "--fixed-output",
        action="store_true",
        help="Store inspector/runner artifacts in deterministic debug/<benchmark>/<task>/<run_label> directories.",
    )
    parser.add_argument(
        "--run-label",
        default="latest",
        help="Folder name to use when --fixed-output is enabled (default: latest).",
    )
    if not any("--reasoning-effort" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--reasoning-effort",
            choices=["low", "medium", "high"],
            help="Set reasoning_effort for the AutoInspector LLM (debug stage).",
        )
    parser.add_argument(
        "--inspector-model",
        help="Override the AutoInspector LLM (defaults to HAL_AUTOFIX_MODEL env var).",
    )


def build_evaluator_namespace(args: argparse.Namespace) -> argparse.Namespace:
    fields = [
        "trace_file",
        "trace_dir",
        "max_tasks",
        "rubric_model",
        "json_mode",
        "rubrics_dir",
        "output_dir",
        "output_mode",
        "reasoning_effort",
        "yes",
    ]
    payload = {field: getattr(args, field, None) for field in fields}
    return argparse.Namespace(**payload)


def build_debug_namespace(args: argparse.Namespace) -> argparse.Namespace:
    rubrics_csv = args.rubrics_csv
    if rubrics_csv:
        candidate = Path(rubrics_csv)
        if candidate.exists():
            rubrics_csv = _rel_to_hal_harness(rubrics_csv) or rubrics_csv

    rubrics_output_dir = _rel_to_hal_harness(args.rubrics_output_dir) or args.rubrics_output_dir
    traces_dir = _rel_to_hal_harness(args.traces_dir) or args.traces_dir
    agent_dir = _rel_to_hal_harness(args.agent_dir) or args.agent_dir
    agent_args = args.agent_args
    if agent_args:
        agent_args_path = Path(agent_args)
        if agent_args_path.exists():
            agent_args = _rel_to_hal_harness(agent_args) or agent_args

    trace_output_dir = _rel_to_hal_harness(args.trace_output_dir) or args.trace_output_dir

    return argparse.Namespace(
        rubrics_csv=rubrics_csv,
        rubrics_output_dir=rubrics_output_dir,
        fixable_output_dir=None,
        traces_dir=traces_dir,
        agent_dir=agent_dir,
        agent_args=agent_args,
        agent_function=args.agent_function,
        benchmark_name=args.benchmark_name,
        task_ids=args.task_ids,
        mode=args.debug_mode,
        trace_output_dir=trace_output_dir,
        reasoning_effort=args.reasoning_effort,
        fixed_output=getattr(args, "fixed_output", False),
        run_label=getattr(args, "run_label", "latest"),
        inspector_model=getattr(args, "inspector_model", None),
    )


def _rel_to_hal_harness(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return path_value
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    try:
        return str(path.relative_to(HAL_HARNESS_PATH))
    except ValueError:
        return str(Path(os.path.relpath(path, HAL_HARNESS_PATH)))


def serialize_debug_cli(args: argparse.Namespace) -> str:
    parts: List[str] = []
    if args.rubrics_csv:
        parts += ["--rubrics-csv", _rel_to_hal_harness(args.rubrics_csv) or args.rubrics_csv]
    else:
        parts += ["--rubrics-output-dir", _rel_to_hal_harness(args.rubrics_output_dir) or args.rubrics_output_dir]
    parts += ["--traces-dir", _rel_to_hal_harness(args.traces_dir) or args.traces_dir]
    parts += ["--agent-dir", _rel_to_hal_harness(args.agent_dir) or args.agent_dir]
    if args.agent_args:
        agent_args_path = Path(args.agent_args)
        if agent_args_path.exists():
            parts += ["--agent-args", _rel_to_hal_harness(args.agent_args) or args.agent_args]
        else:
            parts += ["--agent-args", args.agent_args]
    if args.agent_function:
        parts += ["--agent-function", args.agent_function]
    if args.benchmark_name:
        parts += ["--benchmark-name", args.benchmark_name]
    if args.task_ids:
        for task_id in args.task_ids:
            parts += ["--task-id", task_id]
    parts += ["--mode", args.debug_mode]
    parts += ["--trace-output-dir", _rel_to_hal_harness(args.trace_output_dir) or args.trace_output_dir]
    if getattr(args, "reasoning_effort", None):
        parts += ["--reasoning-effort", args.reasoning_effort]
    if getattr(args, "inspector_model", None):
        parts += ["--inspector-model", args.inspector_model]
    if getattr(args, "fixed_output", False):
        parts.append("--fixed-output")
        parts += ["--run-label", getattr(args, "run_label", "latest")]
    return "python scripts/auto_debug_batch.py " + " ".join(parts)


def _resolve_repo_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def run_evaluate(args: argparse.Namespace) -> None:
    args.trace_file = _resolve_repo_path(getattr(args, "trace_file", None))
    args.trace_dir = _resolve_repo_path(getattr(args, "trace_dir", None) or "traces")
    args.rubrics_dir = _resolve_repo_path(getattr(args, "rubrics_dir", None) or "rubrics")
    args.output_dir = _resolve_repo_path(getattr(args, "output_dir", None) or "rubrics_output")
    rubric_cli.run(build_evaluator_namespace(args))


def run_debug(args: argparse.Namespace) -> None:
    from scripts import auto_debug_batch as debugger_cli  # type: ignore

    args.rubrics_output_dir = _resolve_repo_path(args.rubrics_output_dir)
    args.traces_dir = _resolve_repo_path(args.traces_dir)
    args.agent_dir = _resolve_repo_path(args.agent_dir)
    if args.agent_args and Path(args.agent_args).exists():
        args.agent_args = _resolve_repo_path(args.agent_args)
    args.trace_output_dir = _resolve_repo_path(args.trace_output_dir)

    debug_namespace = build_debug_namespace(args)
    rerun_command = serialize_debug_cli(args)
    original_cwd = Path.cwd()
    try:
        os.chdir(HAL_HARNESS_PATH)
        asyncio.run(debugger_cli._run_pipeline(debug_namespace, rerun_command))  # type: ignore[attr-defined]
    finally:
        os.chdir(original_cwd)


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = getattr(args, "output_dir", None) or getattr(args, "rubrics_output_dir", None) or "rubrics_output"
    output_dir = _resolve_repo_path(output_dir)
    args.output_dir = output_dir

    if not args.skip_evaluate:
        run_evaluate(args)
    else:
        print("⚠️  Skipping rubric evaluation as requested (--skip-evaluate).")

    args.rubrics_output_dir = output_dir

    if not args.skip_debug:
        run_debug(args)
    else:
        print("⚠️  Skipping debugger phase as requested (--skip-debug).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified rubric evaluation and auto-debug pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run the rubric evaluator only.")
    add_rubric_args(eval_parser)
    eval_parser.set_defaults(handler=run_evaluate)

    debug_parser = subparsers.add_parser("debug", help="Run the inspector or runner on rubric CSVs.")
    add_debugger_args(debug_parser)
    debug_parser.set_defaults(handler=run_debug)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run rubric evaluation followed by the debugger.")
    add_rubric_args(pipeline_parser)
    add_debugger_args(pipeline_parser)
    pipeline_parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the rubric evaluation stage (expects CSVs to already exist).",
    )
    pipeline_parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Skip the debugger stage after evaluation completes.",
    )
    pipeline_parser.set_defaults(handler=run_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command provided.")
    handler(args)


if __name__ == "__main__":
    main()
