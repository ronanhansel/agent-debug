#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_DIR = REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"
DEFAULT_AGENT_ARGS = REPO_ROOT / "agent_args.azure.json"
DEFAULT_TRACE_DIR = REPO_ROOT / "traces"
DEFAULT_RUBRIC_DIR = REPO_ROOT / "rubrics"
DEFAULT_RUBRIC_OUTPUT = REPO_ROOT / "rubrics_output"
DEFAULT_TRACE_OUTPUT_DIR = REPO_ROOT / "traces" / "debug_runs"
DEBUG_ROOT = REPO_ROOT / "debug"
LOG_ROOT = REPO_ROOT / "log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end fixer pipeline: rubric grading -> inspection -> codex -> runner -> re-eval."
    )
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="Directory containing baseline traces.")
    parser.add_argument("--rubrics-dir", default=str(DEFAULT_RUBRIC_DIR), help="Directory containing rubric prompts.")
    parser.add_argument(
        "--rubrics-output-dir",
        default=str(DEFAULT_RUBRIC_OUTPUT),
        help="Directory for rubric CSV outputs (structure: rubric/trace.csv).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(DEFAULT_AGENT_DIR),
        help="Agent workspace containing main.py entrypoint.",
    )
    parser.add_argument(
        "--agent-args",
        default=str(DEFAULT_AGENT_ARGS),
        help="JSON file with agent keyword arguments (optional).",
    )
    parser.add_argument("--agent-function", default="main.run", help="Agent entrypoint (default: main.run).")
    parser.add_argument(
        "--benchmark-name",
        default="swebench_verified_mini",
        help="Benchmark identifier used for task metadata and directory layout.",
    )
    parser.add_argument(
        "--trace-output-dir",
        default=str(DEFAULT_TRACE_OUTPUT_DIR),
        help="Directory where runner writes synthetic traces (default: traces/debug_runs).",
    )
    parser.add_argument(
        "--rubric-model",
        required=False,
        help="LLM provider:model for rubric grading (e.g., azure_openai:o3-mini).",
    )
    parser.add_argument(
        "--inspector-model",
        required=False,
        help="LLM provider:model for AutoInspector guidance.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Default reasoning effort hint (rubrics + inspector, unless stage-specific flags override).",
    )
    parser.add_argument(
        "--rubric-reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort override for rubric grading stage.",
    )
    parser.add_argument(
        "--inspector-reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort override for inspector stage.",
    )
    parser.add_argument(
        "--runner-reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort override injected into runner agent args.",
    )
    parser.add_argument(
        "--runner-model",
        help="Override the agent model used during runner replays (updates agent_args model_name).",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Executable used to dispatch inspection reports to Codex (default: codex).",
    )
    parser.add_argument(
        "--skip-codex",
        action="store_true",
        help="Skip invoking Codex (useful for dry-runs when fixes already exist).",
    )
    parser.add_argument(
        "--skip-runner",
        action="store_true",
        help="Skip the runner + re-evaluation stage (only collect inspections).",
    )
    parser.add_argument(
        "--skip-inspector",
        action="store_true",
        help="Reuse previously generated inspection reports without regenerating them.",
    )
    parser.add_argument(
        "--defer-rubric-eval",
        action="store_true",
        help="Run rubric evaluation only after all reruns complete (instead of per task).",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Limit the pipeline to specific task IDs (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks even if fixes/<benchmark>/<task>/status.json reports success.",
    )
    parser.add_argument(
        "--skip-rubric-eval",
        action="store_true",
        help="Reuse existing rubrics_output files without re-grading traces.",
    )
    return parser.parse_args()


class FixingPipelineLogger:
    def __init__(self) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.root = LOG_ROOT / f"fixing-pipeline-{timestamp}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "pipeline.log"
        self.log("Fixing pipeline started.")

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        line = f"[{timestamp}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_json(self, name: str, payload: Dict[str, object]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def load_agent_args_payload(agent_args_value: str | None) -> Dict[str, Any]:
    if not agent_args_value:
        return {}
    path = Path(agent_args_value)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - user input
            raise ValueError(f"Failed to parse JSON in {path}: {exc}") from exc
    try:
        return json.loads(agent_args_value)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"Invalid JSON passed via --agent-args: {exc}") from exc


def default_agent_args_cli_value(agent_args_value: str | None) -> str | None:
    if not agent_args_value:
        return None
    path = Path(agent_args_value)
    if path.exists():
        return str(path)
    return agent_args_value


def runner_agent_args_override(args: argparse.Namespace, runner_effort: str | None) -> str | None:
    if not args.runner_model and not runner_effort:
        return None
    base = load_agent_args_payload(args.agent_args)
    if args.runner_model:
        base["model_name"] = args.runner_model
    if runner_effort:
        base["reasoning_effort"] = runner_effort
    return json.dumps(base)


def stage_reasoning_effort(stage_value: str | None, default_value: str | None) -> str | None:
    return stage_value or default_value


def run_command(
    cmd: Sequence[str],
    logger: FixingPipelineLogger,
    env: Dict[str, str] | None = None,
    *,
    display_override: str | None = None,
) -> None:
    display = display_override or shlex.join(str(part) for part in cmd)
    logger.log(f"Executing: {display}")
    subprocess.run(list(map(str, cmd)), check=True, cwd=REPO_ROOT, env=env)


def discover_traces(trace_dir: Path) -> List[Path]:
    return sorted(p for p in trace_dir.glob("*.json") if p.is_file())


def chunked(sequence: Sequence[str], size: int) -> List[List[str]]:
    return [list(sequence[i : i + size]) for i in range(0, len(sequence), size)]


def trace_matches_benchmark(trace_path: Path, benchmark_name: str) -> bool:
    if not benchmark_name:
        return True
    normalized = benchmark_name.strip().lower()
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        stem = trace_path.stem.lower()
        return normalized in stem or stem.startswith(normalized)
    config = data.get("config") or {}
    candidate = config.get("benchmark_name") or config.get("benchmark")
    if isinstance(candidate, str):
        candidate_norm = candidate.strip().lower()
        if candidate_norm == normalized:
            return True
        if candidate_norm.startswith(normalized) or normalized.startswith(candidate_norm):
            return True
    stem = trace_path.stem.lower()
    return normalized in stem or stem.startswith(normalized)


def evaluate_trace(
    trace_path: Path,
    args: argparse.Namespace,
    logger: FixingPipelineLogger,
) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "evaluate",
        "--trace-file",
        str(trace_path),
        "--rubrics-dir",
        str(Path(args.rubrics_dir)),
        "--output-dir",
        str(Path(args.rubrics_output_dir)),
        "--output-mode",
        "csv",
        "--yes",
    ]
    if args.rubric_model:
        cmd += ["--rubric-model", args.rubric_model]
    rubric_effort = stage_reasoning_effort(args.rubric_reasoning_effort, args.reasoning_effort)
    if rubric_effort:
        cmd += ["--reasoning-effort", rubric_effort]
    run_command(cmd, logger)


def summarize_rubrics(rubrics_root: Path) -> Dict[str, object]:
    combined: Dict[Tuple[str, str], Dict[str, Any]] = {}
    raw_count = 0
    for rubric_dir in sorted(rubrics_root.iterdir()):
        if not rubric_dir.is_dir():
            continue
        rubric_id = rubric_dir.name
        for csv_file in sorted(rubric_dir.glob("*.csv")):
            with csv_file.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    raw_count += 1
                    task_id = (row.get("task_id") or "").strip()
                    grade_raw = (row.get("grade") or "").strip()
                    try:
                        grade_value = float(grade_raw or 0)
                    except ValueError:
                        grade_value = 0.0
                    if task_id and grade_value >= 1.0:
                        criteria = (row.get("criteria") or rubric_id).strip()
                        key = (task_id, criteria)
                        entry = combined.setdefault(
                            key,
                            {
                                "task_id": task_id,
                                "criteria": criteria,
                                "grade": 0.0,
                                "grade_values": set(),
                                "explanations": [],
                                "model_runs": set(),
                                "source_csvs": set(),
                                "occurrences": 0,
                            },
                        )
                        entry["grade"] = max(entry["grade"], grade_value)
                        entry["grade_values"].add(grade_raw or f"{grade_value:.2f}")
                        explanation = (row.get("explanation") or "").strip()
                        if explanation:
                            entry["explanations"].append(explanation)
                        entry["model_runs"].add((row.get("model_run") or csv_file.stem).strip())
                        entry["source_csvs"].add(str(csv_file))
                        entry["occurrences"] += 1

    entries: List[Dict[str, object]] = []
    for entry in combined.values():
        explanation_text = "\n\n---\n\n".join(entry["explanations"]) if entry["explanations"] else ""
        entries.append(
            {
                "task_id": entry["task_id"],
                "criteria": entry["criteria"],
                "grade": entry["grade"],
                "grade_raw": ", ".join(sorted(entry["grade_values"])),
                "explanation": explanation_text,
                "explanations": entry["explanations"],
                "model_runs": sorted(m for m in entry["model_runs"] if m),
                "source_csvs": sorted(entry["source_csvs"]),
                "occurrences": entry["occurrences"],
            }
        )
    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_entries": len(entries),
        "total_raw_rows": raw_count,
        "tasks": sorted(entries, key=lambda item: (item["task_id"], item["criteria"])),
    }
    return summary


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def inspection_report_path(task_id: str, benchmark: str) -> Path:
    sanitized = sanitize(task_id)
    return DEBUG_ROOT / benchmark / sanitized / "latest" / "inspection_report.json"


def task_run_dir(task_id: str, benchmark: str) -> Path:
    sanitized = sanitize(task_id)
    return DEBUG_ROOT / benchmark / sanitized / "latest"


def fix_status_file(task_id: str, benchmark: str) -> Path:
    return REPO_ROOT / "fixes" / sanitize(benchmark) / sanitize(task_id) / "status.json"


def ensure_inspector(args: argparse.Namespace, logger: FixingPipelineLogger) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "debug",
        "--rubrics-output-dir",
        str(Path(args.rubrics_output_dir)),
        "--traces-dir",
        str(Path(args.trace_dir)),
        "--agent-dir",
        str(Path(args.agent_dir)),
        "--agent-function",
        args.agent_function,
        "--benchmark-name",
        args.benchmark_name,
        "--debug-mode",
        "inspect",
        "--trace-output-dir",
        str(Path(args.trace_output_dir)),
        "--fixed-output",
        "--run-label",
        "latest",
    ]
    inspector_agent_args = default_agent_args_cli_value(args.agent_args)
    if inspector_agent_args:
        cmd += ["--agent-args", inspector_agent_args]
    inspector_effort = stage_reasoning_effort(args.inspector_reasoning_effort, args.reasoning_effort)
    if inspector_effort:
        cmd += ["--reasoning-effort", inspector_effort]
    if args.inspector_model:
        cmd += ["--inspector-model", args.inspector_model]
    run_command(cmd, logger)


def run_runner(task_id: str, args: argparse.Namespace, logger: FixingPipelineLogger) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "debug",
        "--rubrics-output-dir",
        str(Path(args.rubrics_output_dir)),
        "--traces-dir",
        str(Path(args.trace_dir)),
        "--agent-dir",
        str(Path(args.agent_dir)),
        "--agent-function",
        args.agent_function,
        "--benchmark-name",
        args.benchmark_name,
        "--task-id",
        task_id,
        "--debug-mode",
        "run",
        "--trace-output-dir",
        str(Path(args.trace_output_dir)),
        "--fixed-output",
        "--run-label",
        "latest",
    ]
    runner_effort = stage_reasoning_effort(args.runner_reasoning_effort, args.reasoning_effort)
    runner_agent_args = runner_agent_args_override(args, runner_effort)
    if not runner_agent_args:
        runner_agent_args = default_agent_args_cli_value(args.agent_args)
    if runner_agent_args:
        cmd += ["--agent-args", runner_agent_args]
    run_command(cmd, logger)


def load_inspection_payload(task_id: str, args: argparse.Namespace, logger: FixingPipelineLogger) -> Dict[str, Any] | None:
    path = inspection_report_path(task_id, args.benchmark_name)
    if not path.exists():
        logger.log(f"Inspection report missing for {task_id} at {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.log(f"Inspection report for {task_id} is invalid JSON: {exc}")
        return None


def build_codex_prompt(task_ids: List[str], args: argparse.Namespace, logger: FixingPipelineLogger) -> str | None:
    tasks_payload: List[Dict[str, Any]] = []
    for task_id in task_ids:
        report = load_inspection_payload(task_id, args, logger)
        if report is None:
            continue
        tasks_payload.append(
            {
                "task_id": task_id,
                "report": report,
                "inspection_report_path": str(inspection_report_path(task_id, args.benchmark_name)),
            }
        )
    if not tasks_payload:
        return None
    common_instructions = (
        "For each task, follow the inspector guidance and create or update a fix package under:\n"
        "  fixes/<benchmark>/<task_id>/\n\n"
        "Use one or more of:\n"
        "- `agent/` overlay files (preferred)\n"
        "- `patch.diff` (only if overlays are impractical)\n"
        "- `problem_statement.txt` (to add environment-specific, actionable steps)\n"
        "- `input_override.json` / `env_override.json` (to adjust inputs/env vars)\n\n"
        "After writing fixes for a task, validate by rerunning the single-task command in its report.\n"
        "Do not directly edit benchmark source files outside the fix package."
    )

    lines: List[str] = []
    lines.append("[SYSTEM PROMPT]")
    lines.append(
        "You are a coding agent operating in this repository. Your job is to implement fix packages for the tasks "
        "listed below. Do not modify repository files directly; place all changes under the fix folders."
    )
    lines.append("")
    lines.append("[BENCHMARK]")
    lines.append(str(args.benchmark_name))
    lines.append("")
    lines.append("[GLOBAL INSTRUCTIONS]")
    lines.append(common_instructions)
    lines.append("")

    for item in tasks_payload:
        task_id = item["task_id"]
        report: Dict[str, Any] = item["report"]
        ctx: Dict[str, Any] = report.get("coding_agent_context") or {}
        fix_folder = ctx.get("fix_folder") or f"fixes/{sanitize(args.benchmark_name)}/{sanitize(task_id)}"
        lines.append("=" * 80)
        lines.append(f"[TASK ID] {task_id}")
        lines.append(f"[INSPECTION REPORT] {item['inspection_report_path']}")
        lines.append(f"[FIX FOLDER] {fix_folder}")
        lines.append("")

        analysis = (report.get("analysis") or "").strip()
        rationale = (report.get("rationale") or "").strip()
        if analysis:
            lines.append("[INSPECTOR ANALYSIS]")
            lines.append(analysis)
            lines.append("")
        if rationale:
            lines.append("[INSPECTOR RATIONALE]")
            lines.append(rationale)
            lines.append("")

        recommended_files = report.get("recommended_files") or []
        if recommended_files:
            lines.append("[RECOMMENDED FILES]")
            for path in recommended_files:
                lines.append(f"- {path}")
            lines.append("")

        recommended_actions = report.get("recommended_actions") or []
        if recommended_actions:
            lines.append("[RECOMMENDED ACTIONS]")
            for action in recommended_actions:
                lines.append(f"- {str(action).strip()}")
            lines.append("")

        next_steps = (report.get("next_steps") or "").strip()
        if next_steps:
            lines.append("[NEXT STEPS]")
            lines.append(next_steps)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _write_codex_payload_file(
    *,
    payload_json: str,
    logger: FixingPipelineLogger,
    batch_index: int,
) -> Path:
    """Write a pretty-printed Codex payload for debugging/log review."""
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        parsed = {"raw_payload": payload_json}
    return logger.write_json(f"codex_payload_batch_{batch_index}.json", parsed)


def _write_codex_prompt_file(*, prompt: str, logger: FixingPipelineLogger, batch_index: int) -> Path:
    path = logger.root / f"codex_prompt_batch_{batch_index}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def dispatch_codex_batches(task_ids: List[str], args: argparse.Namespace, logger: FixingPipelineLogger) -> None:
    if not task_ids:
        return
    batches = chunked(task_ids, 5)
    for index, batch in enumerate(batches):
        prompt = build_codex_prompt(batch, args, logger)
        if not prompt:
            continue

        prompt_path = _write_codex_prompt_file(prompt=prompt, logger=logger, batch_index=index)
        logger.log(f"Codex prompt written to {prompt_path}")

        if index == 0:
            cmd = [args.codex_bin, "exec", prompt]
        else:
            cmd = [args.codex_bin, "exec", "resume", "--last", prompt]

        # Keep the actual argv unchanged (Codex expects the JSON string), but make logs readable.
        display_cmd = list(cmd)
        display_cmd[-1] = f"@{prompt_path}"
        run_command(cmd, logger, display_override=shlex.join(str(part) for part in display_cmd))


def failing_rubrics_for_trace(
    rubrics_root: Path,
    trace_label: str,
    task_id: str,
) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for rubric_dir in sorted(rubrics_root.iterdir()):
        if not rubric_dir.is_dir():
            continue
        csv_file = rubric_dir / f"{trace_label}.csv"
        if not csv_file.exists():
            continue
        with csv_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (row.get("task_id") or "").strip() != task_id:
                    continue
                try:
                    grade_value = float((row.get("grade") or "0").strip() or 0)
                except ValueError:
                    grade_value = 0.0
                if grade_value > 0:
                    failures.append(
                        {
                            "criteria": (row.get("criteria") or rubric_dir.name).strip(),
                            "grade": str(grade_value),
                            "explanation": (row.get("explanation") or "").strip(),
                            "source_csv": str(csv_file),
                        }
                    )
    return failures


def execute_task_cycle(
    task_id: str,
    args: argparse.Namespace,
    logger: FixingPipelineLogger,
    skip_runner: bool,
    force: bool,
    defer_rubric_eval: bool,
) -> Dict[str, object]:
    result: Dict[str, object] = {"task_id": task_id, "status": "pending"}
    status_path = fix_status_file(task_id, args.benchmark_name)
    if status_path.exists() and not force:
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("status") == "passed":
            logger.log(f"Skipping {task_id}: already marked as passed in {status_path}")
            result["status"] = "skipped"
            return result

    if skip_runner:
        result["status"] = "inspection_only"
        return result

    try:
        run_runner(task_id, args, logger)
    except subprocess.CalledProcessError as exc:
        logger.log(f"Runner failed for {task_id} (exit={exc.returncode}).")
        result["status"] = "runner_failed"
        return result

    run_dir = task_run_dir(task_id, args.benchmark_name)
    pointer_path = run_dir / "trace_pointer.txt"
    if not pointer_path.exists():
        logger.log(f"Runner did not record a trace pointer for {task_id} (expected {pointer_path}).")
        result["status"] = "missing_trace"
        return result

    trace_path = Path(pointer_path.read_text(encoding="utf-8").strip())
    if not trace_path.exists():
        logger.log(f"Recorded trace path {trace_path} missing for {task_id}.")
        result["status"] = "missing_trace_file"
        return result

    result["trace_path"] = str(trace_path)
    if defer_rubric_eval or args.skip_rubric_eval:
        logger.log(f"Deferring rubric evaluation for task {task_id} (trace={trace_path}).")
        return result

    return _evaluate_trace_and_update_status(
        task_id=task_id,
        args=args,
        logger=logger,
        result=result,
        trace_path=trace_path,
        status_path=status_path,
    )


def _evaluate_trace_and_update_status(
    *,
    task_id: str,
    args: argparse.Namespace,
    logger: FixingPipelineLogger,
    result: Dict[str, object],
    trace_path: Path,
    status_path: Path,
) -> Dict[str, object]:
    try:
        evaluate_trace(trace_path, args, logger)
    except subprocess.CalledProcessError as exc:
        logger.log(f"Re-evaluating trace {trace_path} failed (exit={exc.returncode}).")
        result["status"] = "regrade_failed"
        result["trace_path"] = str(trace_path)
        return result

    trace_label = trace_path.stem
    rubrics_root = Path(args.rubrics_output_dir)
    failures = failing_rubrics_for_trace(rubrics_root, trace_label, task_id)
    status_payload = {
        "task_id": task_id,
        "benchmark": args.benchmark_name,
        "trace_path": str(trace_path),
        "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "failures": failures,
    }
    if failures:
        logger.log(f"Task {task_id} still failing {len(failures)} rubric(s).")
        result["status"] = "still_failing"
        result["trace_path"] = str(trace_path)
        result["failures"] = failures
        status_payload["status"] = "failed"
    else:
        logger.log(f"Task {task_id} passed re-evaluation (trace={trace_label}).")
        result["status"] = "passed"
        result["trace_path"] = str(trace_path)
        status_payload["status"] = "passed"

    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    logger = FixingPipelineLogger()
    trace_dir = Path(args.trace_dir)
    rubrics_output_dir = Path(args.rubrics_output_dir)
    rubrics_output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_rubric_eval:
        logger.log("Skipping rubric evaluation; reusing existing rubrics_output files.")
    else:
        traces = discover_traces(trace_dir)
        if not traces:
            logger.log(f"No traces found in {trace_dir}.")
            return
        benchmark_traces = [trace for trace in traces if trace_matches_benchmark(trace, args.benchmark_name)]
        if not benchmark_traces:
            logger.log(
                f"No trace files matching benchmark '{args.benchmark_name}' were found under {trace_dir}."
            )
            return
        logger.log(f"Evaluating rubrics for {len(benchmark_traces)} trace(s) matching benchmark {args.benchmark_name}.")
        for trace in benchmark_traces:
            try:
                evaluate_trace(trace, args, logger)
            except subprocess.CalledProcessError as exc:
                logger.log(f"Rubric evaluation failed for {trace} (exit={exc.returncode}).")
                return

    summary = summarize_rubrics(rubrics_output_dir)
    logger.write_json("rubric_summary_initial.json", summary)
    if not summary["tasks"]:
        logger.log("Rubric summary contained zero failing entries. Nothing to fix.")
        return

    if args.skip_inspector:
        logger.log("Skipping inspector; reusing existing inspection reports.")
    else:
        try:
            ensure_inspector(args, logger)
        except subprocess.CalledProcessError as exc:
            logger.log(f"Inspector pipeline failed (exit={exc.returncode}).")
            return

    debug_summary_path = DEBUG_ROOT / args.benchmark_name / "rubric_summary.json"
    if debug_summary_path.exists():
        summary = json.loads(debug_summary_path.read_text(encoding="utf-8"))
    else:
        logger.log(f"Inspector did not emit {debug_summary_path}; using pre-inspection summary.")

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    task_filters = {tid.strip() for tid in (args.task_ids or []) if tid}
    for entry in summary.get("tasks", []):
        task_id = entry.get("task_id")
        grade = entry.get("grade")
        try:
            grade_value = float(grade)
        except (TypeError, ValueError):
            grade_value = 0.0
        if task_id and grade_value >= 1.0:
            if task_filters and task_id not in task_filters:
                continue
            grouped[task_id].append(entry)

    if not grouped:
        logger.log("No failing tasks remain after filtering.")
        return

    ordered_tasks = sorted(grouped.keys())
    results: List[Dict[str, object]] = []
    tasks_with_reports: List[str] = []
    for task_id in ordered_tasks:
        report_path = inspection_report_path(task_id, args.benchmark_name)
        if not report_path.exists():
            logger.log(f"Inspection report missing for {task_id} at {report_path}")
            results.append({"task_id": task_id, "status": "missing_inspection"})
        else:
            tasks_with_reports.append(task_id)

    if not tasks_with_reports:
        logger.log("No inspection reports available; cannot proceed to codex/runner stage.")
        logger.write_json("fixing_results.json", {"results": results})
        logger.log(f"Fixing pipeline finished. Logs available at {logger.root}")
        return

    logger.log(f"Processing {len(tasks_with_reports)} task(s) through codex + runner.")

    if not args.skip_codex:
        try:
            dispatch_codex_batches(tasks_with_reports, args, logger)
        except FileNotFoundError:
            logger.log(f"Codex binary '{args.codex_bin}' not found. Rerun with --codex-bin or --skip-codex.")
            return
        except subprocess.CalledProcessError as exc:
            logger.log(f"Codex execution failed (exit={exc.returncode}).")
            return

    for task_id in tasks_with_reports:
        outcome = execute_task_cycle(
            task_id=task_id,
            args=args,
            logger=logger,
            skip_runner=args.skip_runner,
            force=args.force,
            defer_rubric_eval=args.defer_rubric_eval,
        )
        results.append(outcome)

    if args.defer_rubric_eval and not args.skip_rubric_eval:
        _run_deferred_rubric_checks(results, args, logger)

    logger.write_json("fixing_results.json", {"results": results})
    logger.log(f"Fixing pipeline finished. Logs available at {logger.root}")


def _run_deferred_rubric_checks(
    results: List[Dict[str, object]],
    args: argparse.Namespace,
    logger: FixingPipelineLogger,
) -> None:
    for entry in results:
        task_id = entry.get("task_id")
        trace_path = entry.get("trace_path")
        if not task_id or not trace_path:
            continue
        trace_path_obj = Path(str(trace_path))
        logger.log(f"Running deferred rubric evaluation for trace {trace_path_obj}.")
        status_path = fix_status_file(task_id, args.benchmark_name)
        _evaluate_trace_and_update_status(
            task_id=task_id,
            args=args,
            logger=logger,
            result=entry,
            trace_path=trace_path_obj,
            status_path=status_path,
        )


if __name__ == "__main__":
    main()
