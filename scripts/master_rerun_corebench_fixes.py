#!/usr/bin/env python3
"""
Master rerun pipeline for CoreBench fix packages.

Given a mapping of `model_name -> baseline_trace_run_id`, this script:
  1) loads the baseline trace JSON to find failed task_ids
  2) filters to tasks that have fixes present under fixes/<benchmark>/<task_id>/
  3) reruns those tasks with the specified model using scripts/run_corebench_fixes.py
  4) writes a deterministic merged trace per model/run mapping

Run this script using the `hal` conda env python.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned.strip("_") or "unknown"


def _resolve_trace_path(traces_dir: Path, run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        in_traces = traces_dir / candidate
        if in_traces.exists():
            return in_traces

    # Common case: config value is a "run id" stem.
    for suffix in (".json", "_UPLOAD.json"):
        maybe = traces_dir / f"{run_id_or_path}{suffix}"
        if maybe.exists():
            return maybe

    # Handle callers passing a filename that already ends with .json or _UPLOAD.json.
    if run_id_or_path.endswith(".json"):
        maybe = traces_dir / run_id_or_path
        if maybe.exists():
            return maybe
        stem = run_id_or_path[: -len(".json")]
        maybe = traces_dir / f"{stem}_UPLOAD.json"
        if maybe.exists():
            return maybe
    if run_id_or_path.endswith("_UPLOAD.json"):
        stem = run_id_or_path[: -len("_UPLOAD.json")]
        maybe = traces_dir / f"{stem}.json"
        if maybe.exists():
            return maybe

    raise FileNotFoundError(f"Could not resolve baseline trace: {run_id_or_path} (searched under {traces_dir})")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ModelRunSpec:
    model_name: str
    baseline_trace: str
    reasoning_effort: str | None = None
    model_id: str | None = None


class _LiveState:
    def __init__(self, *, total_specs: int):
        self._lock = Lock()
        self.total_specs = total_specs
        self.completed_specs = 0
        self.failed_specs = 0
        self.running_specs = 0
        self.planned_capsules = 0
        self.completed_capsules = 0
        self.running_capsules = 0

    def _print(self, message: str) -> None:
        with self._lock:
            print(message, flush=True)

    def spec_started(self, *, model_name: str, capsule_total: int) -> None:
        with self._lock:
            self.running_specs += 1
            self.planned_capsules += capsule_total
            print(
                f"[spec][start] model={model_name} capsules={capsule_total} "
                f"(specs {self.completed_specs}/{self.total_specs} done, {self.running_specs} running; "
                f"capsules {self.completed_capsules}/{self.planned_capsules} done, {self.running_capsules} running)",
                flush=True,
            )

    def capsule_started(self, *, model_name: str, capsule_id: str) -> None:
        with self._lock:
            self.running_capsules += 1
            print(
                f"[capsule][start] model={model_name} capsule={capsule_id} "
                f"(capsules {self.completed_capsules}/{self.planned_capsules} done, {self.running_capsules} running)",
                flush=True,
            )

    def capsule_done(self, *, model_name: str, capsule_id: str) -> None:
        with self._lock:
            self.running_capsules = max(0, self.running_capsules - 1)
            self.completed_capsules += 1
            print(
                f"[capsule][done]  model={model_name} capsule={capsule_id} "
                f"(capsules {self.completed_capsules}/{self.planned_capsules} done, {self.running_capsules} running)",
                flush=True,
            )

    def spec_finished(self, *, model_name: str, status: str, merged_trace: str | None) -> None:
        with self._lock:
            self.running_specs = max(0, self.running_specs - 1)
            if status in {"completed", "dry_run", "skipped_no_fixes"}:
                self.completed_specs += 1
            elif status.startswith("failed"):
                self.failed_specs += 1
            extra = f" merged={merged_trace}" if merged_trace else ""
            print(
                f"[spec][{status}] model={model_name}{extra} "
                f"(specs {self.completed_specs}/{self.total_specs} done, {self.failed_specs} failed, {self.running_specs} running; "
                f"capsules {self.completed_capsules}/{self.planned_capsules} done, {self.running_capsules} running)",
                flush=True,
            )

    def stream_line(self, *, model_name: str, line: str) -> None:
        self._print(f"[{model_name}] {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun CoreBench failures with fixes per model/run mapping.")
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to JSON file mapping model_name -> baseline trace run_id (or trace path).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix to prepend to merged run_ids/trace filenames and downstream wandb project/group (default: empty).",
    )
    parser.add_argument(
        "--traces-dir",
        default="traces",
        help="Directory containing baseline trace JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--fixes-root",
        default="fixes",
        help="Fixes root directory (default: %(default)s). Expects fixes/<benchmark>/<task_id>/ folders.",
    )
    parser.add_argument(
        "--agent-dir",
        default="hal-harness/agents/hal_generalist_agent",
        help="Agent directory for HAL eval (default: %(default)s).",
    )
    parser.add_argument(
        "--base-agent-args",
        default="agent_args.azure.json",
        help="Base agent args JSON (default: %(default)s). model_name will be overridden per mapping entry.",
    )
    parser.add_argument(
        "--rubric-model",
        required=True,
        help="Rubric model identifier passed through to scripts/run_corebench_fixes.py.",
    )
    parser.add_argument(
        "--corebench-dataset",
        help="Optional path to core_test.json to pass through to scripts/run_corebench_fixes.py.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="If set, inject reasoning_effort into agent args for all mapping entries (unless overridden per entry).",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Pass --docker through to scripts/run_corebench_fixes.py (recommended if Weave networking is required).",
    )
    parser.add_argument(
        "--vm",
        action="store_true",
        help="Pass --vm through to scripts/run_corebench_fixes.py (run tasks on a VM, e.g. for GPU).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Pass --wandb-mode through to scripts/run_corebench_fixes.py.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Pass --skip-install through to scripts/run_corebench_fixes.py.",
    )
    parser.add_argument(
        "--skip-rubrics",
        action="store_true",
        help="Skip rubric evaluation during reruns (passes --skip-rubrics).",
    )
    parser.add_argument(
        "--output-dir",
        default="traces",
        help="Directory to write merged traces (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rerun without executing.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of rerun subprocesses to run concurrently (default: %(default)s).",
    )
    parser.add_argument(
        "--max-parallel-capsules",
        type=int,
        default=1,
        help="Maximum number of capsules to run concurrently within each rerun subprocess (default: %(default)s).",
    )
    parser.add_argument(
        "--stream-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream child process logs to stdout while running (default: enabled).",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> List[ModelRunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--mapping-file must contain an object: {\"model_name\": \"baseline_run_id\", ...}")
    specs: List[ModelRunSpec] = []
    for model_name, baseline in payload.items():
        if not isinstance(model_name, str):
            raise ValueError("Mapping keys must be strings (model_name).")

        if isinstance(baseline, str):
            specs.append(ModelRunSpec(model_name=model_name, baseline_trace=baseline))
            continue

        if isinstance(baseline, dict):
            baseline_trace = baseline.get("baseline_trace") or baseline.get("baseline_run_id") or baseline.get("trace")
            if not isinstance(baseline_trace, str) or not baseline_trace.strip():
                raise ValueError(
                    f"Mapping entry for {model_name} must include baseline_trace (string). "
                    "Example: {\"baseline_trace\": \"corebench_..._UPLOAD\"}"
                )
            reasoning_effort = baseline.get("reasoning_effort")
            if reasoning_effort is not None and reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError(
                    f"Invalid reasoning_effort for {model_name}: {reasoning_effort} "
                    "(expected low|medium|high)"
                )
            model_id = baseline.get("model_id")
            if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
                raise ValueError(f"Invalid model_id for {model_name}: must be a non-empty string when provided.")
            specs.append(
                ModelRunSpec(
                    model_name=model_name,
                    baseline_trace=baseline_trace,
                    reasoning_effort=reasoning_effort,
                    model_id=model_id,
                )
            )
            continue

        raise ValueError(
            "Mapping values must be either a string baseline run_id/path, or an object with baseline_trace."
        )
    return specs


def filter_tasks_with_fixes(fixes_root: Path, benchmark: str, failed_tasks: List[str]) -> Tuple[List[str], List[str]]:
    benchmark_root = fixes_root / benchmark
    present: List[str] = []
    missing: List[str] = []
    for task_id in failed_tasks:
        if (benchmark_root / task_id).is_dir():
            present.append(task_id)
        else:
            missing.append(task_id)
    return present, missing


def write_temp_agent_args(
    base_agent_args: Path,
    pricing_model_name: str,
    benchmark: str,
    reasoning_effort: str | None,
    api_model_id: str | None,
) -> Path:
    base = json.loads(base_agent_args.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Base agent args must be an object JSON file: {base_agent_args}")
    base["model_name"] = pricing_model_name
    base["benchmark_name"] = benchmark
    if reasoning_effort:
        base["reasoning_effort"] = reasoning_effort
    if api_model_id:
        base["api_model_id"] = api_model_id
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as handle:
        handle.write(json.dumps(base, indent=2) + "\n")
        return Path(handle.name)


def build_output_names(
    output_dir: Path,
    benchmark: str,
    model_name: str,
    baseline_trace_stem: str,
    *,
    prefix: str,
) -> Tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    baseline_slug = _slugify(baseline_trace_stem)
    model_slug = _slugify(model_name.replace("/", "_"))
    prefix_value = prefix.strip()
    prefix_part = f"{_slugify(prefix_value)}_" if prefix_value else ""
    run_id = f"{prefix_part}{model_slug}_MERGED_{benchmark}_{timestamp}_from_{baseline_slug}_FIXED"
    output_path = output_dir / f"{run_id}_UPLOAD.json"
    return run_id, output_path


def _validate_merged_trace(path: Path, *, expected_run_id: str, expected_task_ids: List[str]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Merged trace is not a JSON object.")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError("Merged trace missing config object.")
    run_id = config.get("run_id")
    if run_id != expected_run_id:
        raise ValueError(f"Merged trace run_id mismatch: expected {expected_run_id}, got {run_id}")
    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError("Merged trace missing results object.")
    if "successful_tasks" not in results or "failed_tasks" not in results:
        raise ValueError("Merged trace results missing successful_tasks/failed_tasks.")
    observed = set(str(x) for x in (results.get("successful_tasks") or []) + (results.get("failed_tasks") or []))
    missing = [task_id for task_id in expected_task_ids if task_id not in observed]
    if missing:
        raise ValueError(f"Merged trace missing {len(missing)} expected task_ids (example: {missing[0]}).")


def run_one_spec(args: argparse.Namespace, spec: ModelRunSpec, live: _LiveState) -> Dict[str, Any]:
    traces_dir = (REPO_ROOT / args.traces_dir).resolve()
    fixes_root = (REPO_ROOT / args.fixes_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_trace_path = _resolve_trace_path(traces_dir, spec.baseline_trace)
    baseline = _load_json(baseline_trace_path)

    benchmark = (baseline.get("config") or {}).get("benchmark_name") or "corebench_hard"
    failed_tasks = (baseline.get("results") or {}).get("failed_tasks") or []
    if not isinstance(failed_tasks, list):
        failed_tasks = []
    failed_tasks = [str(item) for item in failed_tasks if item]

    with_fixes, missing_fixes = filter_tasks_with_fixes(fixes_root, benchmark, failed_tasks)

    # Use the API model id for naming if provided (the mapping key is the pricing name).
    name_model = spec.model_id or spec.model_name
    run_id, merged_output = build_output_names(
        output_dir,
        benchmark,
        name_model,
        baseline_trace_path.stem,
        prefix=args.prefix,
    )
    effective_effort = spec.reasoning_effort or args.reasoning_effort
    pricing_model_name = spec.model_name
    api_model_id = spec.model_id

    summary: Dict[str, Any] = {
        "model_name": spec.model_name,
        "model_id": spec.model_id,
        "pricing_model_name": pricing_model_name,
        "api_model_id": api_model_id,
        "baseline_trace": str(baseline_trace_path),
        "benchmark": benchmark,
        "reasoning_effort": effective_effort,
        "failed_tasks_total": len(failed_tasks),
        "tasks_with_fixes": with_fixes,
        "tasks_missing_fixes": missing_fixes,
        "merged_run_id": run_id,
        "merged_trace": str(merged_output),
        "status": "pending",
    }

    if not with_fixes:
        summary["status"] = "skipped_no_fixes"
        live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=None)
        return summary

    base_agent_args = (REPO_ROOT / args.base_agent_args).resolve()
    temp_agent_args = write_temp_agent_args(
        base_agent_args,
        pricing_model_name,
        benchmark,
        effective_effort,
        api_model_id,
    )

    cmd: List[str] = [
        sys.executable,
        "scripts/run_corebench_fixes.py",
        "--fixes-root",
        str(fixes_root / benchmark),
        "--agent-dir",
        str((REPO_ROOT / args.agent_dir).resolve()),
        "--agent-args",
        str(temp_agent_args),
        "--rubric-model",
        args.rubric_model,
        "--benchmark",
        benchmark,
        "--merged-trace-output",
        str(merged_output),
        "--merged-run-id",
        run_id,
    ]
    if args.prefix:
        cmd += ["--prefix", args.prefix]
    if args.corebench_dataset:
        cmd += ["--corebench-dataset", args.corebench_dataset]
    if args.docker:
        cmd.append("--docker")
    if args.vm:
        cmd.append("--vm")
    if args.wandb_mode:
        cmd += ["--wandb-mode", args.wandb_mode]
    if args.skip_install:
        cmd.append("--skip-install")
    if args.skip_rubrics:
        cmd.append("--skip-rubrics")
    if args.max_parallel_capsules and int(args.max_parallel_capsules) > 1:
        cmd += ["--max-parallel-capsules", str(int(args.max_parallel_capsules))]
    for task_id in with_fixes:
        cmd += ["--task-id", task_id]

    summary["command"] = " ".join(cmd)

    if args.dry_run:
        summary["status"] = "dry_run"
        live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=None)
        return summary

    try:
        import subprocess

        logs_dir = (REPO_ROOT / "log" / "master_rerun").resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.log"
        summary["log_path"] = str(log_path)

        capsule_total = len(with_fixes)
        live.spec_started(model_name=spec.model_name, capsule_total=capsule_total)
        started_capsules: set[str] = set()
        done_capsules: set[str] = set()

        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                handle.write(raw_line)
                line = raw_line.rstrip("\n")
                if args.stream_logs:
                    live.stream_line(model_name=spec.model_name, line=line)

                # Parse structured capsule lifecycle markers from run_corebench_fixes.py.
                if line.startswith("[capsule][start]"):
                    parts = dict(item.split("=", 1) for item in line.split() if "=" in item)
                    capsule_id = parts.get("capsule_id")
                    if capsule_id and capsule_id not in started_capsules:
                        started_capsules.add(capsule_id)
                        live.capsule_started(model_name=spec.model_name, capsule_id=capsule_id)
                elif line.startswith("[capsule][done]"):
                    parts = dict(item.split("=", 1) for item in line.split() if "=" in item)
                    capsule_id = parts.get("capsule_id") or "unknown"
                    if capsule_id not in done_capsules:
                        done_capsules.add(capsule_id)
                        live.capsule_done(model_name=spec.model_name, capsule_id=capsule_id)

            returncode = proc.wait()

        if returncode != 0:
            summary["status"] = "failed"
            summary["returncode"] = returncode
            live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=None)
            return summary

        if not merged_output.exists():
            summary["status"] = "failed_missing_merged_trace"
            live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=None)
            return summary
        _validate_merged_trace(merged_output, expected_run_id=run_id, expected_task_ids=with_fixes)
        summary["status"] = "completed"
        live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=str(merged_output))
        return summary
    except Exception as exc:
        summary["status"] = "failed_exception"
        summary["error"] = str(exc)
        live.spec_finished(model_name=spec.model_name, status=summary["status"], merged_trace=None)
        return summary
    finally:
        try:
            temp_agent_args.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    mapping_path = (REPO_ROOT / args.mapping_file).resolve()
    specs = load_mapping(mapping_path)
    live = _LiveState(total_specs=len(specs))
    results: List[Dict[str, Any]] = []
    if args.dry_run or args.max_parallel <= 1 or len(specs) <= 1:
        for spec in specs:
            results.append(run_one_spec(args, spec, live))
    else:
        max_workers = max(1, int(args.max_parallel))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one_spec, args, spec, live): spec for spec in specs}
            for future in as_completed(futures):
                results.append(future.result())

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mapping_file": str(mapping_path),
        "results": results,
    }
    out_path = (
        REPO_ROOT
        / "traces"
        / f"master_rerun_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary -> {out_path}")
    for item in results:
        if item.get("status") == "completed":
            print(f"[merged] {item['model_name']} -> {item['merged_trace']}")
        elif item.get("status", "").startswith("failed"):
            model = item.get("model_name", "unknown")
            log_path = item.get("log_path")
            extra = f" (log: {log_path})" if log_path else ""
            print(f"[failed] {model}{extra}")


if __name__ == "__main__":
    main()
