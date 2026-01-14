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
from datetime import datetime
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun CoreBench failures with fixes per model/run mapping.")
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to JSON file mapping model_name -> baseline trace run_id (or trace path).",
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
    handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    Path(handle.name).write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return Path(handle.name)


def build_output_names(output_dir: Path, benchmark: str, model_name: str, baseline_trace_stem: str) -> Tuple[str, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    baseline_slug = _slugify(baseline_trace_stem)
    model_slug = _slugify(model_name.replace("/", "_"))
    run_id = f"{model_slug}_MERGED_{benchmark}_{timestamp}_from_{baseline_slug}_FIXED"
    output_path = output_dir / f"{run_id}_UPLOAD.json"
    return run_id, output_path


def run_one_spec(args: argparse.Namespace, spec: ModelRunSpec) -> Dict[str, Any]:
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
    run_id, merged_output = build_output_names(output_dir, benchmark, name_model, baseline_trace_path.stem)
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
    for task_id in with_fixes:
        cmd += ["--task-id", task_id]

    summary["command"] = " ".join(cmd)

    if args.dry_run:
        summary["status"] = "dry_run"
        return summary

    try:
        import subprocess

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        summary["status"] = "completed"
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
    results: List[Dict[str, Any]] = []
    for spec in specs:
        results.append(run_one_spec(args, spec))

    output = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mapping_file": str(mapping_path),
        "results": results,
    }
    out_path = REPO_ROOT / "traces" / f"master_rerun_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary -> {out_path}")
    for item in results:
        if item.get("status") == "completed":
            print(f"[merged] {item['model_name']} -> {item['merged_trace']}")


if __name__ == "__main__":
    main()
