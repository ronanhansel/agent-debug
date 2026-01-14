#!/usr/bin/env python3
"""
Apply CoreBench fix packages, run HAL evaluations, and re-grade the new traces.

Usage:
    python scripts/run_corebench_fixes.py \
        --fixes-root fixes/corebench_hard \
        --agent-dir hal-harness/agents/hal_generalist_agent \
        --agent-args agent_args.azure.json \
        --rubric-model azure_openai:o3-mini
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hal-harness"))

COREBENCH_DATA = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "corebench" / "core_test.json"
COREBENCH_BACKUP = COREBENCH_DATA.with_suffix(".bak")
SMOLAGENTS_SRC = REPO_ROOT / "hal-harness" / "agents" / "open_deep_research" / "src" / "smolagents"

from hal.debugger.fix_loader import (  # type: ignore
    load_fix_package,
    apply_agent_overlay,
    apply_agent_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run HAL eval for CoreBench fixes.")
    parser.add_argument(
        "--fixes-root",
        default=str(REPO_ROOT / "fixes" / "corebench_hard"),
        help="Directory containing per-capsule fix packages (default: fixes/corebench_hard).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"),
        help="Path to the base agent directory.",
    )
    parser.add_argument(
        "--agent-args",
        default=str(REPO_ROOT / "agent_args.azure.json"),
        help="Path to the JSON file with agent arguments (model_name, etc.).",
    )
    parser.add_argument(
        "--rubric-model",
        required=True,
        help="Model identifier for rubric evaluation (e.g., azure_openai:o3-mini or openai:o3-mini).",
    )
    parser.add_argument(
        "--skip-rubrics",
        action="store_true",
        help="Skip running main.py evaluate (per-task and merged trace).",
    )
    parser.add_argument(
        "--benchmark",
        default="corebench_hard",
        help="Benchmark variant to evaluate (default: corebench_hard).",
    )
    parser.add_argument(
        "--merged-trace-output",
        help="Optional output path for the merged trace JSON (must end with .json).",
    )
    parser.add_argument(
        "--merged-run-id",
        help="Optional run_id to store in the merged trace metadata (defaults to auto-generated).",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run hal-eval with the --docker flag.",
    )
    parser.add_argument(
        "--vm",
        action="store_true",
        help="Run hal-eval with the --vm flag (use VM execution, e.g. for GPU tasks).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary agent directories (useful for debugging).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing agent requirements before running hal-eval.",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Only run the specified capsule ID(s). Can be repeated.",
    )
    return parser.parse_args()


def load_agent_args(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Agent args JSON must be an object, got {type(data)}")
    return data


def normalize_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return "null"
    return str(value)


def build_hal_eval_cmd(
    *,
    benchmark: str,
    agent_name: str,
    agent_dir: Path,
    agent_args: Dict[str, object],
    run_id: str,
    docker: bool,
    vm: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", agent_name,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]
    for key, value in agent_args.items():
        cmd += ["-A", f"{key}={normalize_value(value)}"]
    if docker:
        cmd.append("--docker")
    if vm:
        cmd.append("--vm")
    return cmd


def copy_agent_with_fix(base_dir: Path, task_id: str, fix_root: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="corebench_fix_"))
    shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
    if SMOLAGENTS_SRC.exists():
        shutil.copytree(SMOLAGENTS_SRC, temp_dir / "smolagents", dirs_exist_ok=True)
    fix = load_fix_package(task_id, fixes_root=fix_root, benchmark="corebench_hard")
    if not fix:
        return temp_dir
    if fix.agent_overlay:
        apply_agent_overlay(temp_dir, fix.agent_overlay)
    if fix.agent_patch:
        apply_agent_patch(temp_dir, fix.agent_patch)
    return temp_dir


def load_corebench_dataset() -> List[dict]:
    return json.loads(COREBENCH_DATA.read_text(encoding="utf-8"))


def write_filtered_dataset(
    *,
    original_tasks: Dict[str, dict],
    capsule_id: str,
    input_override: Dict[str, object] | None,
) -> None:
    if capsule_id not in original_tasks:
        raise ValueError(f"Capsule {capsule_id} not found in core_test.json")
    task = json.loads(json.dumps(original_tasks[capsule_id]))  # deep copy
    if input_override:
        task.update(input_override)
        if "problem_statement" in input_override:
            original_prompt = task.get("task_prompt", "")
            extra = input_override["problem_statement"]
            task["task_prompt"] = f"{original_prompt}\n\n{extra}" if original_prompt else extra
    COREBENCH_DATA.write_text(json.dumps([task], indent=2), encoding="utf-8")

def install_agent_requirements(agent_dir: Path) -> None:
    requirements = agent_dir / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements}")
    print(f"[setup] Installing requirements from {requirements}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", value)
    return slug or fallback


def merge_trace_files(
    trace_paths: List[Path],
    benchmark: str,
    agent_dir: Path,
    *,
    output_path: Path | None = None,
    merged_run_id: str | None = None,
) -> Path:
    first_trace = json.loads(trace_paths[0].read_text(encoding="utf-8"))
    config: Dict[str, Any] = first_trace.get("config", {})
    agent_name = config.get("agent_name") or agent_dir.name
    model_name = (
        config.get("agent_args", {}).get("model_name")
        if isinstance(config.get("agent_args"), dict)
        else None
    ) or "model"

    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        agent_slug = _slugify(agent_dir.name, "agent")
        model_slug = _slugify(str(model_name), "model")
        run_id = merged_run_id or f"{benchmark}_{agent_slug}{model_slug}_{timestamp}_FIXED"
        output_path = REPO_ROOT / "traces" / f"{run_id}_UPLOAD.json"
    else:
        if output_path.suffix != ".json":
            raise ValueError(f"--merged-trace-output must be a .json file path, got {output_path}")
        run_id = merged_run_id or output_path.stem
        if run_id.endswith("_UPLOAD"):
            run_id = run_id[: -len("_UPLOAD")]

    merge_cmd: List[str] = [
        sys.executable,
        "scripts/merge_traces.py",
    ]
    for path in trace_paths:
        merge_cmd += ["--input", str(path)]
    merge_cmd += [
        "--output",
        str(output_path),
        "--run-id",
        run_id,
        "--agent-name",
        agent_name,
        "--date",
        datetime.utcnow().strftime("%Y-%m-%d"),
        "--force",
    ]
    subprocess.run(merge_cmd, check=True, cwd=REPO_ROOT)
    return output_path


def evaluate_trace(trace_path: Path, rubric_model: str, output_dir: Path) -> None:
    eval_cmd = [
        "python",
        "main.py",
        "evaluate",
        "--trace-file",
        str(trace_path),
        "--rubrics-dir",
        str(REPO_ROOT / "rubrics"),
        "--output-dir",
        str(output_dir),
        "--rubric-model",
        rubric_model,
        "--yes",
    ]
    subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)

def main() -> None:
    args = parse_args()
    fixes_root = Path(args.fixes_root)
    agent_dir = Path(args.agent_dir)
    agent_args_path = Path(args.agent_args)
    rubric_output_dir = REPO_ROOT / "rubrics_output"
    rubric_output_dir.mkdir(parents=True, exist_ok=True)

    agent_args = load_agent_args(agent_args_path)
    agent_args["benchmark_name"] = args.benchmark

    if not args.skip_install:
        install_agent_requirements(agent_dir)

    tasks_data = load_corebench_dataset()
    capsule_map = {item["capsule_id"]: item for item in tasks_data}

    fix_dirs = sorted([p for p in fixes_root.iterdir() if p.is_dir()])
    if args.task_ids:
        requested = set(args.task_ids)
        fix_dirs = [p for p in fix_dirs if p.name in requested]
        missing = requested - {p.name for p in fix_dirs}
        if missing:
            print(f"[WARN] Requested task(s) not found under {fixes_root}: {', '.join(sorted(missing))}")
    if not fix_dirs:
        print(f"No fix directories found in {fixes_root}")
        return

    shutil.copyfile(COREBENCH_DATA, COREBENCH_BACKUP)
    print(f"Backed up core_test.json -> {COREBENCH_BACKUP}")

    try:
        fixes_base = fixes_root.parent
        generated_traces: List[Path] = []
        for fix_dir in fix_dirs:
            capsule_id = fix_dir.name
            print(f"\n=== Processing {capsule_id} ===")

            try:
                fix = load_fix_package(capsule_id, fixes_root=fixes_base, benchmark=args.benchmark)
            except Exception as exc:
                print(f"[WARN] Failed to load fix for {capsule_id}: {exc}. Skipping.")
                continue

            input_override = fix.input_override if fix else None
            write_filtered_dataset(
                original_tasks=capsule_map,
                capsule_id=capsule_id,
                input_override=input_override,
            )

            temp_agent_dir = copy_agent_with_fix(agent_dir, capsule_id, fixes_base)
            try:
                run_id = f"{capsule_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                cmd = build_hal_eval_cmd(
                    benchmark=args.benchmark,
                    agent_name=f"hal_generalist_agent ({capsule_id})",
                    agent_dir=temp_agent_dir,
                    agent_args=agent_args,
                    run_id=run_id,
                    docker=args.docker,
                    vm=args.vm,
                )
                print(f"[hal-eval] {' '.join(cmd)}")
                hal_env = os.environ.copy()
                extra_path = str(REPO_ROOT / "hal-harness")
                hal_env["PYTHONPATH"] = (
                    f"{extra_path}{os.pathsep}{hal_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
                )
                hal_env.setdefault("WANDB_MODE", "offline")
                hal_env.setdefault("WANDB_SILENT", "true")
                subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=hal_env)

                results_dir = REPO_ROOT / "results" / args.benchmark / run_id
                trace_path = results_dir / f"{run_id}_UPLOAD.json"
                if not trace_path.exists():
                    raise FileNotFoundError(f"Expected trace {trace_path} not found.")

                final_trace = REPO_ROOT / "traces" / trace_path.name
                shutil.copyfile(trace_path, final_trace)
                print(f"[trace] Copied {trace_path} -> {final_trace}")
                generated_traces.append(final_trace)

                if not args.skip_rubrics:
                    eval_cmd = [
                        "python",
                        "main.py",
                        "evaluate",
                        "--trace-file", str(final_trace),
                        "--rubrics-dir", str(REPO_ROOT / "rubrics"),
                        "--output-dir", str(rubric_output_dir),
                        "--rubric-model", args.rubric_model,
                        "--yes",
                    ]
                    print(f"[rubric] {' '.join(eval_cmd)}")
                    subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)
            finally:
                if not args.keep_temp:
                    shutil.rmtree(temp_agent_dir, ignore_errors=True)

        if generated_traces:
            merged_trace = merge_trace_files(
                generated_traces,
                benchmark=args.benchmark,
                agent_dir=agent_dir,
                output_path=Path(args.merged_trace_output).resolve() if args.merged_trace_output else None,
                merged_run_id=args.merged_run_id,
            )
            print(f"[merge] Saved merged trace -> {merged_trace}")
            if not args.skip_rubrics:
                evaluate_trace(merged_trace, args.rubric_model, rubric_output_dir)
                print(f"[merge] Completed rubric evaluation for {merged_trace}")

    finally:
        shutil.move(COREBENCH_BACKUP, COREBENCH_DATA)
        print("Restored original core_test.json")


if __name__ == "__main__":
    main()
