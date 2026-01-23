#!/usr/bin/env python3
"""
Track per-agent progress for each benchmark run.

Shows completed tasks (from RAW_SUBMISSIONS.jsonl) and evaluated tasks
(from *_eval.jsonl or *_UPLOAD.json when available).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}


def detect_run_root(script_dir: Path) -> Path:
    data_path = os.environ.get("DATA_PATH")
    if data_path and Path(data_path).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or "user"
        return Path(data_path) / "hal_runs" / namespace / script_dir.name
    data_root = os.environ.get("HAL_DATA_ROOT")
    if data_root and Path(data_root).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or "user"
        return Path(data_root) / "hal_runs" / namespace / script_dir.name
    return script_dir


def latest_run_dir(logs_dir: Path) -> Optional[Path]:
    runs = sorted(logs_dir.glob("benchmark_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def parse_total_tasks(text: str) -> Optional[int]:
    totals = [int(m.group(1)) for m in re.finditer(r"\((\d+) tasks\)", text)]
    return max(totals) if totals else None


def parse_run_root(text: str) -> Optional[str]:
    match = re.search(r"\[Storage\] Run root: (.+)", text)
    if match:
        return match.group(1).strip()
    return None


def parse_run_ids(text: str) -> Iterable[str]:
    return sorted(set(re.findall(r"Run ID: (\S+)", text)))


def load_prefix(run_dir: Path) -> Optional[str]:
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text()).get("prefix")
    except Exception:
        return None


def derive_config_key(run_id: str, prefix: Optional[str]) -> str:
    if prefix and run_id.startswith(prefix):
        run_id = run_id[len(prefix):]
    return re.sub(r"_[0-9]{8}_[0-9]{6}$", "", run_id)


def resolve_run_dir(run_root: Path, repo_root: Path, benchmark: str, run_id: str) -> Optional[Path]:
    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    candidates = [
        run_root / "results" / hal_name / run_id,
        run_root / "results" / benchmark / run_id,
        repo_root / "results" / hal_name / run_id,
        repo_root / "results" / benchmark / run_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def count_raw_submissions(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not path.exists():
        return None, None, None
    entries: Dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj:
                    task_id = next(iter(obj.keys()))
                    entries[task_id] = obj[task_id]
    except Exception:
        return None, None, None
    errors = sum(1 for v in entries.values() if isinstance(v, str) and v.startswith("ERROR"))
    completed = sum(1 for v in entries.values() if not (isinstance(v, str) and v.startswith("ERROR")))
    return completed, errors, len(entries)


def count_eval_scienceagentbench(run_dir: Path, run_id: str) -> Optional[int]:
    eval_path = run_dir / f"{run_id}_eval.jsonl"
    if not eval_path.exists():
        return None
    count = 0
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return None
    return count


def count_eval_from_upload(run_dir: Path, run_id: str) -> Optional[int]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    try:
        data = json.loads(upload_path.read_text())
    except Exception:
        return None
    results = data.get("results", {})
    successful = results.get("successful_tasks")
    failed = results.get("failed_tasks")
    if isinstance(successful, list) and isinstance(failed, list):
        return len(successful) + len(failed)
    return None


def status_label(completed_ok: Optional[int], completed_err: Optional[int], total_tasks: Optional[int], eval_count: Optional[int]) -> str:
    if completed_ok is None and eval_count is None:
        return "UNKNOWN"
    if completed_err and completed_err > 0:
        return "ERRORS"
    if total_tasks is not None and completed_ok is not None and completed_ok < total_tasks:
        return "INCOMPLETE"
    if total_tasks is not None and eval_count is not None and eval_count < total_tasks:
        return "EVAL_INCOMPLETE"
    return "DONE"


def print_table(rows: Iterable[Tuple[str, ...]]) -> None:
    col_widths = [0] * 7
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    for row in rows:
        line = "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        print(line)


def render(run_dir: Path, run_root: Path, repo_root: Path) -> None:
    prefix = load_prefix(run_dir)
    header = (
        "benchmark",
        "agent",
        "run_id",
        "completed",
        "evaluated",
        "total",
        "status",
    )
    rows = [header]

    for benchmark in BENCHMARKS:
        log_file = run_dir / f"{benchmark}.log"
        text = read_text(log_file)
        if not text:
            continue
        total_tasks = parse_total_tasks(text)
        run_root_override = parse_run_root(text)
        run_root_for_bench = Path(run_root_override) if run_root_override else run_root
        run_ids = parse_run_ids(text)
        for run_id in run_ids:
            config_key = derive_config_key(run_id, prefix)
            run_path = resolve_run_dir(run_root_for_bench, repo_root, benchmark, run_id)
            raw_path = run_path / f"{run_id}_RAW_SUBMISSIONS.jsonl" if run_path else Path("")
            completed_ok, completed_err, completed_total = count_raw_submissions(raw_path)
            if benchmark == "scienceagentbench" and run_path:
                eval_count = count_eval_scienceagentbench(run_path, run_id)
            elif benchmark in ("scicode", "corebench") and run_path:
                eval_count = count_eval_from_upload(run_path, run_id)
            else:
                eval_count = completed_ok

            completed_str = "?"
            if completed_ok is not None:
                err_part = f", err={completed_err}" if completed_err is not None else ""
                entries_part = f", entries={completed_total}" if completed_total is not None else ""
                total_part = f"/{total_tasks}" if total_tasks is not None else ""
                completed_str = f"{completed_ok}{err_part}{entries_part}{total_part}"

            evaluated_str = "?"
            if eval_count is not None:
                total_part = f"/{total_tasks}" if total_tasks is not None else ""
                evaluated_str = f"{eval_count}{total_part}"

            total_str = str(total_tasks) if total_tasks is not None else "?"
            status = status_label(completed_ok, completed_err, total_tasks, eval_count)
            short_id = run_id if len(run_id) <= 32 else f"{run_id[:14]}..{run_id[-14:]}"
            rows.append((
                benchmark,
                config_key,
                short_id,
                completed_str,
                evaluated_str,
                total_str,
                status,
            ))

    print_table(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track per-agent progress for each benchmark run.")
    parser.add_argument("--run-dir", help="Path to logs/benchmark_run_* directory.")
    parser.add_argument("--watch", action="store_true", help="Refresh every few seconds.")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    run_root = detect_run_root(script_dir)
    repo_root = script_dir

    logs_dir = run_root / "logs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(logs_dir)
    if not run_dir or not run_dir.exists():
        print("No benchmark_run_* directory found.")
        return

    while True:
        os.system("clear")
        print(f"Run: {run_dir}")
        render(run_dir, run_root, repo_root)
        if not args.watch:
            break
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
