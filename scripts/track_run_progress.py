#!/usr/bin/env python3
"""
Track aggregate progress for each benchmark run.

Shows completed tasks (from RAW_SUBMISSIONS.jsonl) and evaluated tasks
(from *_eval.jsonl or *_UPLOAD.json when available). Use --per-agent
to see individual agent rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
TOKEN_RE = re.compile(r"Input tokens:\s*([\d,]+).*Output tokens:\s*([\d,]+)")


@dataclass
class VerboseMetrics:
    start: Optional[datetime]
    end: Optional[datetime]
    tokens: int
    token_start: Optional[datetime]
    token_end: Optional[datetime]


@dataclass
class AggregateMetrics:
    benchmark: str
    runs: int
    tasks_done: Optional[int]
    tasks_total: Optional[int]
    tokens_total: int


class RateTracker:
    def __init__(self, window_seconds: int) -> None:
        self.window_seconds = max(1, window_seconds)
        self.history: Dict[str, List[Tuple[float, Optional[int], int]]] = {}

    def update(self, metrics: List[AggregateMetrics]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        now = time.time()
        rates: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        cutoff = now - self.window_seconds
        for metric in metrics:
            hist = self.history.setdefault(metric.benchmark, [])
            hist.append((now, metric.tasks_done, metric.tokens_total))
            while hist and hist[0][0] < cutoff:
                hist.pop(0)

            tasks_rate = self._compute_rate(hist, value_index=1, cutoff=cutoff)
            tokens_rate = self._compute_rate(hist, value_index=2, cutoff=cutoff)
            rates[metric.benchmark] = (tasks_rate, tokens_rate)
        return rates

    @staticmethod
    def _compute_rate(
        hist: List[Tuple[float, Optional[int], int]],
        value_index: int,
        cutoff: float,
    ) -> Optional[float]:
        if not hist:
            return None
        latest = hist[-1]
        latest_value = latest[value_index]
        if latest_value is None:
            return None
        start_entry = None
        for entry in hist:
            if entry[0] >= cutoff and entry[value_index] is not None:
                start_entry = entry
                break
        if start_entry is None:
            return None
        delta = latest_value - start_entry[value_index]  # type: ignore[operator]
        if delta < 0:
            return None
        dt = latest[0] - start_entry[0]
        if dt <= 0:
            return None
        return delta / (dt / 60.0)


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


def build_table(rows: Iterable[Tuple[str, ...]]) -> List[str]:
    rows = list(rows)
    if not rows:
        return []
    col_count = max(len(row) for row in rows)
    col_widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    lines = []
    for row in rows:
        padded = list(row) + [""] * (col_count - len(row))
        lines.append("  ".join(padded[i].ljust(col_widths[i]) for i in range(col_count)))
    return lines


def print_table(rows: Iterable[Tuple[str, ...]]) -> None:
    for line in build_table(rows):
        print(line)


def parse_timestamp(line: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def parse_verbose_metrics(path: Path) -> VerboseMetrics:
    if not path.exists():
        return VerboseMetrics(None, None, 0, None, None)

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    token_start: Optional[datetime] = None
    token_end: Optional[datetime] = None
    total_tokens = 0
    last_total: Optional[int] = None
    last_ts: Optional[datetime] = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                ts = parse_timestamp(line)
                if ts:
                    last_ts = ts
                    if start is None or ts < start:
                        start = ts
                    if end is None or ts > end:
                        end = ts

                token_match = TOKEN_RE.search(line)
                if token_match:
                    input_tokens = int(token_match.group(1).replace(",", ""))
                    output_tokens = int(token_match.group(2).replace(",", ""))
                    current_total = input_tokens + output_tokens
                    if last_total is None:
                        delta = current_total
                    else:
                        delta = current_total - last_total
                        if delta < 0:
                            delta = current_total
                    if delta > 0:
                        total_tokens += delta
                    last_total = current_total

                    token_ts = ts or last_ts
                    if token_ts:
                        if token_start is None or token_ts < token_start:
                            token_start = token_ts
                        if token_end is None or token_ts > token_end:
                            token_end = token_ts
    except Exception:
        return VerboseMetrics(None, None, 0, None, None)

    return VerboseMetrics(start, end, total_tokens, token_start, token_end)


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:,.1f}"


def format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "?"


def format_window(window_seconds: int) -> str:
    if window_seconds % 60 == 0:
        return f"{window_seconds // 60}m"
    return f"{window_seconds}s"




def token_rate_label(unit: str) -> str:
    return "tokens/sec" if unit == "sec" else "tokens/min"


def format_token_rate(value_per_min: Optional[float], unit: str) -> str:
    if value_per_min is None:
        return "?"
    if unit == "sec":
        return f"{value_per_min / 60.0:,.1f}"
    return f"{value_per_min:,.1f}"


def choose_tasks_done(
    completed_total: Optional[int],
    eval_count: Optional[int],
    completed_ok: Optional[int],
    total_tasks: Optional[int],
) -> Optional[int]:
    for candidate in (completed_total, eval_count, completed_ok, total_tasks):
        if candidate is not None:
            return candidate
    return None


def get_per_agent_rows(run_dir: Path, run_root: Path, repo_root: Path) -> List[Tuple[str, ...]]:
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

    return rows


def collect_aggregate_metrics(run_dir: Path, run_root: Path, repo_root: Path) -> List[AggregateMetrics]:
    metrics: List[AggregateMetrics] = []

    for benchmark in BENCHMARKS:
        log_file = run_dir / f"{benchmark}.log"
        text = read_text(log_file)
        if not text:
            continue

        total_tasks = parse_total_tasks(text)
        run_root_override = parse_run_root(text)
        run_root_for_bench = Path(run_root_override) if run_root_override else run_root
        run_ids = list(parse_run_ids(text))
        if not run_ids:
            continue

        tasks_done_sum = 0
        tasks_done_found = False
        tasks_expected_sum = 0
        tasks_expected_known = True
        tokens_total = 0

        for run_id in run_ids:
            run_path = resolve_run_dir(run_root_for_bench, repo_root, benchmark, run_id)
            if not run_path:
                continue

            raw_path = run_path / f"{run_id}_RAW_SUBMISSIONS.jsonl"
            completed_ok, completed_err, completed_total = count_raw_submissions(raw_path)
            if benchmark == "scienceagentbench":
                eval_count = count_eval_scienceagentbench(run_path, run_id)
            elif benchmark in ("scicode", "corebench"):
                eval_count = count_eval_from_upload(run_path, run_id)
            else:
                eval_count = completed_ok

            tasks_done = choose_tasks_done(completed_total, eval_count, completed_ok, total_tasks)
            if tasks_done is not None:
                tasks_done_sum += tasks_done
                tasks_done_found = True

            expected = total_tasks if total_tasks is not None else completed_total or eval_count or completed_ok
            if expected is None:
                tasks_expected_known = False
            else:
                tasks_expected_sum += expected

            verbose_log = run_path / f"{run_id}_verbose.log"
            metrics_item = parse_verbose_metrics(verbose_log)
            tokens_total += metrics_item.tokens

        metrics.append(
            AggregateMetrics(
                benchmark=benchmark,
                runs=len(run_ids),
                tasks_done=tasks_done_sum if tasks_done_found else None,
                tasks_total=tasks_expected_sum if tasks_expected_known else None,
                tokens_total=tokens_total,
            )
        )

    return metrics


def build_aggregate_rows(
    metrics: List[AggregateMetrics],
    rates: Dict[str, Tuple[Optional[float], Optional[float]]],
    token_rate_unit: str,
) -> List[Tuple[str, ...]]:
    header = (
        "benchmark",
        "runs",
        "tasks_done",
        "tasks_total",
        "tasks/min",
        token_rate_label(token_rate_unit),
    )
    rows: List[Tuple[str, ...]] = [header]
    for item in metrics:
        tasks_rate, tokens_rate = rates.get(item.benchmark, (None, None))
        rows.append((
            item.benchmark,
            str(item.runs),
            format_int(item.tasks_done),
            format_int(item.tasks_total),
            format_rate(tasks_rate),
            format_token_rate(tokens_rate, token_rate_unit),
        ))
    return rows


def get_aggregate_rows(run_dir: Path, run_root: Path, repo_root: Path) -> List[Tuple[str, ...]]:
    metrics = collect_aggregate_metrics(run_dir, run_root, repo_root)
    return build_aggregate_rows(metrics, {}, "min")


def render_per_agent(run_dir: Path, run_root: Path, repo_root: Path) -> None:
    print_table(get_per_agent_rows(run_dir, run_root, repo_root))


def render_aggregate(run_dir: Path, run_root: Path, repo_root: Path) -> None:
    print_table(get_aggregate_rows(run_dir, run_root, repo_root))


def build_output_lines(
    run_dir: Path,
    run_root: Path,
    repo_root: Path,
    per_agent: bool,
    rate_tracker: Optional[RateTracker],
    window_seconds: int,
    token_rate_unit: str,
) -> List[str]:
    rows: List[Tuple[str, ...]]
    if per_agent:
        rows = get_per_agent_rows(run_dir, run_root, repo_root)
    else:
        metrics = collect_aggregate_metrics(run_dir, run_root, repo_root)
        rates = rate_tracker.update(metrics) if rate_tracker else {}
        rows = build_aggregate_rows(metrics, rates, token_rate_unit)
    header = f"Run: {run_dir}"
    if not per_agent:
        header = f"{header} | window={format_window(window_seconds)}"
    lines = [header]
    lines.extend(build_table(rows))
    return lines


def run_tui(
    run_dir: Path,
    run_root: Path,
    repo_root: Path,
    per_agent: bool,
    interval: int,
    window_seconds: int,
    token_rate_unit: str,
) -> None:
    import curses

    def _loop(screen: "curses._CursesWindow") -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(max(100, interval * 1000))
        rate_tracker = RateTracker(window_seconds)
        while True:
            lines = build_output_lines(
                run_dir,
                run_root,
                repo_root,
                per_agent,
                rate_tracker,
                window_seconds,
                token_rate_unit,
            )
            height, width = screen.getmaxyx()
            screen.erase()
            max_lines = max(0, height - 1)
            for idx, line in enumerate(lines[:max_lines]):
                screen.addnstr(idx, 0, line, max(0, width - 1))
            hint = "Press q to quit"
            if height > 0:
                screen.addnstr(height - 1, 0, hint, max(0, width - 1))
            screen.refresh()
            ch = screen.getch()
            if ch in (ord("q"), ord("Q")):
                break

    curses.wrapper(_loop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track aggregate progress for each benchmark run.")
    parser.add_argument("--run-dir", help="Path to logs/benchmark_run_* directory.")
    parser.add_argument("--per-agent", action="store_true", help="Show per-agent rows instead of aggregate benchmark totals.")
    parser.add_argument("--watch", action="store_true", help="Refresh every few seconds.")
    parser.add_argument("--tui", action="store_true", help="Force curses TUI for smoother updates.")
    parser.add_argument("--no-tui", action="store_true", help="Disable curses TUI even if available.")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds.")
    parser.add_argument("--window-seconds", type=int, default=300, help="Rate window in seconds (default: 300).")
    parser.add_argument(
        "--token-rate-unit",
        choices=("min", "sec"),
        default="min",
        help="Display token rate per minute or per second (default: min).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    run_root = detect_run_root(script_dir)
    repo_root = script_dir

    logs_dir = run_root / "logs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(logs_dir)
    if not run_dir or not run_dir.exists():
        print("No benchmark_run_* directory found.")
        return

    use_tui = False
    if args.watch and sys.stdout.isatty() and not args.no_tui:
        use_tui = True
    if args.tui:
        use_tui = True

    if use_tui:
        try:
            run_tui(
                run_dir,
                run_root,
                repo_root,
                args.per_agent,
                args.interval,
                args.window_seconds,
                args.token_rate_unit,
            )
            return
        except Exception:
            use_tui = False

    rate_tracker = RateTracker(args.window_seconds) if args.watch else None
    while True:
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()
        lines = build_output_lines(
            run_dir,
            run_root,
            repo_root,
            args.per_agent,
            rate_tracker,
            args.window_seconds,
            args.token_rate_unit,
        )
        print("\n".join(lines))
        if not args.watch:
            break
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
