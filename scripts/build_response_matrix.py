#!/usr/bin/env python3
"""
Build a binary response matrix for a given prefix.

Rows: benchmark.config_key (agent row).
Columns: benchmark.task_id

Partial scoring:
- Only score tasks with non-error raw submissions.
- Missing/ERROR tasks are left blank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS_PATH = REPO_ROOT / "hal-harness"
if str(HAL_HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HAL_HARNESS_PATH))

BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}
TASK_ID_FIELD = {
    "scicode": "problem_id",
    "scienceagentbench": "instance_id",
    "corebench": "capsule_id",
}

AUTH_ERROR_SNIPPETS = (
    "authenticationerror",
    "invalid_api_key",
    "incorrect api key",
    "unauthorized",
    "error code: 401",
)


def _push_env(key: str, value: Optional[str]) -> Optional[str]:
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return old


def _extract_python_code(history_payload: Dict[str, object]) -> Optional[str]:
    try:
        history = history_payload.get("history", [])
    except AttributeError:
        return None
    if not history:
        return None
    last = history[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return None
    content = last.get("content", "")
    if not isinstance(content, str):
        return None
    match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


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


def find_log_dir_for_prefix(log_base: Path, prefix: str) -> Optional[Path]:
    for run_dir in sorted(log_base.glob("benchmark_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        cfg = run_dir / "config.json"
        if not cfg.exists():
            continue
        try:
            saved_prefix = json.loads(cfg.read_text()).get("prefix")
        except Exception:
            continue
        if saved_prefix == prefix:
            return run_dir
    return None


def parse_run_ids(text: str) -> List[str]:
    return sorted(set(re.findall(r"Run ID: (\S+)", text)))


def parse_dataset_path(text: str, benchmark: str) -> Optional[Path]:
    # Example: Custom dataset: SCICODE_DATASET_PATH=/path/file.json (65 tasks)
    env_var = {
        "scicode": "SCICODE_DATASET_PATH",
        "scienceagentbench": "SCIENCEAGENTBENCH_DATASET_PATH",
        "corebench": "HAL_COREBENCH_DATASET_PATH",
        "colbench": "COLBENCH_BACKEND_DATASET_PATH",
    }.get(benchmark)
    if not env_var:
        return None
    pattern = re.compile(rf"{env_var}=([^\s]+)")
    matches = pattern.findall(text)
    if not matches:
        return None
    return Path(matches[-1])


def derive_config_key(run_id: str, benchmark: str, prefix: str) -> str:
    if run_id.startswith(f"{benchmark}_"):
        run_id = run_id[len(benchmark) + 1 :]
    if run_id.startswith(prefix):
        run_id = run_id[len(prefix) :]
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


def is_error_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("ERROR"):
        return True
    lowered = value.lower()
    return any(snippet in lowered for snippet in AUTH_ERROR_SNIPPETS)


def load_raw_ok_tasks(raw_path: Path) -> Dict[str, object]:
    ok: Dict[str, object] = {}
    if not raw_path.exists():
        return ok
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            task_id = next(iter(obj.keys()))
            value = obj[task_id]
            if not is_error_value(value):
                ok[str(task_id)] = value
    return ok


def load_task_ids(benchmark: str, dataset_path: Path) -> List[str]:
    if benchmark == "colbench":
        tasks = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(line)
        return [str(i) for i in range(len(tasks))]

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    field = TASK_ID_FIELD[benchmark]
    return [str(task[field]) for task in data]


def load_scicode_success(run_dir: Path, run_id: str) -> Optional[set]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    successful = data.get("results", {}).get("successful_tasks")
    if isinstance(successful, list):
        return {str(t) for t in successful}
    return None


def reeval_scicode(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.scicode import SciCodeBenchmark

    env_old = _push_env("SCICODE_DATASET_PATH", str(dataset_path))
    try:
        bench = SciCodeBenchmark(agent_dir=".", config={}, benchmark_name="scicode")
        eval_results = bench.evaluate_output(ok_tasks, run_id="reeval_scicode")
    finally:
        _push_env("SCICODE_DATASET_PATH", env_old)

    details = eval_results.get("details", {})
    results: Dict[str, int] = {}
    for task_id in ok_tasks.keys():
        task = bench.benchmark.get(task_id)
        if not task:
            continue
        total = len(task.get("sub_steps", []))
        passed = len(details.get(task_id, []))
        results[task_id] = 1 if total > 0 and passed == total else 0
    return results


def reeval_corebench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.corebench import CoreBenchHard

    env_old = _push_env("HAL_COREBENCH_DATASET_PATH", str(dataset_path))
    try:
        bench = CoreBenchHard(agent_dir=".", config={})
        eval_results = bench.evaluate_output(ok_tasks, run_id="reeval_corebench")
    finally:
        _push_env("HAL_COREBENCH_DATASET_PATH", env_old)

    results: Dict[str, int] = {}
    for task_id, result in eval_results.items():
        written_correct = result.get("correct_written_answers", 0)
        vision_correct = result.get("correct_vision_answers", 0)
        written_total = result.get("total_written_questions", 0)
        vision_total = result.get("total_vision_questions", 0)
        if written_total == 0 and vision_total == 0:
            results[task_id] = 0
            continue
        results[task_id] = 1 if (written_correct == written_total and vision_correct == vision_total) else 0
    return results


def reeval_colbench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.sweet_rl.utils import code_evaluate

    sorted_ids = sorted(ok_tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)
    trajectories = [ok_tasks[task_id] for task_id in sorted_ids]
    correctness = code_evaluate(trajectories)
    results: Dict[str, int] = {}
    for idx, task_id in enumerate(sorted_ids):
        score = correctness[idx] if idx < len(correctness) else 0
        results[task_id] = 1 if score >= 0.999 else 0
    return results


def reeval_scienceagentbench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    # Mirror ScienceAgentBench imports/path setup
    repo_root = Path(__file__).resolve().parents[1]
    hal_harness = repo_root / "hal-harness"
    submodule_path = hal_harness / "hal" / "benchmarks" / "scienceagentbench" / "ScienceAgentBench_modified"
    if str(submodule_path) not in sys.path:
        sys.path.insert(0, str(submodule_path))

    from evaluation.harness import run_evaluation as sab_eval

    # Load dataset mapping
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    id_to_gold = {str(row["instance_id"]): row["gold_program_name"] for row in data}

    tmp_dir = Path(os.environ.get("HAL_TMP_DIR", "/tmp")) / f"sab_reeval_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pred_programs = tmp_dir / "pred_programs"
    pred_programs.mkdir(parents=True, exist_ok=True)

    instance_ids: List[str] = []
    for task_id, payload in ok_tasks.items():
        code = _extract_python_code(payload) if isinstance(payload, dict) else None
        if not code:
            continue
        gold_name = id_to_gold.get(str(task_id))
        if not gold_name:
            continue
        out_path = pred_programs / f"pred_{gold_name}"
        out_path.write_text(code, encoding="utf-8")
        instance_ids.append(str(task_id))

    if not instance_ids:
        return {}

    log_fname = tmp_dir / "sab_eval.jsonl"

    # Point eval logs into tmp
    sab_eval.RUN_EVALUATION_LOG_DIR = tmp_dir / "logs"

    env_old = _push_env("SCIENCEAGENTBENCH_DATASET_PATH", str(dataset_path))
    try:
        sab_eval.main(
            benchmark_path=str(submodule_path / "benchmark"),
            pred_program_path=str(pred_programs),
            log_fname=str(log_fname),
            dataset_name="osunlp/ScienceAgentBench",
            split="validation",
            instance_ids=instance_ids,
            max_workers=max(1, (os.cpu_count() or 2) // 2),
            force_rebuild=True,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id=f"reeval_{os.getpid()}",
            timeout=1800,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY", ""),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        )
    finally:
        _push_env("SCIENCEAGENTBENCH_DATASET_PATH", env_old)

    results: Dict[str, int] = {}
    if log_fname.exists():
        lines = log_fname.read_text(encoding="utf-8").splitlines()
        ordered_ids = sorted(instance_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
        for idx, task_id in enumerate(ordered_ids):
            if idx >= len(lines):
                continue
            line = lines[idx].strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                success_rate = payload.get("success_rate")
                if isinstance(success_rate, (int, float)):
                    results[task_id] = 1 if success_rate >= 1.0 else 0
    return results


def load_corebench_success(run_dir: Path, run_id: str) -> Optional[set]:
    return load_scicode_success(run_dir, run_id)


def load_colbench_correctness(run_dir: Path, run_id: str) -> Optional[List[float]]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    correctness = data.get("raw_eval_results")
    if isinstance(correctness, list):
        return [float(v) for v in correctness]
    return None


def load_scienceagentbench_eval(run_dir: Path, run_id: str, task_ids: List[str]) -> Dict[str, float]:
    eval_path = run_dir / f"{run_id}_eval.jsonl"
    results: Dict[str, float] = {}
    if not eval_path.exists():
        return results
    ordered_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    for idx, task_id in enumerate(ordered_ids):
        if idx >= len(lines):
            continue
        line = lines[idx].strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            success_rate = payload.get("success_rate")
            if isinstance(success_rate, (int, float)):
                results[task_id] = float(success_rate)
    return results


def get_gspread_client(use_oauth: bool = False):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return None

    gc = None
    sa_path = Path.home() / ".config" / "gspread" / "service_account.json"
    if sa_path.exists():
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
        gc = gspread.authorize(creds)
    elif use_oauth:
        gc = gspread.oauth()
    return gc


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build binary response matrix for a prefix.")
    parser.add_argument("--prefix", required=True, help="Prefix like moon6_")
    parser.add_argument("--log-dir", help="Path to benchmark_run_* directory")
    parser.add_argument("--run-root", help="Override run root path")
    parser.add_argument("--output", help="CSV output path")
    parser.add_argument("--upload", action="store_true", help="Upload to Google Sheets")
    parser.add_argument("--sheet-name", default="Response Matrix", help="Worksheet name")
    parser.add_argument("--oauth", action="store_true", help="Use OAuth for Google Sheets")
    parser.add_argument("--reeval", action="store_true", help="Re-evaluate tasks from raw submissions")
    parser.add_argument("--skip-benchmark", action="append", default=[], help="Skip benchmark(s) by name")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    repo_root = script_dir
    run_root = Path(args.run_root) if args.run_root else detect_run_root(script_dir)

    logs_base = run_root / "logs"
    log_dir = Path(args.log_dir) if args.log_dir else find_log_dir_for_prefix(logs_base, args.prefix)
    if not log_dir or not log_dir.exists():
        print(f"No log dir found for prefix {args.prefix}")
        sys.exit(1)

    # Build task columns
    benchmark_task_ids: Dict[str, List[str]] = {}
    skip_set = set(args.skip_benchmark)
    for benchmark in BENCHMARKS:
        if benchmark in skip_set:
            continue
        log_file = log_dir / f"{benchmark}.log"
        if not log_file.exists():
            continue
        text = log_file.read_text(errors="ignore")
        dataset_path = parse_dataset_path(text, benchmark)
        if not dataset_path or not dataset_path.exists():
            print(f"Missing dataset path for {benchmark}; cannot build columns")
            sys.exit(1)
        benchmark_task_ids[benchmark] = load_task_ids(benchmark, dataset_path)

    columns: List[str] = []
    for benchmark in BENCHMARKS:
        for task_id in benchmark_task_ids.get(benchmark, []):
            columns.append(f"{benchmark}.{task_id}")

    # Build rows
    rows: List[List[str]] = []
    row_labels: List[str] = []

    for benchmark in BENCHMARKS:
        if benchmark in skip_set:
            continue
        log_file = log_dir / f"{benchmark}.log"
        if not log_file.exists():
            continue
        text = log_file.read_text(errors="ignore")
        run_ids = [rid for rid in parse_run_ids(text) if args.prefix in rid]
        for run_id in run_ids:
            config_key = derive_config_key(run_id, benchmark, args.prefix)
            row_label = f"{benchmark}.{config_key}"

            run_dir = resolve_run_dir(run_root, repo_root, benchmark, run_id)
            if not run_dir:
                continue
            raw_path = run_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
            ok_tasks = load_raw_ok_tasks(raw_path)

            success_map: Dict[str, int] = {}
            if args.reeval:
                dataset_path = parse_dataset_path(text, benchmark)
                if not dataset_path or not dataset_path.exists():
                    print(f"Missing dataset path for {benchmark}; cannot re-evaluate")
                    sys.exit(1)
                if benchmark == "scicode":
                    success_map = reeval_scicode(dataset_path, ok_tasks)
                elif benchmark == "corebench":
                    success_map = reeval_corebench(dataset_path, ok_tasks)
                elif benchmark == "colbench":
                    success_map = reeval_colbench(dataset_path, ok_tasks)
                elif benchmark == "scienceagentbench":
                    success_map = reeval_scienceagentbench(dataset_path, ok_tasks)
            else:
                success_set = None
                colbench_scores = None
                sab_scores: Dict[str, float] = {}

                if benchmark == "scicode":
                    success_set = load_scicode_success(run_dir, run_id)
                elif benchmark == "corebench":
                    success_set = load_corebench_success(run_dir, run_id)
                elif benchmark == "colbench":
                    colbench_scores = load_colbench_correctness(run_dir, run_id)
                elif benchmark == "scienceagentbench":
                    sab_scores = load_scienceagentbench_eval(run_dir, run_id, benchmark_task_ids[benchmark])

            row: List[str] = []
            for b in BENCHMARKS:
                if b in skip_set:
                    continue
                for task_id in benchmark_task_ids.get(b, []):
                    if b != benchmark:
                        row.append("")
                        continue
                    if task_id not in ok_tasks:
                        row.append("")
                        continue
                    if args.reeval:
                        if task_id in success_map:
                            row.append("1" if success_map[task_id] == 1 else "0")
                        else:
                            row.append("")
                    else:
                        if benchmark in ("scicode", "corebench"):
                            if success_set is None:
                                row.append("")
                            else:
                                row.append("1" if task_id in success_set else "0")
                        elif benchmark == "colbench":
                            if colbench_scores is None:
                                row.append("")
                            else:
                                idx = int(task_id)
                                if idx < len(colbench_scores):
                                    row.append("1" if colbench_scores[idx] >= 0.999 else "0")
                                else:
                                    row.append("")
                        else:
                            if task_id in sab_scores:
                                row.append("1" if sab_scores[task_id] >= 1.0 else "0")
                            else:
                                row.append("")

            rows.append(row)
            row_labels.append(row_label)

    header = ["agent"] + columns
    output_path = Path(args.output) if args.output else repo_root / "output" / f"response_matrix_{args.prefix}.csv"
    rows_with_labels = [[row_labels[i]] + rows[i] for i in range(len(rows))]
    write_csv(output_path, header, rows_with_labels)
    print(f"Wrote CSV: {output_path}")

    if args.upload:
        try:
            from scripts.tmux_watcher import SPREADSHEET_ID
        except Exception:
            print("Failed to import SPREADSHEET_ID from scripts/tmux_watcher.py")
            sys.exit(1)

        gc = get_gspread_client(use_oauth=args.oauth)
        if gc is None:
            print("gspread not available or no credentials; skipping upload")
            sys.exit(1)
        sheet = gc.open_by_key(SPREADSHEET_ID)
        try:
            worksheet = sheet.worksheet(args.sheet_name)
        except Exception:
            worksheet = sheet.add_worksheet(title=args.sheet_name, rows=max(100, len(rows) + 5), cols=max(10, len(header) + 5))
        worksheet.clear()
        worksheet.update([header] + rows_with_labels, "A1")
        print(f"Uploaded to sheet: {args.sheet_name}")


if __name__ == "__main__":
    main()
