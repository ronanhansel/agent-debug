#!/usr/bin/env python3
"""
Benchmark Monitor - Terminal UI for monitoring benchmark progress.

Usage:
    python monitor.py <benchmark> [--once] [--interval N]

Examples:
    python monitor.py scicode
    python monitor.py scienceagentbench --interval 10
    python monitor.py corebench --once
"""

import os
import sys
import re
import json
import time
import argparse
import select
import termios
import tty
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Benchmark task counts
BENCHMARK_TASKS = {
    "scicode": 65,
    "scienceagentbench": 102,
    "corebench": 45,  # corebench_hard
    "corebench_hard": 45,
    "colbench": 1000,  # colbench_backend_programming (1000 tasks)
    "colbench_backend_programming": 1000,
    "colbench_frontend_design": 100,
}

# Directory name mapping (benchmark arg -> results folder name)
BENCHMARK_DIR_MAP = {
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",  # Default to backend (1000 tasks)
}

# Color codes for terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")


def find_latest_runs(benchmark: str, results_dir: Path) -> Dict[str, Path]:
    """Find the latest run directory for each config."""
    # Map benchmark name to directory name if needed
    dir_name = BENCHMARK_DIR_MAP.get(benchmark, benchmark)
    benchmark_dir = results_dir / dir_name
    if not benchmark_dir.exists():
        return {}

    # Group by config name (without timestamp)
    config_runs = defaultdict(list)

    for run_dir in benchmark_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Extract config name and timestamp
        # Format: {prefix}_{config}_{timestamp} e.g., scicode_moon1_gpt-5_generalist_20260122_100123
        name = run_dir.name

        # Find timestamp pattern (YYYYMMDD_HHMMSS)
        match = re.search(r'_(\d{8}_\d{6})$', name)
        if match:
            timestamp = match.group(1)
            config = name[:match.start()]
            config_runs[config].append((timestamp, run_dir))

    # Get latest for each config
    latest = {}
    for config, runs in config_runs.items():
        runs.sort(key=lambda x: x[0], reverse=True)
        latest[config] = runs[0][1]

    return latest


def parse_verbose_log(log_path: Path) -> Dict:
    """Parse verbose log for task completion and errors."""
    result = {
        "completed_tasks": 0,
        "total_tasks": 0,
        "last_activity": "",
        "last_task": "",
        "errors": [],
        "has_error": False,
        "is_finished": False,
        "active_tasks": 0,
    }

    if not log_path.exists():
        return result

    try:
        content = log_path.read_text()
        lines = content.split("\n")

        # Count completed tasks
        completed = re.findall(r"Completed task (\d+)", content)
        result["completed_tasks"] = len(completed)
        if completed:
            result["last_task"] = completed[-1]

        # Get last activity timestamp
        timestamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content, re.MULTILINE)
        if timestamps:
            result["last_activity"] = timestamps[-1]

        # Check for active tasks
        active_match = re.findall(r"active tasks: (\d+)", content)
        if active_match:
            result["active_tasks"] = int(active_match[-1])

        # Check for errors
        error_patterns = [
            r"(Error.*?)(?:\n|$)",
            r"(AgentGenerationError.*?)(?:\n|$)",
            r"(Traceback.*?)(?:\n\n|\Z)",
            r"(Exception.*?)(?:\n|$)",
            r"(FAILED.*?)(?:\n|$)",
        ]

        for pattern in error_patterns:
            errors = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for err in errors[-3:]:  # Keep last 3 errors
                err_text = err[:500] if len(err) > 500 else err
                if err_text not in result["errors"]:
                    result["errors"].append(err_text)

        result["has_error"] = len(result["errors"]) > 0

        # Check if finished (look for summary or no recent activity)
        if "Evaluation complete" in content or "All tasks completed" in content:
            result["is_finished"] = True

    except Exception as e:
        result["errors"].append(f"Log parse error: {e}")
        result["has_error"] = True

    return result


def get_config_status(run_dir: Path, total_tasks: int) -> Dict:
    """Get status for a single config run."""
    config_name = run_dir.name

    # Find verbose log
    verbose_logs = list(run_dir.glob("*_verbose.log"))
    if not verbose_logs:
        return {
            "config": config_name,
            "completed": 0,
            "total": total_tasks,
            "percent": 0,
            "last_activity": "N/A",
            "status": "no_log",
            "errors": [],
            "has_error": False,
        }

    log_data = parse_verbose_log(verbose_logs[0])

    completed = log_data["completed_tasks"]
    percent = (completed / total_tasks * 100) if total_tasks > 0 else 0

    # Determine status
    if log_data["is_finished"]:
        status = "finished"
    elif log_data["has_error"] and completed == 0:
        status = "error"
    elif log_data["has_error"]:
        status = "running_with_errors"
    elif completed >= total_tasks:
        status = "finished"
    elif completed > 0:
        status = "running"
    else:
        status = "starting"

    return {
        "config": config_name,
        "completed": completed,
        "total": total_tasks,
        "percent": percent,
        "last_activity": log_data["last_activity"],
        "last_task": log_data.get("last_task", ""),
        "active_tasks": log_data.get("active_tasks", 0),
        "status": status,
        "errors": log_data["errors"],
        "has_error": log_data["has_error"],
        "is_finished": log_data["is_finished"] or completed >= total_tasks,
    }


def extract_model_info(config_name: str) -> Tuple[str, str, str]:
    """Extract prefix, model, and agent type from config name."""
    # Example: scicode_moon1_gpt-5_generalist_20260122_100123
    # Returns: (prefix, model, agent_type)

    # Remove timestamp
    name = re.sub(r'_\d{8}_\d{6}$', '', config_name)

    # Known model patterns
    model_patterns = [
        r'(gpt-5[\w.-]*)',
        r'(gpt-4[\w.-]*)',
        r'(gpt_4_1[\w.-]*)',
        r'(o3-mini[\w.-]*)',
        r'(o3[\w.-]*)',
        r'(o4-mini[\w.-]*)',
        r'(o1[\w.-]*)',
        r'(claude[\w.-]*)',
    ]

    model = "unknown"
    for pattern in model_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            model = match.group(1)
            break

    # Extract prefix (first part before model)
    parts = name.split("_")
    prefix = ""
    agent_type = ""

    for i, part in enumerate(parts):
        if model.replace("-", "_").replace(".", "_") in "_".join(parts[i:i+2]).replace("-", "_"):
            prefix = "_".join(parts[:i])
            remaining = "_".join(parts[i:])
            # Agent type is after model
            agent_parts = remaining.replace(model, "").strip("_").split("_")
            agent_type = "_".join(p for p in agent_parts if p)
            break

    return prefix, model, agent_type


def format_progress_bar(percent: float, width: int = 30) -> str:
    """Create a colored progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled

    if percent >= 100:
        color = Colors.GREEN
    elif percent >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.BLUE

    bar = f"{color}{'█' * filled}{'░' * empty}{Colors.RESET}"
    return bar


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal size (rows, cols)."""
    try:
        import shutil
        size = shutil.get_terminal_size()
        return size.lines, size.columns
    except:
        return 24, 80


def get_verbose_log_path(config: Dict) -> Optional[Path]:
    """Get the verbose log path for a config."""
    config_name = config.get("config", "")
    # The run_dir should be stored in config, but we need to find it
    # For now, reconstruct from config name
    return None  # Will be set during display


def read_verbose_log_tail(log_path: Path, lines: int = 50) -> str:
    """Read the last N lines of a verbose log."""
    if not log_path or not log_path.exists():
        return "Log file not found"

    try:
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:]
            return ''.join(tail_lines)
    except Exception as e:
        return f"Error reading log: {e}"


def display_monitor_interactive(benchmark: str, configs: List[Dict], scroll_offset: int = 0,
                                 selected: int = -1, show_log: bool = False, run_dirs: Dict = None):
    """Display the terminal monitor UI with interactive features."""
    rows, cols = get_terminal_size()

    # Clear screen
    print("\033[2J\033[H", end="")

    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * (cols-2)}╗{Colors.RESET}")
    header = f"  BENCHMARK MONITOR - {benchmark.upper():<20} {timestamp}"
    print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}{header:<{cols-2}}{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * (cols-2)}╝{Colors.RESET}")

    # Sort configs by model then agent type
    configs.sort(key=lambda x: (extract_model_info(x["config"])[1], x["config"]))

    # Summary
    total_completed = sum(c["completed"] for c in configs)
    total_tasks = sum(c["total"] for c in configs)
    running = sum(1 for c in configs if c["status"] == "running")
    errors = sum(1 for c in configs if c["has_error"])
    finished = sum(1 for c in configs if c["is_finished"])

    print(f"\n  {Colors.BOLD}Summary:{Colors.RESET} {len(configs)} configs | "
          f"{Colors.GREEN}✓ {finished} finished{Colors.RESET} | "
          f"{Colors.BLUE}⟳ {running} running{Colors.RESET} | "
          f"{Colors.RED}✗ {errors} errors{Colors.RESET} | "
          f"Tasks: {total_completed}/{total_tasks}")

    # Controls help
    print(f"  {Colors.GRAY}[1-9,0] View log | [↑↓] Scroll | [q] Quit | [r] Refresh{Colors.RESET}")
    print()

    # Calculate visible area
    if show_log:
        max_visible = min(5, len(configs))  # Show fewer configs when log is visible
        log_lines = rows - max_visible - 15
    else:
        max_visible = min(rows - 12, len(configs))
        log_lines = 0

    # Config details header
    print(f"  {Colors.BOLD}{'#':<3} {'CONFIG':<42} {'PROGRESS':<35} {'STATUS':<10} {'LAST':<8}{Colors.RESET}")
    print(f"  {'-'*3} {'-'*42} {'-'*35} {'-'*10} {'-'*8}")

    # Apply scroll offset
    visible_configs = configs[scroll_offset:scroll_offset + max_visible]

    for i, cfg in enumerate(visible_configs):
        idx = scroll_offset + i
        num = idx + 1
        num_display = str(num) if num <= 9 else ('0' if num == 10 else '+')

        _, model, agent = extract_model_info(cfg["config"])
        short_name = f"{model}_{agent}"[:40]

        bar = format_progress_bar(cfg["percent"], 18)
        progress = f"{bar} {cfg['completed']:>3}/{cfg['total']:<3} ({cfg['percent']:>5.1f}%)"

        # Status with color
        status = cfg["status"]
        if status == "finished":
            status_str = f"{Colors.GREEN}done{Colors.RESET}"
        elif status == "running":
            status_str = f"{Colors.BLUE}run{Colors.RESET}"
        elif status == "error":
            status_str = f"{Colors.RED}ERR{Colors.RESET}"
        elif status == "running_with_errors":
            status_str = f"{Colors.YELLOW}run!{Colors.RESET}"
        else:
            status_str = f"{Colors.GRAY}{status[:4]}{Colors.RESET}"

        # Last activity time only
        last = cfg["last_activity"].split(" ")[-1][:8] if cfg["last_activity"] else "N/A"

        # Highlight selected row
        if idx == selected:
            print(f"{Colors.BOLD}{Colors.CYAN}▶ {num_display:<3} {short_name:<42} {progress:<50} {status_str:<18} {last:<8}{Colors.RESET}")
        else:
            print(f"  {num_display:<3} {short_name:<42} {progress:<50} {status_str:<18} {last:<8}")

    # Show scroll indicators
    if scroll_offset > 0:
        print(f"  {Colors.GRAY}↑ {scroll_offset} more above{Colors.RESET}")
    if scroll_offset + max_visible < len(configs):
        print(f"  {Colors.GRAY}↓ {len(configs) - scroll_offset - max_visible} more below{Colors.RESET}")

    # Show log preview if selected
    if show_log and selected >= 0 and selected < len(configs):
        cfg = configs[selected]
        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}{'─' * 70}{Colors.RESET}")
        _, model, agent = extract_model_info(cfg["config"])
        print(f"  {Colors.BOLD}{Colors.MAGENTA}LOG: {model}_{agent}{Colors.RESET}")
        print(f"  {Colors.MAGENTA}{'─' * 70}{Colors.RESET}")

        # Get log path from run_dirs
        if run_dirs and cfg["config"] in run_dirs:
            run_dir = run_dirs[cfg["config"]]
            verbose_logs = list(run_dir.glob("*_verbose.log"))
            if verbose_logs:
                log_content = read_verbose_log_tail(verbose_logs[0], lines=log_lines)
                # Print log with line limit
                for line in log_content.split('\n')[-log_lines:]:
                    # Truncate long lines
                    if len(line) > cols - 4:
                        line = line[:cols-7] + "..."
                    print(f"  {Colors.GRAY}{line}{Colors.RESET}")
            else:
                print(f"  {Colors.GRAY}No verbose log found{Colors.RESET}")
        else:
            print(f"  {Colors.GRAY}Run directory not found{Colors.RESET}")

    return max_visible


def display_monitor(benchmark: str, configs: List[Dict]):
    """Display the terminal monitor UI (non-interactive version)."""
    # Clear screen
    print("\033[2J\033[H", end="")

    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}  {Colors.BOLD}BENCHMARK MONITOR{Colors.RESET} - {benchmark.upper():<20} {Colors.GRAY}{timestamp}{Colors.RESET}  {Colors.BOLD}{Colors.CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()

    # Sort configs by model then agent type
    configs.sort(key=lambda x: (extract_model_info(x["config"])[1], x["config"]))

    # Summary
    total_completed = sum(c["completed"] for c in configs)
    total_tasks = sum(c["total"] for c in configs)
    running = sum(1 for c in configs if c["status"] == "running")
    errors = sum(1 for c in configs if c["has_error"])
    finished = sum(1 for c in configs if c["is_finished"])

    print(f"  {Colors.BOLD}Summary:{Colors.RESET} {len(configs)} configs | "
          f"{Colors.GREEN}✓ {finished} finished{Colors.RESET} | "
          f"{Colors.BLUE}⟳ {running} running{Colors.RESET} | "
          f"{Colors.RED}✗ {errors} errors{Colors.RESET} | "
          f"Tasks: {total_completed}/{total_tasks}")
    print()

    # Config details
    print(f"  {Colors.BOLD}{'#':<3} {'CONFIG':<42} {'PROGRESS':<35} {'STATUS':<10} {'LAST':<8}{Colors.RESET}")
    print(f"  {'-'*3} {'-'*42} {'-'*35} {'-'*10} {'-'*8}")

    for i, cfg in enumerate(configs):
        num = i + 1
        num_display = str(num) if num <= 9 else ('0' if num == 10 else '+')

        _, model, agent = extract_model_info(cfg["config"])
        short_name = f"{model}_{agent}"[:40]

        bar = format_progress_bar(cfg["percent"], 18)
        progress = f"{bar} {cfg['completed']:>3}/{cfg['total']:<3} ({cfg['percent']:>5.1f}%)"

        # Status with color
        status = cfg["status"]
        if status == "finished":
            status_str = f"{Colors.GREEN}done{Colors.RESET}"
        elif status == "running":
            status_str = f"{Colors.BLUE}run{Colors.RESET}"
        elif status == "error":
            status_str = f"{Colors.RED}ERR{Colors.RESET}"
        elif status == "running_with_errors":
            status_str = f"{Colors.YELLOW}run!{Colors.RESET}"
        else:
            status_str = f"{Colors.GRAY}{status[:4]}{Colors.RESET}"

        # Last activity time only
        last = cfg["last_activity"].split(" ")[-1][:8] if cfg["last_activity"] else "N/A"

        print(f"  {num_display:<3} {short_name:<42} {progress:<50} {status_str:<18} {last:<8}")

        # Show errors if any (condensed)
        if cfg["errors"] and cfg["status"] in ["error", "running_with_errors"]:
            err_preview = cfg["errors"][-1][:60].replace("\n", " ")
            print(f"      {Colors.RED}└─ {err_preview}...{Colors.RESET}")

    print()


def log_to_spreadsheet(benchmark: str, config_data: Dict, event: str):
    """Log events to Google Spreadsheet.

    Events: 'start', 'finished', 'error'
    """
    spreadsheet_id = os.environ.get("GOOGLE_SPREADSHEET_ID")
    if not spreadsheet_id:
        return False

    prefix, model, agent_type = extract_model_info(config_data["config"])
    task_id = config_data["config"]  # Full config name as task_id

    row_data = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        benchmark,                                       # benchmark
        task_id,                                         # task_id
        model,                                           # model
        agent_type,                                      # config
        event,                                           # status (start/finished/error)
        config_data.get("completed", 0),                 # completed
        config_data.get("total", 0),                     # total
        "\n".join(config_data.get("errors", [])[-3:]) if event == "error" else "",  # trace
    ]

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # Try service account - check project directory first, then home
        script_dir = Path(__file__).parent
        sa_path = script_dir.parent / ".config" / "service_account.json"
        if not sa_path.exists():
            sa_path = Path.home() / ".config" / "gspread" / "service_account.json"
        if sa_path.exists():
            scopes = ['https://www.googleapis.com/auth/spreadsheets']
            creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
            gc = gspread.authorize(creds)

            sheet = gc.open_by_key(spreadsheet_id)
            try:
                worksheet = sheet.worksheet("Monitor Log")
            except gspread.WorksheetNotFound:
                worksheet = sheet.add_worksheet(title="Monitor Log", rows=1000, cols=15)
                # Add headers
                headers = ["timestamp", "benchmark", "task_id", "model", "config",
                          "status", "completed", "total", "trace"]
                worksheet.append_row(headers)

            worksheet.append_row(row_data)
            print(f"  {Colors.GREEN}📊 Logged: {event} - {model}_{agent_type}{Colors.RESET}")
            return True

    except Exception as e:
        print(f"{Colors.YELLOW}Spreadsheet log failed: {e}{Colors.RESET}")

    return False


def save_local_log(benchmark: str, config_data: Dict, event: str, log_dir: Path):
    """Save log entry to local JSON file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{benchmark}_monitor_log.jsonl"

    prefix, model, agent_type = extract_model_info(config_data["config"])

    entry = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": benchmark,
        "task_id": config_data["config"],
        "model": model,
        "config": agent_type,
        "status": event,
        "completed": config_data.get("completed", 0),
        "total": config_data.get("total", 0),
        "trace": "\n".join(config_data.get("errors", [])[-3:]) if event == "error" else "",
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# Global batch queue for spreadsheet logging
_spreadsheet_batch_queue = []
_spreadsheet_last_flush = 0
BATCH_FLUSH_INTERVAL = 2  # seconds between batch flushes
MAX_BATCH_SIZE = 20  # max rows per batch


def queue_spreadsheet_log(benchmark: str, config_data: Dict, event: str):
    """Queue a log entry for batch spreadsheet update."""
    global _spreadsheet_batch_queue

    prefix, model, agent_type = extract_model_info(config_data["config"])
    task_id = config_data["config"]

    row_data = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        benchmark,                                       # benchmark
        task_id,                                         # task_id
        model,                                           # model
        agent_type,                                      # config
        event,                                           # status (start/finished/error)
        config_data.get("completed", 0),                 # completed
        config_data.get("total", 0),                     # total
        "\n".join(config_data.get("errors", [])[-3:]) if event == "error" else "",  # trace
    ]

    _spreadsheet_batch_queue.append(row_data)


def flush_spreadsheet_batch(force: bool = False) -> bool:
    """Flush queued log entries to Google Spreadsheet with retry logic.

    Args:
        force: If True, flush immediately regardless of timing

    Returns:
        True if flush was successful or queue was empty, False on error
    """
    global _spreadsheet_batch_queue, _spreadsheet_last_flush

    if not _spreadsheet_batch_queue:
        return True

    current_time = time.time()

    # Only flush if enough time has passed or forced
    if not force and (current_time - _spreadsheet_last_flush) < BATCH_FLUSH_INTERVAL:
        return True

    spreadsheet_id = os.environ.get("GOOGLE_SPREADSHEET_ID")
    if not spreadsheet_id:
        _spreadsheet_batch_queue.clear()
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # Try service account - check project directory first, then home
        script_dir = Path(__file__).parent
        sa_path = script_dir.parent / ".config" / "service_account.json"
        if not sa_path.exists():
            sa_path = Path.home() / ".config" / "gspread" / "service_account.json"

        if not sa_path.exists():
            print(f"  {Colors.YELLOW}No service account found{Colors.RESET}")
            _spreadsheet_batch_queue.clear()
            return False

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
        gc = gspread.authorize(creds)

        sheet = gc.open_by_key(spreadsheet_id)
        try:
            worksheet = sheet.worksheet("Monitor Log")
        except gspread.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title="Monitor Log", rows=1000, cols=15)
            # Add headers
            headers = ["timestamp", "benchmark", "task_id", "model", "config",
                      "status", "completed", "total", "trace"]
            worksheet.append_row(headers)

        # Batch update with retry logic
        rows_to_add = _spreadsheet_batch_queue[:MAX_BATCH_SIZE]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Use append_rows for batch insert (more efficient than multiple append_row)
                worksheet.append_rows(rows_to_add, value_input_option='RAW')

                # Remove successfully added rows from queue
                _spreadsheet_batch_queue = _spreadsheet_batch_queue[len(rows_to_add):]
                _spreadsheet_last_flush = current_time

                print(f"  {Colors.GREEN}📊 Logged {len(rows_to_add)} entries to spreadsheet{Colors.RESET}")
                return True

            except gspread.exceptions.APIError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 3, 5 seconds
                    print(f"  {Colors.YELLOW}Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries}){Colors.RESET}")
                    time.sleep(wait_time)
                else:
                    raise

        # All retries failed
        print(f"  {Colors.RED}Failed to log after {max_retries} attempts, keeping in queue{Colors.RESET}")
        return False

    except Exception as e:
        print(f"  {Colors.YELLOW}Spreadsheet batch log failed: {e}{Colors.RESET}")
        return False


def get_key_nonblocking(timeout: float = 0.1) -> Optional[str]:
    """Get a keypress without blocking, with timeout."""
    if not sys.stdin.isatty():
        return None

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if key == '\x1b':
                # Read additional characters for escape sequences
                sys.stdin.read(1)  # [
                arrow = sys.stdin.read(1)
                if arrow == 'A':
                    return 'UP'
                elif arrow == 'B':
                    return 'DOWN'
                elif arrow == 'C':
                    return 'RIGHT'
                elif arrow == 'D':
                    return 'LEFT'
            return key
    except:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return None


def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark progress")
    parser.add_argument("benchmark", nargs="?", default=None,
                        choices=["scicode", "scienceagentbench", "corebench", "colbench", "all"],
                        help="Benchmark to monitor (or 'all' for all benchmarks)")
    parser.add_argument("--all", action="store_true", help="Monitor all benchmarks")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with log preview")
    args = parser.parse_args()

    # Handle --all flag or "all" argument
    if args.all or args.benchmark == "all":
        benchmarks = ["scicode", "scienceagentbench", "corebench", "colbench"]
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        print("Usage: monitor.py <benchmark> or monitor.py --all")
        print("Benchmarks: scicode, scienceagentbench, corebench, colbench, all")
        return

    load_env()

    # Determine results directory
    script_dir = Path(__file__).parent
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = script_dir.parent / "results"

    log_dir = script_dir.parent / "monitor_logs"

    # Track state: config -> {"status": str, "completed": int, "logged_start": bool}
    config_states = {}

    try:
        last_refresh = 0
        while True:
            current_time = time.time()

            # Refresh data at interval
            if current_time - last_refresh >= args.interval or last_refresh == 0:
                last_refresh = current_time

                # Clear screen
                print("\033[2J\033[H", end="")

                # Header
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
                title = "ALL BENCHMARKS" if len(benchmarks) > 1 else benchmarks[0].upper()
                print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}  {Colors.BOLD}BENCHMARK MONITOR{Colors.RESET} - {title:<20} {Colors.GRAY}{timestamp}{Colors.RESET}  {Colors.BOLD}{Colors.CYAN}║{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
                print()

                all_configs = []

                for benchmark in benchmarks:
                    total_tasks = BENCHMARK_TASKS.get(benchmark, 100)

                    # Find latest runs
                    latest_runs = find_latest_runs(benchmark, results_dir)

                    if not latest_runs:
                        print(f"  {Colors.GRAY}[{benchmark}] No runs found{Colors.RESET}")
                        continue

                    # Get status for each config and log events
                    configs = []
                    for config, run_dir in latest_runs.items():
                        status = get_config_status(run_dir, total_tasks)
                        status["benchmark"] = benchmark  # Add benchmark to status
                        configs.append(status)

                        config_key = f"{benchmark}:{status['config']}"
                        prev_state = config_states.get(config_key, {})

                        # Log START event (first time we see this config)
                        if config_key not in config_states:
                            queue_spreadsheet_log(benchmark, status, "start")
                            save_local_log(benchmark, status, "start", log_dir)
                            config_states[config_key] = {
                                "status": status["status"],
                                "completed": status["completed"],
                                "logged_error": False,
                                "logged_finished": False,
                            }
                            prev_state = config_states[config_key]

                        # Log ERROR event (first error for this config)
                        if status["has_error"] and not prev_state.get("logged_error"):
                            queue_spreadsheet_log(benchmark, status, "error")
                            save_local_log(benchmark, status, "error", log_dir)
                            config_states[config_key]["logged_error"] = True

                        # Log FINISHED event
                        if status["is_finished"] and not prev_state.get("logged_finished"):
                            queue_spreadsheet_log(benchmark, status, "finished")
                            save_local_log(benchmark, status, "finished", log_dir)
                            config_states[config_key]["logged_finished"] = True

                        config_states[config_key]["completed"] = status["completed"]

                    all_configs.extend(configs)

                    # Display summary per benchmark
                    total_completed = sum(c["completed"] for c in configs)
                    total_t = sum(c["total"] for c in configs)
                    finished = sum(1 for c in configs if c["is_finished"])
                    errors = sum(1 for c in configs if c["has_error"])
                    running = sum(1 for c in configs if c["status"] == "running")

                    print(f"  {Colors.BOLD}[{benchmark}]{Colors.RESET} {len(configs)} configs | "
                          f"{Colors.GREEN}✓{finished}{Colors.RESET} "
                          f"{Colors.BLUE}⟳{running}{Colors.RESET} "
                          f"{Colors.RED}✗{errors}{Colors.RESET} | "
                          f"{total_completed}/{total_t} tasks")

                # Grand total
                if all_configs:
                    print()
                    total_completed = sum(c["completed"] for c in all_configs)
                    total_t = sum(c["total"] for c in all_configs)
                    finished = sum(1 for c in all_configs if c["is_finished"])
                    errors = sum(1 for c in all_configs if c["has_error"])
                    print(f"  {Colors.BOLD}TOTAL:{Colors.RESET} {len(all_configs)} configs | "
                          f"{Colors.GREEN}✓ {finished} finished{Colors.RESET} | "
                          f"{Colors.RED}✗ {errors} errors{Colors.RESET} | "
                          f"{total_completed}/{total_t} tasks")

                # Flush any queued spreadsheet entries
                flush_spreadsheet_batch(force=True)

                if args.once:
                    break

                print(f"\n  {Colors.GRAY}Refreshing in {args.interval}s... (Ctrl+C to stop){Colors.RESET}")

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        # Final flush of any remaining queued entries (may need multiple batches)
        while _spreadsheet_batch_queue:
            remaining = len(_spreadsheet_batch_queue)
            print(f"\n  {Colors.GRAY}Flushing {min(remaining, MAX_BATCH_SIZE)}/{remaining} remaining entries...{Colors.RESET}")
            if not flush_spreadsheet_batch(force=True):
                break  # Stop if flush fails to prevent infinite loop
            time.sleep(1)  # Brief pause between batches to avoid rate limits
        print(f"\n{Colors.CYAN}Monitor stopped.{Colors.RESET}")


if __name__ == "__main__":
    main()
