#!/usr/bin/env python3
"""
TMux Session Watcher - Monitors benchmark runs and updates Google Sheets.

Usage:
    python scripts/tmux_watcher.py [--interval 30] [--sessions long,long_2,long_3,long_4]

Requirements:
    pip install gspread google-auth google-auth-oauthlib

Google Sheets Setup:
    1. Go to Google Cloud Console -> APIs & Services -> Credentials
    2. Create a Service Account or OAuth credentials
    3. Download JSON and save as ~/.config/gspread/service_account.json
    OR
    4. Use OAuth: run with --oauth flag first time to authenticate
"""

import subprocess
import re
import time
import json
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Google Sheets configuration
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mhbKQ6TqqIssa5_UsCbFYdp4fRPKHzUaQc_Lw-2lNOU/edit"
SPREADSHEET_ID = "1mhbKQ6TqqIssa5_UsCbFYdp4fRPKHzUaQc_Lw-2lNOU"

# Session to benchmark mapping
SESSION_BENCHMARKS = {
    "long": "scicode",
    "long_2": "scienceagentbench",
    "long_3": "corebench",
    "long_4": "colbench",
    # Add more as needed
    "long_10": "unknown",
}


def capture_tmux_output(session: str, lines: int = 5000) -> str:
    """Capture output from a tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", f"-{lines}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def find_benchmark_processes() -> Dict[str, Dict]:
    """Find running benchmark processes directly (bypasses tmux issues)."""
    processes = {}
    try:
        result = subprocess.run(
            ["pgrep", "-af", "run_benchmark_fixes.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid, cmdline = parts

            # Extract benchmark name
            benchmark = "unknown"
            if "--benchmark" in cmdline:
                match = re.search(r"--benchmark\s+(\w+)", cmdline)
                if match:
                    benchmark = match.group(1)

            # Extract prefix
            prefix = ""
            if "--prefix" in cmdline:
                match = re.search(r"--prefix\s+(\S+)", cmdline)
                if match:
                    prefix = match.group(1)

            # Find the terminal
            try:
                tty = os.readlink(f"/proc/{pid}/fd/1")
            except:
                tty = "unknown"

            processes[benchmark] = {
                "pid": pid,
                "cmdline": cmdline,
                "tty": tty,
                "prefix": prefix,
            }
    except Exception as e:
        pass
    return processes


def capture_process_output(pid: str, lines: int = 200) -> str:
    """Try to capture recent output from a process's log file or scrollback."""
    # Check if there's a log file being written
    try:
        # Look for open log files
        result = subprocess.run(
            ["lsof", "-p", pid],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split("\n"):
            if ".log" in line or "output" in line.lower():
                parts = line.split()
                if len(parts) > 8:
                    log_path = parts[-1]
                    if os.path.exists(log_path):
                        with open(log_path, "r") as f:
                            return "\n".join(f.readlines()[-lines:])
    except:
        pass

    return ""


def parse_benchmark_status(output: str) -> Dict:
    """Parse benchmark run output for statistics."""
    stats = {
        "benchmark": "unknown",
        "prefix": "",
        "total_configs": 0,
        "total_tasks": 0,
        "running_configs": [],
        "completed": 0,
        "failed": 0,
        "success": 0,
        "current_tasks": [],
        "run_ids": [],
        "models": set(),
        "last_activity": "",
        "status": "unknown",
        "raw_summary": "",
        "success_details": [],
        "failed_details": [],
        "progress_pct": 0,
    }

    lines = output.strip().split("\n")

    # Parse benchmark name
    for line in lines:
        if "Benchmark:" in line and "[main]" in line:
            match = re.search(r"Benchmark:\s+(\w+)", line)
            if match:
                stats["benchmark"] = match.group(1)

        # Parse prefix
        if "prefix" in line.lower() and "'" in line:
            match = re.search(r"prefix\s+'([^']+)'", line)
            if match:
                stats["prefix"] = match.group(1)

        # Parse total configs
        if "Found" in line and "configurations" in line:
            match = re.search(r"Found\s+(\d+)\s+configurations", line)
            if match:
                stats["total_configs"] = int(match.group(1))

        # Parse total tasks
        if "total tasks" in line.lower() or "Loaded" in line and "tasks" in line:
            match = re.search(r"(\d+)\s+(?:total\s+)?tasks", line)
            if match:
                stats["total_tasks"] = int(match.group(1))

        # Parse running configs
        if "[hal] Running:" in line:
            match = re.search(r"Running:\s+(\S+)", line)
            if match:
                config = match.group(1)
                if config not in stats["running_configs"]:
                    stats["running_configs"].append(config)

        # Parse run IDs
        if "[hal] Run ID:" in line:
            match = re.search(r"Run ID:\s+(\S+)", line)
            if match:
                run_id = match.group(1)
                if run_id not in stats["run_ids"]:
                    stats["run_ids"].append(run_id)

        # Parse models
        if "[hal] Model:" in line:
            match = re.search(r"Model:\s+(\S+)", line)
            if match:
                stats["models"].add(match.group(1))

        # Parse SUCCESS/FAILED with details (format: [timestamp] [key] SUCCESS|FAILED)
        # Also handles: status = "SUCCESS" if success else "FAILED"
        success_match = re.search(r"\[([^\]]+)\]\s*SUCCESS", line)
        if success_match:
            stats["success"] += 1
            stats["completed"] += 1
            key = success_match.group(1)
            if key not in stats["success_details"]:
                stats["success_details"].append(key)

        failed_match = re.search(r"\[([^\]]+)\]\s*FAILED", line)
        if failed_match:
            stats["failed"] += 1
            stats["completed"] += 1
            key = failed_match.group(1)
            if key not in stats["failed_details"]:
                stats["failed_details"].append(key)

        # Also check for summary lines at the end
        if line.startswith("Failed:") or line.startswith("Success:"):
            match = re.search(r"(\d+)", line)
            if match:
                count = int(match.group(1))
                if "Failed:" in line:
                    stats["failed"] = max(stats["failed"], count)
                elif "Success:" in line:
                    stats["success"] = max(stats["success"], count)

    # Get last activity timestamp
    timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", output)
    if timestamps:
        stats["last_activity"] = timestamps[-1]

    # Calculate progress
    total_expected = stats["total_configs"]
    if total_expected > 0:
        stats["progress_pct"] = round((stats["completed"] / total_expected) * 100, 1)

    # Determine status
    if stats["completed"] > 0 and stats["completed"] >= stats["total_configs"] and stats["total_configs"] > 0:
        stats["status"] = "completed"
    elif stats["failed"] > 0 and stats["success"] == 0:
        stats["status"] = "failing"
    elif stats["running_configs"]:
        stats["status"] = "running"
    elif "ERROR" in output.upper() or "Traceback" in output:
        stats["status"] = "error"
    elif stats["total_configs"] > 0:
        stats["status"] = "starting"
    else:
        stats["status"] = "unknown"

    # Get summary from last lines
    last_lines = lines[-30:] if len(lines) > 30 else lines
    stats["raw_summary"] = "\n".join(last_lines)

    # Convert set to list for JSON serialization
    stats["models"] = list(stats["models"])

    return stats


def scan_results_directory(benchmark: str, prefix: str) -> Dict:
    """Scan results directory for completed/failed runs."""
    stats = {
        "success": 0,
        "failed": 0,
        "completed": 0,
        "running_configs": [],
        "models": set(),
        "last_activity": "",
        "success_details": [],
        "failed_details": [],
    }

    results_dir = Path(f"results/{benchmark}")
    if not results_dir.exists():
        return stats

    # Find directories matching the prefix
    for run_dir in results_dir.glob(f"{prefix}*"):
        if not run_dir.is_dir():
            continue

        config_name = run_dir.name.replace(prefix, "").split("_202")[0]  # Remove timestamp

        # Check for results.json or output.json
        result_file = run_dir / "results.json"
        output_file = run_dir / "0" / "output.json"

        has_result = result_file.exists() or output_file.exists()

        # Check log file for errors
        log_files = list(run_dir.glob("*.log"))
        has_error = False
        for log_file in log_files:
            if "verbose" not in log_file.name:
                try:
                    content = log_file.read_text()[-5000:]  # Last 5KB
                    if "ERROR" in content or "FAILED" in content or "Traceback" in content:
                        has_error = True
                    # Get last timestamp
                    timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", content)
                    if timestamps:
                        stats["last_activity"] = timestamps[-1]
                except:
                    pass

        # Extract model name from config
        model_match = re.search(r"(gpt-[\w.-]+|o[134]-\w+|claude-\w+)", config_name)
        if model_match:
            stats["models"].add(model_match.group(1))

        if has_result and not has_error:
            stats["success"] += 1
            stats["success_details"].append(config_name)
        elif has_error:
            stats["failed"] += 1
            stats["failed_details"].append(config_name)
        else:
            stats["running_configs"].append(config_name)

        stats["completed"] = stats["success"] + stats["failed"]

    stats["models"] = list(stats["models"])
    return stats


def get_process_stats() -> Dict[str, Dict]:
    """Get statistics from running benchmark processes directly."""
    all_stats = {}
    processes = find_benchmark_processes()

    for benchmark, proc_info in processes.items():
        pid = proc_info["pid"]
        prefix = proc_info["prefix"]

        # Scan results directory for actual completion status
        result_stats = scan_results_directory(benchmark, prefix)

        stats = {
            "benchmark": benchmark,
            "prefix": prefix,
            "status": "running",
            "total_configs": 0,
            "running_configs": result_stats.get("running_configs", []),
            "completed": result_stats.get("completed", 0),
            "failed": result_stats.get("failed", 0),
            "success": result_stats.get("success", 0),
            "models": result_stats.get("models", []),
            "last_activity": result_stats.get("last_activity", ""),
            "progress_pct": 0,
            "success_details": result_stats.get("success_details", []),
            "failed_details": result_stats.get("failed_details", []),
        }

        stats["pid"] = pid
        stats["tty"] = proc_info["tty"]
        stats["captured_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_stats[benchmark] = stats

    return all_stats


def get_all_session_stats(sessions: List[str]) -> Dict[str, Dict]:
    """Get statistics for all specified sessions."""
    all_stats = {}
    for session in sessions:
        output = capture_tmux_output(session)
        if output and not output.startswith("ERROR"):
            stats = parse_benchmark_status(output)
            stats["session"] = session
            stats["captured_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_stats[session] = stats
        else:
            all_stats[session] = {
                "session": session,
                "status": "not_found" if "no session" in output.lower() else "error",
                "error": output,
                "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    return all_stats


def format_stats_for_display(all_stats: Dict[str, Dict]) -> str:
    """Format statistics for terminal display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"TMux Benchmark Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    for session, stats in all_stats.items():
        lines.append(f"\n📺 Session: {session}")
        lines.append("-" * 40)

        if stats.get("status") == "not_found":
            lines.append("  ❌ Session not found")
            continue

        benchmark = stats.get("benchmark", "unknown")
        prefix = stats.get("prefix", "N/A")
        status = stats.get("status", "unknown")

        status_emoji = {
            "running": "🟢",
            "completed": "✅",
            "error": "🔴",
            "failing": "🟠",
            "starting": "🟡",
            "unknown": "⚪"
        }.get(status, "⚪")

        lines.append(f"  Benchmark: {benchmark}")
        lines.append(f"  Prefix: {prefix}")
        lines.append(f"  Status: {status_emoji} {status}")

        # Progress bar
        total = stats.get("total_configs", 0)
        completed = stats.get("completed", 0)
        progress_pct = stats.get("progress_pct", 0)
        if total > 0:
            bar_width = 20
            filled = int(bar_width * completed / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  Progress: [{bar}] {completed}/{total} ({progress_pct}%)")
        else:
            lines.append(f"  Configs: {len(stats.get('running_configs', []))}/? (initializing)")

        success = stats.get('success', 0)
        failed = stats.get('failed', 0)
        lines.append(f"  Results: ✓ {success} success | ✗ {failed} failed")

        # Show recent failures if any
        failed_details = stats.get("failed_details", [])[-3:]
        if failed_details:
            lines.append(f"  Recent failures: {', '.join(failed_details)}")

        lines.append(f"  Last Activity: {stats.get('last_activity', 'N/A')}")

        models = stats.get("models", [])
        if models:
            lines.append(f"  Models: {', '.join(m.split('/')[-1] for m in models[:5])}{'...' if len(models) > 5 else ''}")

    # Summary
    lines.append("\n" + "-" * 80)
    total_success = sum(s.get("success", 0) for s in all_stats.values())
    total_failed = sum(s.get("failed", 0) for s in all_stats.values())
    lines.append(f"TOTAL: ✓ {total_success} success | ✗ {total_failed} failed")
    lines.append("=" * 80)
    return "\n".join(lines)


def post_to_google_form(all_stats: Dict[str, Dict], form_url: str):
    """Post stats to a Google Form (no auth required).

    To set up:
    1. Create a Google Form with these fields:
       - Timestamp (short answer)
       - Session (short answer)
       - Benchmark (short answer)
       - Status (short answer)
       - Success (short answer)
       - Failed (short answer)
       - Models (paragraph)
    2. Get the form URL and entry IDs from the form's HTML
    3. Pass the form response URL to this function
    """
    import urllib.request
    import urllib.parse

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for session, stats in all_stats.items():
        data = {
            'entry.1': timestamp,  # Replace with actual entry IDs
            'entry.2': session,
            'entry.3': stats.get("benchmark", "unknown"),
            'entry.4': stats.get("status", "unknown"),
            'entry.5': str(stats.get("success", 0)),
            'entry.6': str(stats.get("failed", 0)),
            'entry.7': ", ".join(stats.get("models", [])[:5]),
        }

        try:
            encoded = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(form_url, data=encoded)
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"⚠️ Form submit failed: {e}")

    print(f"📝 Posted to Google Form")
    return True


def post_to_jsonbin(all_stats: Dict[str, Dict], bin_id: str = None, api_key: str = None):
    """Post stats to JSONBin.io (free, simple API key).

    To set up:
    1. Go to https://jsonbin.io and create free account
    2. Create a bin and get the bin ID
    3. Get your API key from account settings
    4. Set env vars: JSONBIN_ID and JSONBIN_KEY
    """
    import urllib.request

    bin_id = bin_id or os.environ.get("JSONBIN_ID")
    api_key = api_key or os.environ.get("JSONBIN_KEY")

    if not bin_id or not api_key:
        print("⚠️ JSONBin not configured. Set JSONBIN_ID and JSONBIN_KEY env vars")
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data
    data = {
        "timestamp": timestamp,
        "sessions": {}
    }
    for session, stats in all_stats.items():
        data["sessions"][session] = {
            "benchmark": stats.get("benchmark", "unknown"),
            "status": stats.get("status", "unknown"),
            "success": stats.get("success", 0),
            "failed": stats.get("failed", 0),
            "progress": stats.get("progress_pct", 0),
            "models": stats.get("models", []),
        }

    try:
        url = f"https://api.jsonbin.io/v3/b/{bin_id}"
        req = urllib.request.Request(url, method="PUT")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Master-Key", api_key)
        req.data = json.dumps(data).encode()
        urllib.request.urlopen(req, timeout=10)
        print(f"📤 Updated JSONBin: https://jsonbin.io/b/{bin_id}")
        return True
    except Exception as e:
        print(f"⚠️ JSONBin update failed: {e}")
        return False


def generate_html_dashboard(all_stats: Dict[str, Dict], html_path: str = "benchmark_dashboard.html"):
    """Generate a simple HTML dashboard that auto-refreshes."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_success = sum(s.get("success", 0) for s in all_stats.values())
    total_failed = sum(s.get("failed", 0) for s in all_stats.values())

    rows = ""
    for session, stats in all_stats.items():
        status = stats.get("status", "unknown")
        status_colors = {
            "running": "#22c55e",
            "completed": "#3b82f6",
            "error": "#ef4444",
            "failing": "#f97316",
            "starting": "#eab308",
            "unknown": "#6b7280"
        }
        color = status_colors.get(status, "#6b7280")

        progress = stats.get("progress_pct", 0)
        rows += f"""
        <tr>
            <td>{session}</td>
            <td>{stats.get("benchmark", "unknown")}</td>
            <td style="color: {color}; font-weight: bold;">{status.upper()}</td>
            <td>
                <div style="background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                    <div style="background: {color}; width: {progress}%; height: 20px;"></div>
                </div>
                {progress}%
            </td>
            <td style="color: #22c55e;">✓ {stats.get("success", 0)}</td>
            <td style="color: #ef4444;">✗ {stats.get("failed", 0)}</td>
            <td>{stats.get("last_activity", "N/A")}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Monitor</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f9fafb; }}
        h1 {{ color: #1f2937; }}
        table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card h2 {{ margin: 0 0 10px 0; font-size: 14px; color: #6b7280; }}
        .card .value {{ font-size: 32px; font-weight: bold; }}
        .success {{ color: #22c55e; }}
        .failed {{ color: #ef4444; }}
        .timestamp {{ color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>🖥️ Benchmark Monitor</h1>
    <p class="timestamp">Last updated: {timestamp} (auto-refreshes every 30s)</p>

    <div class="summary">
        <div class="card">
            <h2>Total Success</h2>
            <div class="value success">✓ {total_success}</div>
        </div>
        <div class="card">
            <h2>Total Failed</h2>
            <div class="value failed">✗ {total_failed}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Session</th>
                <th>Benchmark</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Success</th>
                <th>Failed</th>
                <th>Last Activity</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html)
    print(f"🌐 Dashboard: file://{Path(html_path).absolute()}")
    return True


def save_stats_to_csv(all_stats: Dict[str, Dict], csv_path: str = "benchmark_status.csv"):
    """Save statistics to a local CSV file (always works)."""
    import csv

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = [
        "Timestamp", "Session", "Benchmark", "Prefix", "Status",
        "Total Configs", "Running", "Success", "Failed", "Last Activity", "Models"
    ]

    rows = []
    for session, stats in all_stats.items():
        row = [
            timestamp,
            session,
            stats.get("benchmark", "unknown"),
            stats.get("prefix", ""),
            stats.get("status", "unknown"),
            stats.get("total_configs", 0),
            len(stats.get("running_configs", [])),
            stats.get("success", 0),
            stats.get("failed", 0),
            stats.get("last_activity", ""),
            ", ".join(m.split("/")[-1] for m in list(stats.get("models", []))[:5])
        ]
        rows.append(row)

    # Append mode to keep history
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(rows)

    print(f"📄 Saved to {csv_path}")
    return True


def update_google_sheet(all_stats: Dict[str, Dict], use_oauth: bool = False):
    """Update Google Sheet with current statistics."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("⚠️ gspread not installed. Run: pip install gspread google-auth")
        print("   Falling back to local CSV...")
        return save_stats_to_csv(all_stats)

    # Try different authentication methods
    gc = None

    # Method 1: Service account
    sa_path = Path.home() / ".config" / "gspread" / "service_account.json"
    if sa_path.exists():
        try:
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(str(sa_path), scopes=scopes)
            gc = gspread.authorize(creds)
            print("✅ Authenticated with service account")
        except Exception as e:
            print(f"⚠️ Service account auth failed: {e}")

    # Method 2: OAuth (interactive)
    if gc is None and use_oauth:
        try:
            gc = gspread.oauth()
            print("✅ Authenticated with OAuth")
        except Exception as e:
            print(f"⚠️ OAuth failed: {e}")

    # Method 3: No credentials - save to CSV instead
    if gc is None:
        print("⚠️ No Google credentials found. Saving to local CSV instead.")
        print("\nTo set up Google Sheets access:")
        print("1. Create a service account at https://console.cloud.google.com/")
        print("2. Download JSON credentials")
        print(f"3. Save to: {sa_path}")
        print("4. Share the spreadsheet with the service account email")
        print("\nOr run with --oauth flag for interactive OAuth flow")
        return save_stats_to_csv(all_stats)

    try:
        # Open spreadsheet
        sheet = gc.open_by_key(SPREADSHEET_ID)

        # Try to get or create "TMux Monitor" worksheet
        try:
            worksheet = sheet.worksheet("TMux Monitor")
        except gspread.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title="TMux Monitor", rows=100, cols=15)

        # Prepare data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headers = [
            "Timestamp", "Session", "Benchmark", "Prefix", "Status",
            "Total Configs", "Running", "Success", "Failed", "Last Activity",
            "Models"
        ]

        rows = [headers]
        for session, stats in all_stats.items():
            row = [
                timestamp,
                session,
                stats.get("benchmark", "unknown"),
                stats.get("prefix", ""),
                stats.get("status", "unknown"),
                stats.get("total_configs", 0),
                len(stats.get("running_configs", [])),
                stats.get("success", 0),
                stats.get("failed", 0),
                stats.get("last_activity", ""),
                ", ".join(m.split("/")[-1] for m in list(stats.get("models", []))[:5])
            ]
            rows.append(row)

        # Clear and update
        worksheet.clear()
        worksheet.update(rows, 'A1')

        print(f"✅ Updated Google Sheet at {timestamp}")
        return True

    except Exception as e:
        print(f"❌ Failed to update sheet: {e}")
        print("   Falling back to local CSV...")
        return save_stats_to_csv(all_stats)


def get_available_sessions() -> List[str]:
    """Get list of available tmux sessions."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
    except Exception:
        pass
    return []


def setup_oauth():
    """Interactive setup for Google OAuth credentials."""
    print("\n🔐 Google Sheets OAuth Setup")
    print("=" * 50)
    print("\nTo update Google Sheets, you need to authenticate:")
    print("1. Go to: https://console.cloud.google.com/")
    print("2. Create a project (or select existing)")
    print("3. Enable Google Sheets API")
    print("4. Go to APIs & Services -> Credentials")
    print("5. Create OAuth 2.0 Client ID (Desktop app)")
    print("6. Download the JSON file")
    print(f"7. Save it as: ~/.config/gspread/credentials.json")
    print()

    creds_dir = Path.home() / ".config" / "gspread"
    creds_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gspread
        print("\nAttempting OAuth flow...")
        gc = gspread.oauth()
        print("✅ OAuth successful! Credentials saved.")
        return gc
    except Exception as e:
        print(f"❌ OAuth failed: {e}")
        print("\nAlternative: Use a service account")
        print("1. Create a service account in Google Cloud Console")
        print("2. Download JSON key file")
        print(f"3. Save as: ~/.config/gspread/service_account.json")
        print("4. Share your spreadsheet with the service account email")
        return None


def main():
    parser = argparse.ArgumentParser(description="Monitor tmux benchmark sessions")
    parser.add_argument("--interval", type=int, default=30,
                        help="Update interval in seconds (default: 30)")
    parser.add_argument("--sessions", type=str, default="long,long_2,long_3,long_4",
                        help="Comma-separated session names to monitor")
    parser.add_argument("--oauth", action="store_true",
                        help="Use OAuth for Google Sheets authentication")
    parser.add_argument("--setup-oauth", action="store_true",
                        help="Run OAuth setup wizard")
    parser.add_argument("--no-sheets", action="store_true",
                        help="Skip Google Sheets update (terminal only)")
    parser.add_argument("--csv", type=str, default="benchmark_status.csv",
                        help="CSV file path for local logging")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (don't watch)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover all available tmux sessions")
    parser.add_argument("--jsonbin", action="store_true",
                        help="Post to JSONBin.io (set JSONBIN_ID and JSONBIN_KEY env vars)")
    parser.add_argument("--html", type=str, default=None,
                        help="Generate live HTML dashboard at this path")
    parser.add_argument("--by-process", action="store_true",
                        help="Find benchmarks by running processes (bypasses tmux)")
    args = parser.parse_args()

    # Setup mode
    if args.setup_oauth:
        setup_oauth()
        return

    # Auto-discover or use specified sessions
    available = get_available_sessions()
    if args.auto_discover:
        sessions = available
    else:
        sessions = [s.strip() for s in args.sessions.split(",")]

    # Show available sessions
    print(f"Available tmux sessions: {', '.join(available) if available else 'none'}")
    print(f"Monitoring sessions: {', '.join(sessions)}")
    print(f"Update interval: {args.interval}s")
    print(f"Google Sheets: {'disabled' if args.no_sheets else 'enabled'}")
    print(f"CSV output: {args.csv}")
    print()

    if not sessions and not args.by_process:
        print("❌ No sessions to monitor. Start some tmux sessions first.")
        print("   Or use --by-process to find running benchmarks directly.")
        return

    try:
        iteration = 0
        while True:
            iteration += 1
            # Capture stats
            if args.by_process:
                all_stats = get_process_stats()
                if not all_stats:
                    print("⚠️ No benchmark processes found. Looking for run_benchmark_fixes.py...")
            else:
                all_stats = get_all_session_stats(sessions)

            if args.json:
                print(json.dumps(all_stats, indent=2))
            else:
                # Display in terminal
                print("\033[2J\033[H")  # Clear screen
                print(format_stats_for_display(all_stats))

            # Always save to CSV for history
            save_stats_to_csv(all_stats, args.csv)

            # Generate HTML dashboard if requested
            if args.html:
                generate_html_dashboard(all_stats, args.html)

            # Post to JSONBin if requested
            if args.jsonbin:
                post_to_jsonbin(all_stats)

            # Update Google Sheets
            if not args.no_sheets and not args.jsonbin:
                update_google_sheet(all_stats, use_oauth=args.oauth)

            if args.once:
                break

            # Wait for next interval
            print(f"\n⏳ Next update in {args.interval}s... (Ctrl+C to stop) [iteration {iteration}]")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n👋 Stopped monitoring.")
        print(f"📄 History saved to: {args.csv}")


if __name__ == "__main__":
    main()
