#!/usr/bin/env python3
"""
Cleanup errors in RAW_SUBMISSIONS.jsonl files and optionally remove duplicate older runs.

Usage:
    python3 cleanup_submissions.py --timestamp 20260126_064943 --prefix sun12_ --keep-latest
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Matches YYYYMMDD_HHMMSS pattern in run IDs, potentially followed by other strings
TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})(?:_.*)?$")

def get_timestamp_from_run_id(run_id: str) -> Optional[str]:
    """Extracts timestamp string (YYYYMMDD_HHMMSS) from run_id."""
    match = TIMESTAMP_RE.search(run_id)
    if match:
        return match.group(1)
    return None

def get_base_id(run_id: str) -> str:
    """Returns the run ID without the timestamp and any following suffix."""
    match = TIMESTAMP_RE.search(run_id)
    if match:
        return run_id[:match.start()]
    return run_id

def list_results_roots(base_dir: Path) -> List[Path]:
    roots = []
    # Check common locations
    candidates = [
        base_dir / ".hal_data" / "results",
        base_dir / "results",
        base_dir / ".results",
    ]
    
    # Check env var
    if os.environ.get("HAL_RESULTS_DIR"):
        candidates.insert(0, Path(os.environ["HAL_RESULTS_DIR"]))

    seen_real_paths = set()
    for c in candidates:
        if c.exists() and c.is_dir():
            real_p = c.resolve()
            if real_p not in seen_real_paths:
                roots.append(real_p)
                seen_real_paths.add(real_p)
    
    return roots

def clean_file_errors(file_path: Path, dry_run: bool = False) -> int:
    """Removes error lines from RAW_SUBMISSIONS.jsonl. Returns count of removed lines."""
    if not file_path.exists():
        return 0
    
    kept_lines = []
    removed_count = 0
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Check for errors
                    is_error = False
                    if isinstance(data, dict):
                        # Some formats have task_id: result_obj, others have result_obj directly
                        # We look for "ERROR" in values
                        def check_for_error(obj):
                            if isinstance(obj, str) and (obj.startswith("ERROR") or "ERROR" in obj):
                                return True
                            if isinstance(obj, dict):
                                return any(check_for_error(v) for v in obj.values())
                            return False
                        
                        if check_for_error(data):
                            is_error = True
                    
                    if is_error:
                        removed_count += 1
                    else:
                        kept_lines.append(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
        
        if removed_count > 0:
            if not dry_run:
                with file_path.open("w", encoding="utf-8") as f:
                    for line in kept_lines:
                        f.write(line + "\n")
            print(f"  - Cleaned {removed_count} errors from {file_path.name}")
            
    except Exception as e:
        print(f"  ! Failed to process {file_path}: {e}")
        
    return removed_count

def main():
    parser = argparse.ArgumentParser(description="Cleanup RAW_SUBMISSIONS and manage run artifacts.")
    parser.add_argument("--timestamp", help="Filter runs starting from this timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument("--prefix", help="Filter runs containing this prefix string.")
    parser.add_argument("--keep-latest", action="store_true", help="Delete older duplicate runs, keeping only the latest.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files.")
    
    args = parser.parse_args()
    
    if not args.timestamp and not args.prefix:
        print("Warning: No timestamp or prefix provided. This might scan a lot of files.")
        confirm = input("Continue? [y/N] ")
        if confirm.lower() != 'y':
            sys.exit(0)

    start_ts = args.timestamp
    prefix = args.prefix
    
    base_dir = Path.cwd()
    roots = list_results_roots(base_dir)
    print(f"Scanning result roots: {[str(r) for r in roots]}")
    
    runs_by_base: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    total_runs_found = 0
    
    # Track seen run directories to avoid duplicates from symlinks etc.
    seen_run_dirs = set()

    for root in roots:
        # Pattern 1: root/benchmark/run_id
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            
            benchmark_dir = entry
            for subentry in benchmark_dir.iterdir():
                if subentry.is_dir():
                    real_run_dir = subentry.resolve()
                    if real_run_dir in seen_run_dirs:
                        continue
                    
                    run_id = real_run_dir.name
                    ts = get_timestamp_from_run_id(run_id)
                    if ts:
                        if prefix and prefix not in run_id:
                            continue
                        if start_ts and ts < start_ts:
                            continue
                        
                        base_id = get_base_id(run_id)
                        runs_by_base[base_id].append((ts, real_run_dir))
                        seen_run_dirs.add(real_run_dir)
                        total_runs_found += 1
            
            # Pattern 2: root/run_id
            real_entry_dir = entry.resolve()
            if real_entry_dir in seen_run_dirs:
                continue
            
            run_id = real_entry_dir.name
            ts = get_timestamp_from_run_id(run_id)
            if ts:
                if prefix and prefix not in run_id:
                    continue
                if start_ts and ts < start_ts:
                    continue
                
                base_id = get_base_id(run_id)
                runs_by_base[base_id].append((ts, real_entry_dir))
                seen_run_dirs.add(real_entry_dir)
                total_runs_found += 1

    print(f"Found {total_runs_found} matching runs.")    
    # Process Keep Latest
    dirs_to_remove = []
    dirs_to_process = []
    
    for base_id, runs in runs_by_base.items():
        # Sort by timestamp ascending
        runs.sort(key=lambda x: x[0])
        
        if args.keep_latest and len(runs) > 1:
            # All but last are candidates for removal
            for _, run_dir in runs[:-1]:
                dirs_to_remove.append(run_dir)
            # Last one is kept
            dirs_to_process.append(runs[-1][1])
        else:
            # Keep all
            for _, run_dir in runs:
                dirs_to_process.append(run_dir)

    # Execute removal
    if dirs_to_remove:
        print(f"Found {len(dirs_to_remove)} older runs to remove (--keep-latest).")
        for d in dirs_to_remove:
            print(f"  Delete: {d}")
            if not args.dry_run:
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"  ! Failed to delete {d}: {e}")

    # Execute cleanup
    print(f"Processing {len(dirs_to_process)} runs for error cleanup...")
    for run_dir in dirs_to_process:
        if not run_dir.exists():
            continue
            
        run_id = run_dir.name
        # Look for RAW_SUBMISSIONS.jsonl
        # Pattern: {run_id}_RAW_SUBMISSIONS.jsonl
        # Or sometimes just ending in _RAW_SUBMISSIONS.jsonl
        
        candidates = list(run_dir.glob("*_RAW_SUBMISSIONS.jsonl"))
        if not candidates:
            # Fallback search
            candidates = list(run_dir.glob("RAW_SUBMISSIONS.jsonl"))
            
        for fpath in candidates:
            clean_file_errors(fpath, args.dry_run)

    print("Done.")

if __name__ == "__main__":
    main()
