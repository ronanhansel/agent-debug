#!/usr/bin/env python3
"""
Item Fixer - Analyzes failed traces and creates item-level fixes.

This script analyzes WHY an agent failed and creates appropriate fixes
that address the root cause WITHOUT lowering benchmark difficulty.

Fix Types:
1. env_override.json - Add missing packages (e.g., matplotlib, specific R packages)
2. input_override.json - Clarify unclear instructions (without giving answers)

What This Does NOT Do:
- Tell agents "don't install packages" (they CAN install now)
- Give away answers or solution hints
- Lower the computational difficulty of the task
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_ROOT = REPO_ROOT / "fixes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze failed traces and create item-level fixes."
    )
    parser.add_argument(
        "--trace-file",
        required=True,
        help="Path to the trace file (verbose log or JSON).",
    )
    parser.add_argument(
        "--benchmark",
        default="corebench_hard",
        help="Benchmark name for fix folder structure.",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID (auto-detected from trace file name if not provided).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print analysis without creating files.",
    )
    return parser.parse_args()


class TraceAnalyzer:
    """Analyzes a trace to identify why an agent failed."""

    # Known missing package patterns
    MISSING_PACKAGE_PATTERNS = [
        # Python packages
        (r"ModuleNotFoundError: No module named ['\"](\w+)['\"]", "python"),
        (r"ImportError: No module named ['\"](\w+)['\"]", "python"),
        (r"Import of (\w+) is not allowed", "python_sandbox"),
        # R packages
        (r"there is no package called ['\"]?(\w+)['\"]?", "r"),
        (r"Error in library\((\w+)\)", "r"),
        (r"Package ['\"]?(\w+)['\"]? required but not available", "r"),
    ]

    # Patterns indicating empty/failed output
    EMPTY_OUTPUT_PATTERNS = [
        r"Not found in results",
        r"contains no content/results",
        r"no substantive data or results",
        r"file is either empty",
    ]

    # Patterns indicating unclear instructions
    UNCLEAR_INSTRUCTION_PATTERNS = [
        r"cannot determine",
        r"unclear what",
        r"ambiguous",
        r"not specified",
    ]

    def __init__(self, trace_content: str, task_id: str):
        self.trace_content = trace_content
        self.task_id = task_id
        self.issues: List[Dict[str, Any]] = []

    def analyze(self) -> Dict[str, Any]:
        """Run all analysis checks and return findings."""
        self._check_missing_packages()
        self._check_empty_output()
        self._check_unclear_instructions()
        self._check_execution_errors()

        return {
            "task_id": self.task_id,
            "issues": self.issues,
            "summary": self._generate_summary(),
            "recommended_fixes": self._recommend_fixes(),
        }

    def _check_missing_packages(self) -> None:
        """Check for missing package errors."""
        for pattern, pkg_type in self.MISSING_PACKAGE_PATTERNS:
            matches = re.findall(pattern, self.trace_content, re.IGNORECASE)
            for match in matches:
                pkg_name = match if isinstance(match, str) else match[0]
                self.issues.append({
                    "type": "missing_package",
                    "package_type": pkg_type,
                    "package_name": pkg_name,
                    "severity": "high",
                    "fixable": True,
                })

    def _check_empty_output(self) -> None:
        """Check for empty/failed output patterns."""
        for pattern in self.EMPTY_OUTPUT_PATTERNS:
            if re.search(pattern, self.trace_content, re.IGNORECASE):
                self.issues.append({
                    "type": "empty_output",
                    "pattern": pattern,
                    "severity": "medium",
                    "fixable": False,  # Needs manual investigation
                })
                break

    def _check_unclear_instructions(self) -> None:
        """Check for signs of unclear instructions."""
        for pattern in self.UNCLEAR_INSTRUCTION_PATTERNS:
            if re.search(pattern, self.trace_content, re.IGNORECASE):
                self.issues.append({
                    "type": "unclear_instructions",
                    "pattern": pattern,
                    "severity": "low",
                    "fixable": True,
                })

    def _check_execution_errors(self) -> None:
        """Check for execution errors."""
        # Check for Python errors
        if "Traceback (most recent call last)" in self.trace_content:
            # Extract the error type
            error_match = re.search(
                r"(\w+Error): (.+?)(?:\n|$)",
                self.trace_content
            )
            if error_match:
                self.issues.append({
                    "type": "execution_error",
                    "error_class": error_match.group(1),
                    "error_message": error_match.group(2)[:200],
                    "severity": "high",
                    "fixable": False,
                })

    def _generate_summary(self) -> str:
        """Generate a human-readable summary."""
        if not self.issues:
            return "No obvious issues detected. Manual investigation needed."

        issue_types = set(i["type"] for i in self.issues)
        parts = []

        if "missing_package" in issue_types:
            pkgs = [i for i in self.issues if i["type"] == "missing_package"]
            pkg_names = [f"{i['package_name']} ({i['package_type']})" for i in pkgs]
            parts.append(f"Missing packages: {', '.join(pkg_names)}")

        if "empty_output" in issue_types:
            parts.append("Output was empty or missing content")

        if "execution_error" in issue_types:
            errors = [i for i in self.issues if i["type"] == "execution_error"]
            error_types = [i["error_class"] for i in errors]
            parts.append(f"Execution errors: {', '.join(error_types)}")

        return "; ".join(parts)

    def _recommend_fixes(self) -> Dict[str, Any]:
        """Recommend fixes based on issues found."""
        fixes = {
            "env_override": {},
            "input_override": None,
            "needs_investigation": [],
        }

        # Collect missing packages
        python_packages = []
        r_packages = []

        for issue in self.issues:
            if issue["type"] == "missing_package":
                if issue["package_type"] == "python":
                    python_packages.append(issue["package_name"])
                elif issue["package_type"] == "r":
                    r_packages.append(f"r-{issue['package_name'].lower()}")
                elif issue["package_type"] == "python_sandbox":
                    # These can't be fixed via env_override
                    fixes["needs_investigation"].append(
                        f"Sandboxed import blocked: {issue['package_name']}"
                    )

            elif issue["type"] == "empty_output":
                fixes["needs_investigation"].append(
                    "Notebook/script produced empty output - investigate why"
                )

            elif issue["type"] == "execution_error":
                fixes["needs_investigation"].append(
                    f"Execution error: {issue['error_class']}"
                )

        # Build env_override
        if python_packages or r_packages:
            conda_packages = []
            if r_packages:
                conda_packages.extend(r_packages)
            # Note: Python packages can be installed via pip by the agent
            # but we can add them to conda if they're available

            if conda_packages:
                fixes["env_override"] = {
                    "HAL_CONDA_CHANNELS": "conda-forge",
                    "HAL_CONDA_PACKAGES": " ".join(conda_packages),
                }

            if python_packages:
                fixes["needs_investigation"].append(
                    f"Python packages needed (agent should pip install): {', '.join(python_packages)}"
                )

        return fixes


def extract_task_id_from_filename(filepath: Path) -> Optional[str]:
    """Extract task ID from trace filename."""
    # Pattern: ..._capsule-XXXXXXX_...
    match = re.search(r"(capsule-\d+)", filepath.name)
    if match:
        return match.group(1)
    return None


def create_fix_package(
    task_id: str,
    benchmark: str,
    analysis: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Create fix package based on analysis."""
    fix_dir = FIXES_ROOT / benchmark / task_id
    recommended = analysis["recommended_fixes"]

    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")
    print(f"\nSUMMARY: {analysis['summary']}")

    # Show issues
    if analysis["issues"]:
        print(f"\nISSUES FOUND ({len(analysis['issues'])}):")
        for i, issue in enumerate(analysis["issues"], 1):
            print(f"  {i}. [{issue['type']}] {issue.get('package_name', issue.get('pattern', 'N/A'))}")
            print(f"     Severity: {issue['severity']}, Fixable: {issue['fixable']}")

    # Show recommended fixes
    print("\nRECOMMENDED FIXES:")

    if recommended["env_override"]:
        print("\n  env_override.json:")
        print(f"    {json.dumps(recommended['env_override'], indent=4)}")

    if recommended["input_override"]:
        print("\n  input_override.json:")
        print(f"    {json.dumps(recommended['input_override'], indent=4)}")

    if recommended["needs_investigation"]:
        print("\n  NEEDS MANUAL INVESTIGATION:")
        for item in recommended["needs_investigation"]:
            print(f"    - {item}")

    if dry_run:
        print("\n[DRY RUN] No files created.")
        return

    # Create fix directory
    fix_dir.mkdir(parents=True, exist_ok=True)

    # Write env_override.json if needed
    if recommended["env_override"]:
        env_path = fix_dir / "env_override.json"
        # Merge with existing if present
        existing = {}
        if env_path.exists():
            existing = json.loads(env_path.read_text())

        # Merge conda packages
        if "HAL_CONDA_PACKAGES" in recommended["env_override"]:
            existing_pkgs = existing.get("HAL_CONDA_PACKAGES", "").split()
            new_pkgs = recommended["env_override"]["HAL_CONDA_PACKAGES"].split()
            all_pkgs = list(set(existing_pkgs + new_pkgs))
            recommended["env_override"]["HAL_CONDA_PACKAGES"] = " ".join(sorted(all_pkgs))

        merged = {**existing, **recommended["env_override"]}
        env_path.write_text(json.dumps(merged, indent=2))
        print(f"\n[CREATED] {env_path}")

    # Write investigation notes
    if recommended["needs_investigation"]:
        notes_path = fix_dir / "investigation_notes.txt"
        notes = [
            f"# Investigation Notes for {task_id}",
            f"# Generated by item_fixer.py",
            "",
            "## Issues Requiring Manual Investigation:",
            "",
        ]
        for item in recommended["needs_investigation"]:
            notes.append(f"- {item}")
        notes.append("")
        notes.append("## Analysis Summary:")
        notes.append(analysis["summary"])
        notes_path.write_text("\n".join(notes))
        print(f"[CREATED] {notes_path}")


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_file)

    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}")
        sys.exit(1)

    # Determine task ID
    task_id = args.task_id or extract_task_id_from_filename(trace_path)
    if not task_id:
        print("Error: Could not determine task ID. Use --task-id to specify.")
        sys.exit(1)

    # Read trace content
    trace_content = trace_path.read_text(encoding="utf-8", errors="replace")

    # Analyze
    analyzer = TraceAnalyzer(trace_content, task_id)
    analysis = analyzer.analyze()

    # Create fixes
    create_fix_package(
        task_id=task_id,
        benchmark=args.benchmark,
        analysis=analysis,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
