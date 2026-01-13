#!/usr/bin/env python3
"""
Aggregate rubric CSV outputs into a single file for easier analysis.

By default this script walks rubrics_output/**.csv, adds a `source_csv` column
pointing to the original file, and writes all rows into a merged CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

DEFAULT_COLUMNS = ["task_id", "criteria", "grade", "correct", "explanation", "model_run"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge rubric CSV files into a single CSV.")
    parser.add_argument(
        "--rubrics-root",
        default="rubrics_output",
        help="Directory containing rubric CSV outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="rubrics_output/merged_rubrics.csv",
        help="Destination CSV file (default: %(default)s).",
    )
    parser.add_argument(
        "--criteria",
        action="append",
        help="Limit to specific rubric criteria (repeatable). Uses CSV 'criteria' column.",
    )
    parser.add_argument(
        "--model-run-substring",
        help="Only include rows whose model_run contains this substring.",
    )
    return parser.parse_args()


def discover_csv_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def merge_csvs(
    csv_files: List[Path],
    criteria_filter: set[str] | None,
    model_run_substring: str | None,
) -> tuple[List[str], List[Dict[str, str]]]:
    all_rows: List[Dict[str, str]] = []
    observed_columns: List[str] = []

    for csv_path in csv_files:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue

            for row in reader:
                if not row:
                    continue

                if criteria_filter and row.get("criteria") not in criteria_filter:
                    continue
                if model_run_substring and model_run_substring not in (row.get("model_run") or ""):
                    continue

                row = dict(row)  # copy to avoid mutating csv module internals
                row["source_csv"] = str(csv_path)
                all_rows.append(row)

                for column in row.keys():
                    if column not in observed_columns:
                        observed_columns.append(column)

    if not all_rows:
        return observed_columns, all_rows

    ordered_columns: List[str] = []
    for column in DEFAULT_COLUMNS + ["source_csv"]:
        if column in observed_columns and column not in ordered_columns:
            ordered_columns.append(column)

    for column in observed_columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    return ordered_columns, all_rows


def write_merged_csv(output_path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rubrics_root = Path(args.rubrics_root)
    if not rubrics_root.is_dir():
        raise SystemExit(f"Rubrics root not found: {rubrics_root}")

    csv_files = discover_csv_files(rubrics_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {rubrics_root}")

    criteria_filter = set(args.criteria) if args.criteria else None
    columns, rows = merge_csvs(csv_files, criteria_filter, args.model_run_substring)

    if not rows:
        raise SystemExit("No rows matched the provided filters.")

    write_merged_csv(Path(args.output), columns, rows)
    print(f"Merged {len(rows)} rows from {len(csv_files)} files into {args.output}")


if __name__ == "__main__":
    main()
