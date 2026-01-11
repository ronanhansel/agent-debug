import argparse
import asyncio
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lunette.analysis import IssueDetectionPlan, TrajectoryFilters
from lunette.client import LunetteClient
from dotenv import load_dotenv

load_dotenv()


ENVIRONMENTAL_BARRIER_DESCRIPTION = (
    "An Environmental Barrier in SWE-bench describes a failure mode where an agent is prevented from solving a task "
    "due to impassable infrastructure faults rather than a lack of coding capability. These barriers arise from defects "
    "in the evaluation setup itself—such as crashing Docker containers, broken shell environments, or pre-existing "
    "dependency conflicts—that render the codebase unrunnable regardless of the agent's actions. Unlike a capability "
    "failure where an agent writes incorrect code, an environmental barrier effectively blocks the agent from even "
    "attempting the task, often due to missing files or system-level restrictions that are outside the agent's control."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect graded trajectories and launch Lunette investigations.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/environmental_barrier_rubrics.csv"),
        help="Path to the rubric CSV produced by the grader.",
    )
    parser.add_argument(
        "--grade-threshold",
        type=float,
        default=1.0,
        help="Only investigate rows whose grade is >= this threshold.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to investigate (per run).",
    )
    return parser.parse_args()


def load_candidates(csv_path: Path, min_grade: float) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Rubric CSV not found: {csv_path}")

    candidates: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                grade = float(row.get("grade", "0") or 0)
            except ValueError:
                continue
            if grade >= min_grade:
                candidates.append(row)
    return candidates


def group_by_run(rows: Iterable[dict[str, str]], max_samples: int | None) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        run_id = row.get("model_run") or row.get("run_id")
        task_id = row.get("task_id")
        if not run_id or not task_id:
            continue
        if max_samples and len(grouped[run_id]) >= max_samples:
            continue
        grouped[run_id].append(task_id)
    return grouped


def build_plan(task_ids: list[str]) -> IssueDetectionPlan:
    return IssueDetectionPlan(
        name="environmental-barrier-root-cause",
        prompt=(
            "You are investigating SWE-bench trajectories that likely encountered environmental barriers.\n"
            f"Definition: {ENVIRONMENTAL_BARRIER_DESCRIPTION}\n"
            "For each trajectory, determine whether the failure is caused by environment/infrastructure or agent behavior. "
            "Highlight concrete evidence (log lines, errors) and propose remediation steps."
        ),
        trajectory_filters=TrajectoryFilters(sample=task_ids),
    )


def summarize_results(run_id: str, results: Any) -> None:
    trajectory_results = getattr(results, "results", [])
    if not trajectory_results:
        print(f"⚠️  Investigation for run {run_id} returned no results.")
        return

    print(f"\n🔎 Investigation results for run {run_id}:")
    for item in trajectory_results:
        sample_id = item.result_key or str(item.original_trajectory_id)
        data = item.data or {}
        name = data.get("name", "issue")
        role = data.get("role", "unknown")
        description = data.get("description", "").strip()
        proof = data.get("proof", "").strip()
        confidence = data.get("confidence")
        print(f"   • Sample {sample_id} — {name} ({role})")
        if confidence is not None:
            print(f"      Confidence: {confidence:.2f}")
        if description:
            print(f"      Description: {description}")
        if proof:
            print(f"      Evidence: {proof}")

    recommendations = build_recommendations(trajectory_results)
    if recommendations:
        print("\n🛠️  Proposed setup/question fixes:")
        for rec in recommendations:
            print(f"   - {rec}")


def build_recommendations(results: Iterable[Any]) -> list[str]:
    recs: list[str] = []
    for item in results:
        data = item.data or {}
        role = data.get("role")
        description = data.get("description", "")
        sample_id = item.result_key or str(item.original_trajectory_id)
        if role == "environment":
            recs.append(
                f"Sample {sample_id}: address environment issue – {description or 'review sandbox logs for blockers.'}"
            )
        elif role == "agent":
            recs.append(
                f"Sample {sample_id}: improve question/setup to clarify agent expectations – {description}"
            )
        else:
            recs.append(
                f"Sample {sample_id}: investigate root cause further – {description or 'insufficient detail provided.'}"
            )
    return recs


async def investigate_candidates(grouped: dict[str, list[str]], max_samples: int | None) -> None:
    if not grouped:
        print("No samples meet the grade threshold.")
        return

    async with LunetteClient() as client:
        for run_id, samples in grouped.items():
            plan = build_plan(samples)
            print(f"\n🚀 Launching investigation for run {run_id} (samples: {len(samples)})...")
            limit = min(len(samples), max_samples or len(samples))
            results = await client.investigate(run_id=run_id, plan=plan, limit=limit)
            summarize_results(run_id, results)


def main() -> None:
    args = parse_args()
    try:
        rows = load_candidates(args.csv, args.grade_threshold)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return

    grouped = group_by_run(rows, args.max_samples)
    asyncio.run(investigate_candidates(grouped, args.max_samples))


if __name__ == "__main__":
    main()
