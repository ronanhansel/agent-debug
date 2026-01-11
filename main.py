import asyncio
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Iterable, Sequence

try:
    import dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dotenv = None

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from lunette.client import LunetteClient
    from lunette.models.messages import AssistantMessage, SystemMessage, UserMessage
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory


TRACE_DIR = Path("traces")
ORIGINAL_HOME = Path(os.environ.get("HOME", Path.home()))
_HOME_PREPARED: bool = False
RUBRIC_OUTPUT_PATH = Path("environmental_barrier_rubrics.csv")
SCAFFOLD_TAGS = {
    "<start_code>",
    "<end_code>",
    "<start_plan>",
    "<end_plan>",
    "<start_thought>",
    "<end_thought>",
    "<start_solution>",
    "<end_solution>",
}
ENVIRONMENTAL_BARRIER_DESCRIPTION = (
    "An Environmental Barrier in SWE-bench describes a failure mode where an agent is prevented from solving a task "
    "due to impassable infrastructure faults rather than a lack of coding capability. These barriers arise from defects "
    "in the evaluation setup itself—such as crashing Docker containers, broken shell environments, or pre-existing "
    "dependency conflicts—that render the codebase unrunnable regardless of the agent's actions. Unlike a capability "
    "failure where an agent writes incorrect code, an environmental barrier effectively blocks the agent from even "
    "attempting the task, often due to missing files or system-level restrictions that are outside the agent's control."
)

if dotenv:
    dotenv.load_dotenv()


@dataclass
class LunetteDependencies:
    """Runtime references to Lunette SDK classes."""

    client_cls: type
    system_message_cls: type
    user_message_cls: type
    assistant_message_cls: type
    run_cls: type
    trajectory_cls: type
    grading_plan_cls: type
    trajectory_filters_cls: type


@dataclass
class TraceMessage:
    """Normalized representation of a single message in a task trace."""

    role: str
    content: str
    entry_id: str | None
    timestamp: str | None


@dataclass
class TaskConversation:
    """Aggregated conversation for one SWE-bench task."""

    task_id: str
    entries: list[dict[str, Any]]
    messages: list[TraceMessage]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


def prepare_lunette_home() -> Path:
    """Ensure Lunette can write logs/config inside the workspace sandbox."""
    global _HOME_PREPARED
    if _HOME_PREPARED:
        return Path(os.environ["HOME"])

    workspace_home = (Path.cwd() / ".lunette_home").resolve()
    workspace_home.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(workspace_home)
    os.environ.setdefault("LUNETTE_HOME", str(workspace_home / ".lunette"))

    original_config = ORIGINAL_HOME / ".lunette" / "config.json"
    destination_config = workspace_home / ".lunette" / "config.json"
    if original_config.exists() and not destination_config.exists():
        destination_config.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(original_config, destination_config)
        except PermissionError:
            pass

    _HOME_PREPARED = True
    return workspace_home


def load_lunette_dependencies() -> LunetteDependencies:
    """Import Lunette SDK modules lazily after preparing the workspace home."""
    prepare_lunette_home()

    from lunette.client import LunetteClient
    from lunette.analysis import GradingPlan, TrajectoryFilters
    from lunette.models.messages import AssistantMessage, SystemMessage, UserMessage
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory

    return LunetteDependencies(
        client_cls=LunetteClient,
        system_message_cls=SystemMessage,
        user_message_cls=UserMessage,
        assistant_message_cls=AssistantMessage,
        run_cls=Run,
        trajectory_cls=Trajectory,
        grading_plan_cls=GradingPlan,
        trajectory_filters_cls=TrajectoryFilters,
    )


def load_trace_file(trace_dir: Path) -> dict[str, Any]:
    """Load the first JSON trace file inside trace_dir."""
    trace_files = sorted(trace_dir.glob("*.json"))
    if not trace_files:
        raise FileNotFoundError("No *.json trace files found in traces/")

    trace_file = trace_files[0]
    print(f"📂 Loading trace data from {trace_file.name} ...")

    with trace_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_overview(data: dict[str, Any]) -> None:
    """Print a concise overview of the dataset/config metadata."""
    config = data.get("config", {})
    results = data.get("results", {})

    agent_name = config.get("agent_name", "unknown-agent")
    model_name = config.get("agent_args", {}).get("model_name", "unknown-model")
    benchmark = config.get("benchmark_name", "unknown-benchmark")

    print("\n📊 Dataset Information:")
    print(f"   Agent: {agent_name}")
    print(f"   Model: {model_name}")
    print(f"   Benchmark: {benchmark}")

    accuracy = results.get("accuracy")
    total_cost = results.get("total_cost")
    failed_tasks = results.get("failed_tasks", [])
    successful_tasks = results.get("successful_tasks", [])

    print("\n📈 Results Summary:")
    if accuracy is not None:
        print(f"   Accuracy: {accuracy:.1%}")
    if total_cost is not None:
        print(f"   Total Cost: ${total_cost:.2f}")
    print(f"   Successful Tasks: {len(successful_tasks)}")
    print(f"   Failed Tasks: {len(failed_tasks)}")


def group_entries_by_task(raw_entries: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group raw logging entries by their weave_task_id."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for entry in raw_entries:
        task_id = (
            entry.get("attributes", {}).get("weave_task_id")
            or entry.get("weave_task_id")
            or entry.get("inputs", {}).get("task_id")
            or "unknown"
        )
        tasks.setdefault(task_id, []).append(entry)
    return tasks


def sanitize_text(text: str) -> str:
    """Strip agent scaffolding tags and trim whitespace."""
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if line.strip() in SCAFFOLD_TAGS:
            continue
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


def normalize_content(content: Any) -> str:
    """Convert OpenAI-style message content into a simple string."""
    if isinstance(content, str):
        return sanitize_text(content.strip())

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "reasoning":
                # reasoned content may include summaries we can surface
                summary = block.get("summary")
                if summary:
                    parts.append(summary)
                elif block.get("redacted") is False:
                    parts.append(block.get("reasoning", ""))
        raw_text = "\n".join(part for part in (p.strip() for p in parts) if part)
        return sanitize_text(raw_text)

    if content is None:
        return ""

    return sanitize_text(str(content).strip())


def entry_timestamp(entry: dict[str, Any]) -> str | None:
    """Best-effort timestamp for ordering entries."""
    return entry.get("created_timestamp") or entry.get("started_at") or entry.get("ended_at")


def sort_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort entries chronologically using available timestamps."""
    return sorted(entries, key=lambda e: (entry_timestamp(e) or "", e.get("id", "")))


def build_trace_message(raw_message: dict[str, Any], entry: dict[str, Any]) -> TraceMessage | None:
    """Normalize a raw OpenAI message dict into TraceMessage."""
    role = raw_message.get("role")
    if not role:
        return None

    content = normalize_content(raw_message.get("content"))
    if not content:
        return None

    return TraceMessage(role=role, content=content, entry_id=entry.get("id"), timestamp=entry_timestamp(entry))


def build_assistant_message(entry: dict[str, Any]) -> TraceMessage | None:
    """Extract the assistant response from the entry output."""
    output = entry.get("output") or {}
    choices = output.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message")
    if not message:
        return None

    role = message.get("role", "assistant")
    content = normalize_content(message.get("content"))
    if not content:
        return None

    return TraceMessage(role=role, content=content, entry_id=entry.get("id"), timestamp=entry_timestamp(entry))


def extract_task_messages(task_id: str, entries: list[dict[str, Any]]) -> TaskConversation:
    """Convert task-specific entries into an ordered conversation."""
    ordered_entries = sort_entries(entries)
    conversation: list[TraceMessage] = []
    previous_message_count = 0

    def append_assistant_output_if_new(entry: dict[str, Any]) -> None:
        assistant_message = build_assistant_message(entry)
        if not assistant_message:
            return

        if conversation:
            last_message = conversation[-1]
            if (
                last_message.role == assistant_message.role
                and last_message.content == assistant_message.content
            ):
                return

        conversation.append(assistant_message)

    for entry in ordered_entries:
        raw_messages = entry.get("inputs", {}).get("messages") or []

        if len(raw_messages) < previous_message_count:
            previous_message_count = 0

        if len(raw_messages) > previous_message_count:
            new_messages = raw_messages[previous_message_count:]
            for raw in new_messages:
                normalized = build_trace_message(raw, entry)
                if normalized:
                    if conversation:
                        last_message = conversation[-1]
                        if (
                            last_message.role == normalized.role
                            and last_message.content == normalized.content
                        ):
                            continue
                    conversation.append(normalized)

        previous_message_count = len(raw_messages)

        append_assistant_output_if_new(entry)

    return TaskConversation(task_id=task_id, entries=ordered_entries, messages=conversation)


def convert_to_lunette_messages(
    messages: Sequence[TraceMessage],
    system_message_cls: type,
    user_message_cls: type,
    assistant_message_cls: type,
) -> list[Any]:
    """Map TraceMessages to Lunette message models."""
    converted: list[Any] = []

    for position, message in enumerate(messages):
        content = message.content or ""
        if message.role == "system":
            converted.append(system_message_cls(position=position, content=content))
        elif message.role == "assistant":
            converted.append(assistant_message_cls(position=position, content=content))
        else:
            converted.append(user_message_cls(position=position, content=content))

    return converted


def create_environmental_barrier_plan(
    deps: LunetteDependencies,
    sample_filter: Sequence[str] | None = None,
) -> Any:
    """Build a grading plan focused on environmental barrier classification."""
    filters = deps.trajectory_filters_cls()
    if sample_filter:
        filters.sample = list(sample_filter)

    prompt = dedent(
        f"""
        You are evaluating whether a SWE-bench trajectory failed because of an environmental barrier.
        Definition: {ENVIRONMENTAL_BARRIER_DESCRIPTION}

        Scoring instructions:
        - Return score 1.0 when the failure is caused by infrastructure or environment issues that prevent progress
          (e.g., sandbox crashes, missing files that predate the agent, dependency conflicts, permission errors, or
          other system-level blockers outside the agent's control).
        - Return score 0.0 when the agent's reasoning, planning, or code is at fault, or when evidence of an
          environmental barrier is insufficient.

        Requirements:
        - Always set `name` to "environmentalbarrier".
        - Provide a concise explanation citing the specific log lines or messages that justify the score.
        - If the evidence is unclear, default to score 0.0 and explain why the barrier is unproven.
        """
    ).strip()

    return deps.grading_plan_cls(
        name="environmentalbarrier",
        prompt=prompt,
        trajectory_filters=filters,
    )


def preview_task(conversation: TaskConversation, limit: int = 8) -> None:
    """Print a human-readable preview for one task."""
    print(f"\n🔍 Previewing task {conversation.task_id}")
    print(f"   Entries: {conversation.entry_count}")
    print(f"   Messages extracted: {len(conversation.messages)}")

    if not conversation.messages:
        print("   (No messages extracted)")
        return

    for message in conversation.messages[:limit]:
        snippet = message.content.replace("\n", " ")[:160]
        print(f"   [{message.role}] {snippet}")

    remaining = len(conversation.messages) - limit
    if remaining > 0:
        print(f"   ... ({remaining} more messages)")


def confirm(prompt: str) -> bool:
    """Prompt the user before uploading."""
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def print_grading_results(results: Any) -> None:
    """Pretty-print grading results returned by Lunette."""
    trajectory_results = getattr(results, "results", None)
    if not trajectory_results:
        print("⚠️  Grading completed but returned no trajectory results.")
        return

    print("\n🧪 Environmental Barrier Grades:")
    for item in trajectory_results:
        data = item.data or {}
        name = data.get("name", "environmentalbarrier")
        score = data.get("score", "N/A")
        explanation = (data.get("explanation") or "").strip()
        sample_key = item.result_key or str(item.original_trajectory_id)

        print(f"   • Sample {sample_key}: {name} = {score}")
        if explanation:
            print(f"      Explanation: {explanation}")


def write_cloud_grading_csv(
    grading_results: Any,
    model_run: str,
    trajectory_id_to_sample: dict[str, str] | None = None,
    output_path: Path = RUBRIC_OUTPUT_PATH,
    default_criteria: str = "environmentalbarrier",
) -> Path | None:
    """Persist cloud grading results to CSV."""
    trajectory_results = getattr(grading_results, "results", None)
    if not trajectory_results:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "criteria", "grade", "explanation", "model_run"])
        for item in trajectory_results:
            data = item.data or {}
            sample_id = None
            if trajectory_id_to_sample:
                sample_id = trajectory_id_to_sample.get(str(item.original_trajectory_id))
            sample_id = sample_id or item.result_key or str(item.original_trajectory_id)
            writer.writerow(
                [
                    sample_id,
                    data.get("name") or default_criteria,
                    data.get("score", ""),
                    (data.get("explanation") or "").strip(),
                    model_run,
                ]
            )
    return output_path


def build_trajectories(
    conversations: Sequence[TaskConversation],
    failed_tasks: set[str],
    agent_name: str,
    deps: LunetteDependencies,
) -> list[Any]:
    """Create Lunette trajectories from task conversations."""
    trajectories: list[Any] = []
    for conversation in conversations:
        if not conversation.messages:
            continue

        lunette_messages = convert_to_lunette_messages(
            conversation.messages,
            deps.system_message_cls,
            deps.user_message_cls,
            deps.assistant_message_cls,
        )
        metadata = {
            "failed": conversation.task_id in failed_tasks,
            "task_id": conversation.task_id,
            "entry_count": conversation.entry_count,
            "message_count": len(lunette_messages),
            "agent": agent_name,
        }

        trajectory = deps.trajectory_cls(
            sample=conversation.task_id,
            messages=lunette_messages,
            metadata=metadata,
        )
        trajectories.append(trajectory)

    return trajectories


async def upload_and_optionally_grade(
    deps: LunetteDependencies,
    run: Any,
    grading_plan: Any | None,
) -> tuple[dict[str, Any], Any | None]:
    """Upload the run and optionally launch grading."""
    async with deps.client_cls() as client:
        upload_result = await client.save_run(run)

        investigation_results = None
        run_id = upload_result.get("run_id")
        if grading_plan and run_id:
            try:
                print("\n🧪 Running environmental barrier grading...")
                investigation_results = await client.investigate(
                    run_id=run_id,
                    plan=grading_plan,
                )
            except Exception as exc:  # pragma: no cover - depends on remote service
                print(f"⚠️  Grading failed: {exc}")

        return upload_result, investigation_results


def main() -> None:
    try:
        data = load_trace_file(TRACE_DIR)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return

    dataset_overview(data)

    raw_entries = data.get("raw_logging_results") or []
    if not raw_entries:
        print("❌ No raw logging results found in trace file.")
        return

    tasks = group_entries_by_task(raw_entries)
    print(f"\n🧵 Found {len(tasks)} unique tasks")

    conversations: list[TaskConversation] = []
    for task_id, entries in tasks.items():
        conversation = extract_task_messages(task_id, entries)
        if conversation.messages:
            conversations.append(conversation)

    conversations.sort(key=lambda conv: conv.task_id)

    if not conversations:
        print("❌ Failed to extract any conversations.")
        return

    preview_task(conversations[0])

    failed_tasks = set(data.get("results", {}).get("failed_tasks", []))
    agent_name = data.get("config", {}).get("agent_name", "unknown-agent")
    model_name = data.get("config", {}).get("agent_args", {}).get("model_name", "unknown-model")
    benchmark = data.get("config", {}).get("benchmark_name", "unknown-benchmark")
    model_run_id = data.get("config", {}).get("run_id") or data.get("config", {}).get("date") or "local-run"

    if not confirm("\nProceed with uploading all tasks to Lunette?"):
        print("ℹ️  Upload canceled. Inspect the preview above and rerun when ready.")
        return

    if not os.getenv("LUNETTE_API_KEY"):
        print("❌ LUNETTE_API_KEY not set. Export the key or add it to a .env file before uploading.")
        return

    try:
        deps = load_lunette_dependencies()
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"❌ Unable to import Lunette SDK: {exc}")
        print("   Ensure lunette-sdk is installed inside the virtual environment.")
        return

    trajectories = build_trajectories(conversations, failed_tasks, agent_name, deps)
    if not trajectories:
        print("❌ No trajectories with messages were produced.")
        return
    trajectory_samples = [str(traj.sample) for traj in trajectories]

    grading_plan = None
    if confirm("\nRun Lunette environmental-barrier grading after upload?"):
        grading_plan = create_environmental_barrier_plan(deps)

    print(f"\n☁️  Uploading {len(trajectories)} trajectories to Lunette...")
    run = deps.run_cls(task=benchmark, model=model_name, trajectories=trajectories)
    upload_result, grading_results = asyncio.run(
        upload_and_optionally_grade(deps, run, grading_plan),
    )

    run_id = upload_result.get("run_id")
    traj_ids = upload_result.get("trajectory_ids", [])
    trajectory_id_to_sample: dict[str, str] = {}
    if traj_ids and len(traj_ids) == len(trajectory_samples):
        trajectory_id_to_sample = {
            str(traj_id): trajectory_samples[idx] for idx, traj_id in enumerate(traj_ids)
        }
    elif traj_ids:
        print(
            "⚠️  Warning: trajectory ID count does not match local trajectories; CSV may lack task IDs."
        )

    if run_id:
        print(f"\n✅ Upload complete! Run ID: {run_id}")
        print(f"   Trajectories uploaded: {len(traj_ids)}")
        print("   Visit https://lunette.dev/ to inspect the traces.")
    else:
        print("⚠️  Upload finished but no run_id was returned. Please verify on Lunette.")

    if grading_results:
        print_grading_results(grading_results)
        csv_path = write_cloud_grading_csv(
            grading_results,
            model_run_id,
            trajectory_id_to_sample=trajectory_id_to_sample,
        )
        if csv_path:
            print(f"\n🗂️  Cloud grading CSV written to {csv_path}")


if __name__ == "__main__":
    main()
