import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Sequence

try:
    import dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dotenv = None

TRACE_DIR = Path("traces")
RUBRICS_DIR = Path("rubrics")
RUBRIC_OUTPUT_DIR = Path("rubrics_output")
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


def _ensure_llm_cache_path() -> None:
    """Docent expects LLM_CACHE_PATH; default to .llm_cache if not provided."""
    if os.getenv("LLM_CACHE_PATH"):
        return
    default_cache = Path(".llm_cache")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CACHE_PATH"] = str(default_cache.resolve())


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


os.environ.setdefault("ENV_RESOLUTION_STRATEGY", "os_environ")
_ensure_llm_cache_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docent rubric extraction and grading utility.")
    parser.add_argument(
        "--trace-file",
        type=str,
        help="Path to a single trace JSON file. If omitted, the newest *.json in --trace-dir is used.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=str(TRACE_DIR),
        help="Directory to scan for *.json trace files when --trace-file is not provided.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks evaluated after grouping (useful for smoke tests).",
    )
    parser.add_argument(
        "--rubric-model",
        type=str,
        default=None,
        help=(
            "Override the rubric model in the format provider:model_name (e.g., azure_openai:o3-mini). "
            "This takes precedence over DOCENT_RUBRIC_MODEL and related env vars."
        ),
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help=(
            "Request JSON-mode completions (response_format=json_object). Only supported for OpenAI/Azure deployments "
            "that enable structured output."
        ),
    )
    parser.add_argument(
        "--rubrics-dir",
        type=str,
        default=str(RUBRICS_DIR),
        help="Directory containing *.txt rubric definitions (optional matching .schema.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RUBRIC_OUTPUT_DIR),
        help="Directory to write rubric CSV files when output-mode=csv (default: rubrics_output/).",
    )
    parser.add_argument(
        "--output-mode",
        choices=["csv", "stdout"],
        default="csv",
        help="Choose 'csv' to write rubric outputs to disk or 'stdout' to print results without writing files.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Override the model's reasoning_effort parameter (OpenAI/Azure reasoning models only).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before running rubric evaluation.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only evaluate tasks that appear in the trace's failed_tasks list.",
    )
    return parser.parse_args()


DOCENT_REPO_PATH = Path(__file__).parent / "docent"
if DOCENT_REPO_PATH.exists():
    sys.path.insert(0, str(DOCENT_REPO_PATH))

DOCENT_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message
    from docent_core.docent.ai_tools.rubric.rubric import Rubric, evaluate_rubric
    from docent_core._llm_util.providers.preferences import ModelOption
except Exception as exc:  # pragma: no cover - optional dependency
    DOCENT_IMPORT_ERROR = exc
    AgentRun = None  # type: ignore[assignment]
    Transcript = None  # type: ignore[assignment]
    parse_chat_message = None  # type: ignore[assignment]
    Rubric = None  # type: ignore[assignment]
    ModelOption = None  # type: ignore[assignment]
    evaluate_rubric = None  # type: ignore[assignment]

GLOBAL_JSON_REQUIREMENTS = dedent(
    """
    JSON response requirements:
    - Respond with only the JSON object that matches the schema (no prose, code fences, or leading text).
    - Escape newline characters as \\n; do not emit raw control characters inside JSON strings.
    """
).strip()

DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "enum": [0, 1],
            "description": "Binary score: use 1 only when the rubric is satisfied, else 0.",
        },
        "explanation": {
            "type": "string",
            "citations": True,
            "description": "Concise justification referencing transcript blocks.",
        },
    },
    "required": ["score", "explanation"],
}

DEFAULT_RUBRIC_TEXT = dedent(
    f"""
    Rubric: Environmental Barrier

    Detection target:
    {ENVIRONMENTAL_BARRIER_DESCRIPTION}

    Label the rubric as "present" when the transcript contains clear evidence that the agent was blocked by these
    infrastructural faults. Label it as "absent" when failures stem from the agent's own reasoning, planning, or code,
    or when no convincing evidence of an environmental issue appears.

    Evidence guidelines:
    - Presence indicators (label as present):
      • Sandbox or container crashes, unrecoverable kernel panics, or persistent harness errors unrelated to the agent's code.
      • Missing files, permissions, or dependencies that existed before the agent took any action.
      • System-level restrictions (e.g., read-only filesystems, broken shell environments) that halt progress for any agent.
    - Absence indicators (label as absent):
      • Errors caused by malformed patches, wrong file edits, or logical mistakes.
      • Tool or command failures that follow from the agent's own incorrect inputs.
      • Speculative or insufficient evidence; if uncertain, default to absent.

    Explanation requirements:
    - Reference the specific transcript blocks or tool outputs that justify the classification.
    - Highlight both the failure symptoms and why they originate from the environment (or why they do not).
    - Keep explanations concise and cite block IDs directly.

    {GLOBAL_JSON_REQUIREMENTS}
    """
).strip()

DEFAULT_RUBRIC_PROVIDER = os.getenv("DOCENT_RUBRIC_PROVIDER", "azure_openai")
DEFAULT_RUBRIC_BATCH_SIZE = int(os.getenv("DOCENT_RUBRIC_BATCH_SIZE", "4"))

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


@dataclass
class LocalRubricEvaluation:
    """Container for rubric results produced by Docent."""

    task_id: str
    rubric_id: str
    rubric_version: int
    output: dict[str, Any] | None
    error: str | None = None

    @property
    def score(self) -> float | None:
        if not self.output:
            return None
        score = self.output.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    @property
    def explanation(self) -> str:
        if not self.output:
            return ""
        explanation = self.output.get("explanation")
        return explanation if isinstance(explanation, str) else ""


@dataclass
class RubricDefinition:
    rubric_id: str
    rubric_text: str
    output_schema: dict[str, Any]


def ensure_default_rubric_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    txt_files = list(directory.glob("*.txt"))
    if txt_files:
        return

    default_path = directory / "environmentalbarrier.txt"
    default_path.write_text(DEFAULT_RUBRIC_TEXT, encoding="utf-8")
    schema_path = default_path.with_suffix(".schema.json")
    schema_path.write_text(json.dumps(DEFAULT_OUTPUT_SCHEMA, indent=2), encoding="utf-8")


def load_rubric_definitions(directory: Path) -> list[RubricDefinition]:
    ensure_default_rubric_files(directory)
    definitions: list[RubricDefinition] = []
    for txt_path in sorted(directory.glob("*.txt")):
        rubric_text = txt_path.read_text(encoding="utf-8").strip()
        if not rubric_text:
            continue
        if GLOBAL_JSON_REQUIREMENTS not in rubric_text:
            rubric_text = f"{rubric_text.rstrip()}\n\n{GLOBAL_JSON_REQUIREMENTS}"
        schema_path = txt_path.with_suffix(".schema.json")
        output_schema = DEFAULT_OUTPUT_SCHEMA
        if schema_path.exists():
            try:
                output_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"⚠️  Failed to parse schema for {txt_path.name}: {exc}. Using default schema.")
        rubric_id = _slugify(txt_path.stem.lower())
        definitions.append(
            RubricDefinition(
                rubric_id=rubric_id,
                rubric_text=rubric_text,
                output_schema=output_schema,
            )
        )
    return definitions


def resolve_trace_path(trace_file: str | None, trace_dir: Path) -> Path:
    if trace_file:
        candidate = Path(trace_file).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Trace file not found: {candidate}")
        return candidate

    trace_files = sorted(trace_dir.expanduser().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trace_files:
        raise FileNotFoundError(f"No *.json trace files found in {trace_dir}")
    return trace_files[0]


def load_trace_file(trace_path: Path) -> dict[str, Any]:
    """Load the selected JSON trace file."""
    print(f"📂 Loading trace data from {trace_path} ...")
    with trace_path.open("r", encoding="utf-8") as handle:
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


def resolve_rubric_model_option(
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> "ModelOption":
    """Derive the Docent ModelOption to use for rubric evaluation."""
    if ModelOption is None:
        raise RuntimeError("Docent ModelOption class is unavailable. Ensure docent is installed.")

    raw_model = model_override or os.getenv("DOCENT_RUBRIC_MODEL")
    provider = DEFAULT_RUBRIC_PROVIDER
    model_name = None

    if raw_model:
        if ":" in raw_model:
            provider, model_name = raw_model.split(":", 1)
        else:
            model_name = raw_model

    if not model_name:
        model_name = os.getenv("DOCENT_RUBRIC_MODEL_NAME")

    if not model_name and provider == "azure_openai":
        for candidate in (
            "AZURE_OPENAI_RUBRIC_MODEL",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
        ):
            value = os.getenv(candidate)
            if value:
                model_name = value
                break

    if not model_name:
        raise ValueError(
            "Unable to determine an Azure OpenAI deployment for rubric evaluation. "
            "Set DOCENT_RUBRIC_MODEL (provider:model) or DOCENT_RUBRIC_MODEL_NAME / "
            "AZURE_OPENAI_DEPLOYMENT_NAME in your environment."
        )

    reasoning_effort = os.getenv("DOCENT_RUBRIC_REASONING_EFFORT")
    if reasoning_effort:
        reasoning_effort = reasoning_effort.lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "DOCENT_RUBRIC_REASONING_EFFORT must be one of: low, medium, high."
            )

    return ModelOption(
        provider=provider,
        model_name=model_name,
        reasoning_effort=(reasoning_effort_override or reasoning_effort),  # type: ignore[arg-type]
    )


def build_rubric_from_definition(
    definition: RubricDefinition,
    model_option: "ModelOption",
) -> "Rubric":
    if Rubric is None:
        raise RuntimeError("Docent Rubric class is unavailable. Ensure docent is installed.")

    return Rubric(
        id=definition.rubric_id,
        version=1,
        rubric_text=definition.rubric_text.strip(),
        judge_model=model_option,
        output_schema=definition.output_schema,
    )


def validate_provider_environment(model_option: "ModelOption") -> None:
    """Ensure required environment variables exist for the selected provider."""
    if model_option.provider != "azure_openai":
        return

    missing: list[str] = []
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        if not os.getenv(var):
            missing.append(var)

    api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    if not api_version:
        missing.append("OPENAI_API_VERSION (or AZURE_OPENAI_API_VERSION)")
    else:
        os.environ.setdefault("OPENAI_API_VERSION", api_version)

    if missing:
        raise EnvironmentError(
            "Azure OpenAI environment configuration is incomplete. Missing: "
            + ", ".join(missing)
        )


def build_docent_agent_runs(
    conversations: Sequence[TaskConversation],
    failed_tasks: set[str],
    agent_name: str,
) -> list["AgentRun"]:
    if AgentRun is None or Transcript is None or parse_chat_message is None:
        raise RuntimeError(
            "Docent data models are unavailable. Ensure the docent package is installed and importable."
        )

    agent_runs: list[AgentRun] = []
    for conversation in conversations:
        parsed_messages = []
        for message in conversation.messages:
            payload = {"role": message.role, "content": message.content or ""}
            try:
                parsed_messages.append(parse_chat_message(payload))
            except Exception as exc:  # pragma: no cover - best effort logging
                print(
                    f"⚠️  Skipping malformed message in task {conversation.task_id}: {exc}"
                )
        if not parsed_messages:
            continue

        transcript_metadata = {
            "task_id": conversation.task_id,
            "entry_count": conversation.entry_count,
        }
        transcript = Transcript(messages=parsed_messages, metadata=transcript_metadata)

        run_metadata = {
            "task_id": conversation.task_id,
            "failed": conversation.task_id in failed_tasks,
            "agent": agent_name,
            "entry_count": conversation.entry_count,
            "message_count": len(parsed_messages),
        }
        try:
            agent_run = AgentRun(
                id=str(conversation.task_id),
                transcripts=[transcript],
                metadata=run_metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️  Failed to build AgentRun for {conversation.task_id}: {exc}")
            continue
        agent_runs.append(agent_run)

    return agent_runs


async def evaluate_environmental_barrier(
    agent_runs: Sequence["AgentRun"],
    rubric: "Rubric",
    batch_size: int = DEFAULT_RUBRIC_BATCH_SIZE,
    json_mode: bool = False,
) -> list[LocalRubricEvaluation]:
    if evaluate_rubric is None:
        raise RuntimeError("Docent rubric evaluator is unavailable. Ensure docent is installed.")

    if batch_size <= 0:
        batch_size = 1

    evaluations: list[LocalRubricEvaluation] = []
    for start in range(0, len(agent_runs), batch_size):
        batch = list(agent_runs[start : start + batch_size])
        response_format = {"type": "json_object"} if json_mode else None
        outputs = await evaluate_rubric(batch, rubric, response_format=response_format)
        for agent_run, output in zip(batch, outputs):
            error = None
            if output is None:
                error = "Rubric evaluation returned no valid output."
            task_id = str(agent_run.metadata.get("task_id") or agent_run.id)
            evaluations.append(
                LocalRubricEvaluation(
                    task_id=task_id,
                    rubric_id=rubric.id,
                    rubric_version=rubric.version,
                    output=output,
                    error=error,
                )
            )
    return evaluations


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


def print_grading_results(
    rubric_id: str,
    results: Sequence[LocalRubricEvaluation],
) -> None:
    """Pretty-print grading results returned by Docent rubric evaluation."""
    if not results:
        print("⚠️  Rubric evaluation returned no results.")
        return

    print(f"\n🧪 {rubric_id} Grades:")
    for item in results:
        score = f"{item.score:.2f}" if item.score is not None else "N/A"
        print(f"   • Task {item.task_id}: {item.rubric_id} = {score}")
        if item.error:
            print(f"      Error: {item.error}")
        elif item.explanation:
            print(f"      Explanation: {item.explanation.strip()}")


def write_cloud_grading_csv(
    grading_results: Sequence[LocalRubricEvaluation],
    model_run: str,
    output_path: Path | None = None,
    default_criteria: str = "environmentalbarrier",
    task_success_map: dict[str, bool | None] | None = None,
) -> Path | None:
    """Persist cloud grading results to CSV."""
    if not grading_results:
        return None

    final_path = output_path or (RUBRIC_OUTPUT_DIR / f"{default_criteria}.csv")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "criteria", "grade", "correct", "explanation", "model_run"])
        for item in grading_results:
            if item.score is None:
                grade_value = ""
            else:
                grade_value = f"{item.score:.2f}"

            success = ""
            if task_success_map is not None:
                success_flag = task_success_map.get(item.task_id)
                if success_flag is True:
                    success = "1"
                elif success_flag is False:
                    success = "0"

            writer.writerow(
                [
                    item.task_id,
                    item.rubric_id or default_criteria,
                    grade_value,
                    success,
                    item.explanation.strip(),
                    model_run,
                ]
            )
    return final_path


def default_rubric_output_path(
    rubric_id: str,
    output_dir: Path,
    trace_label: str,
) -> Path:
    rubric_slug = _slugify(rubric_id)
    trace_slug = _slugify(trace_label)
    return output_dir / rubric_slug / f"{trace_slug}.csv"


def run(args: argparse.Namespace) -> None:
    try:
        trace_path = resolve_trace_path(args.trace_file, Path(args.trace_dir))
        data = load_trace_file(trace_path)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return

    trace_label = trace_path.stem

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

    results_block = data.get("results", {}) or {}
    failed_tasks = set(results_block.get("failed_tasks", []))
    successful_tasks = set(results_block.get("successful_tasks", []))
    task_success_map: dict[str, bool | None] = {}
    for task_id in successful_tasks:
        task_success_map[task_id] = True
    for task_id in failed_tasks:
        task_success_map[task_id] = False

    if args.failed_only:
        original = len(conversations)
        conversations = [conv for conv in conversations if conv.task_id in failed_tasks]
        if not conversations:
            print("❌ No failed tasks available for rubric evaluation (--failed-only).")
            return
        print(f"🎯 Filtering to {len(conversations)} failed task(s) out of {original} (--failed-only).")

    if not conversations:
        print("❌ Failed to extract any conversations.")
        return

    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            print("❌ --max-tasks must be positive when provided.")
            return
        original_count = len(conversations)
        conversations = conversations[: args.max_tasks]
        print(f"🔬 Limiting evaluation to {len(conversations)} of {original_count} tasks (--max-tasks).")

    preview_task(conversations[0])

    agent_name = data.get("config", {}).get("agent_name", "unknown-agent")
    if not (args.yes or confirm("\nProceed with Docent rubric evaluation using the configured LLM provider?")):
        print("ℹ️  Evaluation canceled. Inspect the preview above and rerun when ready.")
        return

    if DOCENT_IMPORT_ERROR is not None:
        print(f"❌ Unable to import Docent modules: {DOCENT_IMPORT_ERROR}")
        print("   Ensure the docent repo is installed (e.g., `pip install -e docent`).")
        return

    try:
        agent_runs = build_docent_agent_runs(conversations, failed_tasks, agent_name)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    if not agent_runs:
        print("❌ No agent runs could be constructed for rubric evaluation.")
        return

    effort_override = args.reasoning_effort.lower() if args.reasoning_effort else None

    try:
        model_option = resolve_rubric_model_option(args.rubric_model, reasoning_effort_override=effort_override)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    try:
        validate_provider_environment(model_option)
    except EnvironmentError as exc:
        print(f"❌ {exc}")
        print("   Set the missing environment variables (see docs.transluce.org self-hosting env vars).")
        return

    if args.json_mode and model_option.provider not in {"openai", "azure_openai"}:
        print("❌ JSON mode is only supported for OpenAI or Azure OpenAI providers.")
        return

    rubrics_dir = Path(args.rubrics_dir).expanduser()
    rubric_definitions = load_rubric_definitions(rubrics_dir)
    if not rubric_definitions:
        print(f"❌ No rubric definitions found in {rubrics_dir}. Add *.txt files and retry.")
        return

    batch_size = DEFAULT_RUBRIC_BATCH_SIZE
    env_batch = os.getenv("DOCENT_RUBRIC_BATCH_SIZE")
    if env_batch:
        try:
            batch_size = max(1, int(env_batch))
        except ValueError:
            pass

    output_dir = Path(args.output_dir).expanduser()
    if args.output_mode == "csv":
        output_dir.mkdir(parents=True, exist_ok=True)

    for definition in rubric_definitions:
        rubric = build_rubric_from_definition(definition, model_option)
        output_path = default_rubric_output_path(definition.rubric_id, output_dir, trace_label)

        print(
            f"\n🧪 Running rubric '{definition.rubric_id}' on {len(agent_runs)} agent runs "
            f"with {model_option.provider}:{model_option.model_name} (batch_size={batch_size})..."
        )

        try:
            grading_results = asyncio.run(
                evaluate_environmental_barrier(
                    agent_runs,
                    rubric,
                    batch_size=batch_size,
                    json_mode=args.json_mode,
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on provider availability
            print(f"❌ Rubric '{definition.rubric_id}' evaluation failed: {exc}")
            continue

        print_grading_results(definition.rubric_id, grading_results)

        if args.output_mode == "csv":
            csv_path = write_cloud_grading_csv(
                grading_results,
                trace_label,
                output_path=output_path,
                default_criteria=definition.rubric_id,
                task_success_map=task_success_map,
            )
            if csv_path:
                print(f"🗂️  Rubric CSV written to {csv_path}")
        else:
            print("🗂️  Output mode 'stdout' selected; skipping CSV export.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
