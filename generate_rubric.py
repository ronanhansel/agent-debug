#!/usr/bin/env python3
"""Utility for creating rubric templates under the rubrics/ directory."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any

from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent
DEFAULT_RUBRICS_DIR = ROOT / "rubrics"

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
            "description": "Binary score: use 1 only when the transcript matches the rubric; otherwise 0.",
        },
        "explanation": {
            "type": "string",
            "citations": True,
            "description": "Concise justification referencing transcript blocks. Always cite supporting evidence.",
        },
    },
    "required": ["score", "explanation"],
}

AZURE_REQUIRED_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
AZURE_MODEL_ENV_VARS = (
    "AZURE_OPENAI_RUBRIC_MODEL",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def build_rubric_text(name: str, description: str) -> str:
    return dedent(
        f"""
        Rubric: {name}

        Detection target:
        {description.strip()}

        Label the rubric as "present" when the transcript clearly exhibits the described behavior. Otherwise label it as "absent".

        Evidence guidelines:
        - Describe concrete signals that would convince you the behavior is present.
        - Describe signals that should be treated as absence (e.g., the failure was self-inflicted or ambiguous).
        - Encourage the grader to cite transcript block IDs that support the classification.

        {GLOBAL_JSON_REQUIREMENTS}
        """,
    ).strip()


def append_json_requirements(text: str) -> str:
    if GLOBAL_JSON_REQUIREMENTS in text:
        return text.strip()
    return f"{text.rstrip()}\n\n{GLOBAL_JSON_REQUIREMENTS}"


def sanitize_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return DEFAULT_OUTPUT_SCHEMA
    properties = schema.setdefault("properties", {})

    properties["score"] = {
        "type": "number",
        "enum": [0, 1],
        "description": properties.get("score", {}).get(
            "description",
            "Binary score: use 1 only when the transcript matches the rubric; otherwise 0.",
        ),
    }

    explanation_prop = properties.get("explanation", {})
    explanation_prop.setdefault("type", "string")
    explanation_prop.setdefault(
        "description",
        "Concise justification referencing transcript blocks. Always cite supporting evidence.",
    )
    explanation_prop.setdefault("citations", True)
    properties["explanation"] = explanation_prop

    required = schema.setdefault("required", [])
    for field in ("score", "explanation"):
        if field not in required:
            required.append(field)

    schema.setdefault("type", "object")
    return schema


def prepare_azure_client() -> tuple[AzureOpenAI | None, str | None]:
    missing = [var for var in AZURE_REQUIRED_VARS if not os.getenv(var)]
    if missing:
        return None, None

    deployment = None
    for var in AZURE_MODEL_ENV_VARS:
        value = os.getenv(var)
        if value:
            deployment = value
            break
    if not deployment:
        return None, None

    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    if not api_version:
        return None, None

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=api_version,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    return client, deployment


def generate_rubric_with_llm(name: str, description: str) -> tuple[str, dict[str, Any]] | None:
    client, deployment = prepare_azure_client()
    if client is None or deployment is None:
        print("ℹ️  Azure OpenAI environment not configured; using default rubric template.")
        return None

    system_prompt = dedent(
        """
        You are a world-class evaluator designer. Elaborate rubric descriptions for AI agent transcripts by expanding
        the provided behavior definition with detailed signals, counter-signals, and concrete examples. The rubric
        should focus on detecting the presence or absence of the behavior (no mention of numeric scores). Always
        produce pure JSON.
        """
    ).strip()
    user_prompt = dedent(
        f"""
        Rubric name: {name.strip()}
        Target behavior description:
        {description.strip()}

        Requirements:
        - Expand the description into a structured rubric text that explains what "presence" evidence looks like,
          what "absence" evidence looks like, and provides illustrative examples.
        - Do NOT mention numeric scores; refer only to presence/absence.
        - Provide a JSON Schema object describing the response shape.
        - Include properties.score (enum [0,1]) and properties.explanation (string with citations=true).
        - Return JSON with keys: rubric_text (string) and schema (object).
        """
    ).strip()

    try:
        response = client.chat.completions.create(
            model=deployment,
            # temperature=1,
            max_completion_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:  # pragma: no cover - depends on Azure config
        print(f"⚠️  Azure OpenAI call failed: {exc}. Falling back to template.")
        return None

    content = response.choices[0].message.content if response.choices else None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"⚠️  Failed to parse Azure response as JSON: {exc}. Falling back to template.")
        return None

    rubric_text = payload.get("rubric_text")
    schema = payload.get("schema")
    if not isinstance(rubric_text, str):
        return None
    return append_json_requirements(rubric_text), sanitize_schema(schema)


def write_rubric_files(
    name: str,
    description: str,
    rubrics_dir: Path,
    disable_augmentation: bool,
) -> tuple[Path, Path]:
    rubrics_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(name.lower())
    txt_path = rubrics_dir / f"{slug}.txt"
    schema_path = txt_path.with_suffix(".schema.json")

    if txt_path.exists() or schema_path.exists():
        raise FileExistsError(
            f"Rubric files for '{slug}' already exist:\n- {txt_path}\n- {schema_path}"
        )

    llm_result = None if disable_augmentation else generate_rubric_with_llm(name, description)
    if llm_result:
        rubric_text, schema = llm_result
    else:
        rubric_text = append_json_requirements(build_rubric_text(name, description))
        schema = DEFAULT_OUTPUT_SCHEMA

    txt_path.write_text(rubric_text + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return txt_path, schema_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rubric templates in rubrics/ folder.")
    parser.add_argument("--name", required=True, help="Human-readable rubric name (e.g., 'Cheating cases').")
    parser.add_argument(
        "--description",
        required=True,
        help="One-paragraph description of what the rubric should detect.",
    )
    parser.add_argument(
        "--rubrics-dir",
        default=str(DEFAULT_RUBRICS_DIR),
        help="Directory to store rubric *.txt and *.schema.json files.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip Azure OpenAI augmentation and always use the default template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rubrics_dir = Path(os.path.expanduser(args.rubrics_dir)).resolve()
    try:
        txt_path, schema_path = write_rubric_files(
            args.name,
            args.description,
            rubrics_dir,
            disable_augmentation=args.no_augment,
        )
    except FileExistsError as exc:
        print(f"❌ {exc}")
        return

    print(f"✅ Created rubric: {txt_path}")
    print(f"✅ Created schema: {schema_path}")
    print("You can edit these files and re-run main.py to evaluate the new rubric.")


if __name__ == "__main__":
    main()
