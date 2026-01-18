# Instructions: Adding a New Benchmark to the Item Fixing Pipeline

This document provides step-by-step instructions for Claude Code to set up a new benchmark for the Item Fixing Pipeline, including rubric templates AND Claude fixer scripts.

---

## Overview

When adding a new benchmark, you need to create:
1. **Rubric template** - For LLM-based IFE detection
2. **Claude fixer script** - For automated fix generation
3. **Fix runner script** (optional) - For applying fixes and re-running evaluation
4. **Documentation updates** - CLAUDE.md entries

---

## Step 1: Explore the Benchmark

First, understand the benchmark structure:

```bash
# Find the agent implementation
ls -la hal-harness/agents/ | grep -i <benchmark_name>

# Find the benchmark definition
cat hal-harness/hal/benchmarks/<benchmark_name>.py
```

Answer these questions:
- What is the agent's entry point? (e.g., `main.py`, `agent.py`)
- What function is called? (e.g., `run`, `main.run`)
- Does it require Docker?
- What are common failure patterns?
- How is success/failure determined (metrics, test cases, etc.)?
- What are the trace file structures?

---

## Step 2: Research Known Issues

Before creating the rubric, research the benchmark:

```bash
# Search for known issues, papers, GitHub issues
# Use WebSearch tool to find:
# - Academic papers analyzing benchmark limitations
# - GitHub issues about broken/flaky tasks
# - Community discussions about evaluation problems
```

Document findings about:
- Known problematic tasks
- Common false positive/negative patterns
- Environment/dependency issues
- Evaluation metric limitations

---

## Step 3: Create the Rubric Template

Create `rubric_templates/<benchmark_name>.txt` using this structure:

```markdown
# <Benchmark Name> Intrinsic Formation Error Detection Rubric

## Purpose
This rubric evaluates whether a task failure on the <Benchmark Name> benchmark stems from an **Intrinsic Formation Error (IFE)** in the benchmark itself, rather than agent capability limitations.

## Scoring
- **Score 1**: An intrinsic benchmark defect exists AND caused the failure (no agent could succeed)
- **Score 0**: The failure is due to agent capability issues (a better agent could have succeeded)

## Two-Question Framework

### Question 1: Does an Intrinsic Formation Defect Exist?
Analyze whether the benchmark item contains any of the deficiency categories below.

### Question 2: Did the Defect Cause This Failure?
Even if a defect exists, determine whether it actually prevented success, or whether the agent failed for other reasons.

---

## Deficiency Categories for <Benchmark Name>

### 1. <Category Name>
<Description>:
- <Indicator>
- <Indicator>

### 2. <Category Name>
<Description>:
- <Indicator>

[Add 4-6 benchmark-specific categories based on research]

---

## CRITICAL EXCLUSIONS: Agent Capability Issues (Score 0)

Do NOT classify the following as benchmark deficiencies:

### 1. <Agent Issue Type>
- <Example>
- These are agent capability issues, not benchmark defects

[Add 4-6 agent issue categories]

---

## Evidence Requirements

For Score 1, you MUST provide:
1. **Specific defect identification**: Quote the exact issue
2. **Impossibility proof**: Explain why NO agent could overcome this
3. **Exclusion of alternatives**: Show no workaround exists

For Score 0, explain:
1. What the agent did wrong
2. What a capable agent could have done instead

---

## Response Format

Respond with a JSON object:

```json
{
  "score": 0 or 1,
  "deficiency_exists": true/false,
  "deficiency_caused_failure": true/false,
  "deficiency_type": "category name or 'none'",
  "existence_reasoning": "analysis of whether a benchmark defect exists",
  "causation_reasoning": "analysis of whether the defect caused failure",
  "evidence": "specific quotes from transcript supporting conclusion"
}
```

---

## Common <Benchmark Name> Failure Patterns

### Likely Agent Issues (Score 0):
- <Pattern>
- <Pattern>

### Potential Benchmark Issues (Score 1):
- <Pattern>
- <Pattern>

---

## Cross-Run Analysis Guidelines

When evaluating IFEs, use evidence from multiple model runs:

### Strong IFE Indicators:
1. Same error across ALL models
2. Valid output rejected by evaluation
3. Environment/dependency issues block ALL models

### Weak IFE Indicators (Lean toward Score 0):
1. Only one model fails with an error
2. Different error types across models
3. Some models succeed on the task

---

## Exploratory Analysis: Discovering Hidden Benchmark Issues

[Add benchmark-specific investigation prompts - use suggestions rather than strict rules so the LLM can freely explore and reason]

### Areas to Investigate (Suggestions, Not Rules)

**1. <Domain-Specific Area>**
Consider whether...

**2. <Evaluation Fairness>**
Examine whether...

### Questions to Ask Yourself

1. **"Would a domain expert find this solvable?"**
2. **"Is the evaluation measuring the right thing?"**
3. **"Could the environment be the problem?"**

### Known Problematic Task Patterns

[Add specific task IDs and patterns discovered during trace analysis]
```

---

## Step 4: Analyze Traces (If Available)

If trace files exist, analyze them to populate the rubric with real data:

```python
# Example analysis pattern
import json
from pathlib import Path

traces_dir = Path("traces")
for trace_file in traces_dir.glob(f"{benchmark_name}_*.json"):
    data = json.loads(trace_file.read_text())
    # Analyze:
    # - Success/failure rates
    # - Common error patterns
    # - Cross-model failure correlation
```

Update the rubric with:
- Actual failure statistics (e.g., "67/102 tasks failed across ALL 4 models")
- Specific problematic task IDs
- Error message patterns

---

## Step 5: Create the Claude Fixer Script

Create `scripts/claude_fixer_<benchmark_name>.py` based on the template:

```python
#!/usr/bin/env python3
"""
Claude Code CLI-based <Benchmark Name> Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
in <Benchmark Name> tasks WITHOUT nerfing the problems.

Usage:
    # List tasks with IFEs detected
    python scripts/claude_fixer_<benchmark_name>.py --list-ife-tasks

    # Fix a specific task
    python scripts/claude_fixer_<benchmark_name>.py --task-id <id>

    # Fix all tasks with Grade=1 in judge verdict
    python scripts/claude_fixer_<benchmark_name>.py --all-ife

    # Batch mode (multiple tasks per Claude session)
    python scripts/claude_fixer_<benchmark_name>.py --all-ife --batch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "traces"
FIXES_DIR = REPO_ROOT / "fixes"
RUBRICS_OUTPUT_DIR = REPO_ROOT / "rubrics_output" / "<benchmark_name>"
JUDGE_OUTPUT = REPO_ROOT / "judge_output" / "<benchmark_name>_verdict.csv"
BENCHMARK = "<benchmark_name>"

# === COPY UTILITY FUNCTIONS FROM claude_fixer_scicode.py or claude_fixer_scienceagentbench.py ===
# - log(), log_progress()
# - has_existing_fix()
# - load_judge_verdicts()
# - load_all_rubric_evaluations()
# - load_task_conversations()  # Adapt to benchmark's trace structure
# - get_ife_tasks()
# - format_stream_json()
# - run_claude_code()
# - process_task()
# - process_batch()

def build_claude_prompt_batch(tasks_data: List[Dict[str, Any]]) -> str:
    """Build the prompt for Claude Code CLI with multiple tasks."""

    # === CUSTOMIZE THIS PROMPT FOR THE BENCHMARK ===
    # This is a RIGOROUS prompt template - be thorough in error checking
    prompt = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in <Benchmark Name> tasks.

## CRITICAL CONSTRAINTS
1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the problem easier
2. **PRESERVE DIFFICULTY** - The task should remain as challenging as intended
3. **NO NERFING** - Do not simplify, give hints, or pre-compute results

## <BENCHMARK NAME> HARNESS STRUCTURE

**First, read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/<benchmark_name>.py` - Main benchmark class
- `hal-harness/agents/<agent_dir>/` - Agent implementation
- [Other relevant files specific to this benchmark]

**How Evaluation Works:**
[Describe the evaluation process specific to this benchmark]

## THOROUGH ERROR ANALYSIS CHECKLIST

For EACH task, systematically check ALL of these potential error sources:

### 1. Environment/Dependency Issues
- [ ] Missing Python packages in Docker container
- [ ] Package version conflicts between dependencies
- [ ] Missing system libraries (C headers, compilers)
- [ ] Conda vs pip installation conflicts
- [ ] GPU/CUDA requirements not met
- [ ] Memory/timeout limits too restrictive

### 2. Data/Input Issues
- [ ] Missing or corrupted data files
- [ ] Data format differs from documentation
- [ ] Column names don't match task description
- [ ] Encoding issues (UTF-8, binary files)
- [ ] Relative vs absolute path issues

### 3. Task Specification Issues
- [ ] Ambiguous output format requirements
- [ ] Unclear success criteria
- [ ] Missing domain knowledge in instructions
- [ ] Conflicting requirements between steps
- [ ] Unstated assumptions from source papers

### 4. Evaluation Script Issues
- [ ] Numerical tolerance too strict
- [ ] Format-sensitive comparison (whitespace, ordering)
- [ ] Evaluation crashes on valid outputs
- [ ] Metrics don't match task description
- [ ] GPT-4 judge subjectivity (for figure tasks)

### 5. Gold Program/Reference Issues
- [ ] Gold program has hardcoded paths
- [ ] Gold program uses unavailable libraries
- [ ] Multiple valid approaches rejected
- [ ] Gold program doesn't match task requirements

### 6. Cross-Model Failure Patterns
- [ ] Same error across ALL models → likely IFE
- [ ] Valid output rejected by evaluation → evaluation issue
- [ ] Environment blocks ALL models identically → setup issue

## KNOWN IFE PATTERNS FOR <BENCHMARK NAME>

[List known patterns from trace analysis - be specific with task IDs and error messages]

## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/<benchmark_name>/TASK_ID/`

**Environment Fixes** (`env_override.json`):
```json
{{
  "HAL_CONDA_PACKAGES": "package1 package2",
  "HAL_PIP_PACKAGES": "package3",
  "HAL_SYSTEM_PACKAGES": "libfoo-dev cmake",
  "HAL_TIMEOUT_SECONDS": 600,
  "notes": "Justification for these changes"
}}
```

**Evaluation Fixes** (`evaluation_override.json`):
```json
{{
  "tolerance": 1e-4,
  "allow_alternative_outputs": true,
  "skip_format_check": false,
  "notes": "Justification for tolerance adjustment"
}}
```

**Instruction Clarifications** (`instruction_override.json`):
```json
{{
  "clarifications": [
    "Use scipy.integrate.simpson instead of deprecated simps",
    "Output file should be named exactly 'results.csv'"
  ],
  "additional_context": "Any missing domain knowledge to include"
}}
```

**Documentation** (`README.md`):
- Root cause analysis of the IFE
- What fix was applied and why
- Why this preserves task difficulty
- Expected outcome after fix

## FIX RUNNER SCRIPT GENERATION

After creating fixes, also update or create the fix runner script:
`scripts/run_<benchmark_name>_fixes.py`

The fix runner must:
1. Load fixes from `fixes/<benchmark_name>/<task_id>/`
2. Apply environment overrides before evaluation
3. Inject instruction clarifications into prompts
4. Adjust evaluation parameters as specified
5. Run HAL evaluation with the fixes applied
6. Output new traces with a configurable prefix

**Reference implementation**: See `scripts/run_scicode_fixes.py` for the pattern.

## TASKS TO PROCESS

{tasks_text}

## BEGIN - SYSTEMATIC APPROACH

For EACH task:

1. **Read the benchmark code** to understand evaluation
2. **Load the specific task** from the dataset/trace
3. **Analyze ALL error messages** from agent conversations
4. **Check EACH item** in the error analysis checklist above
5. **Cross-reference with other models** - same error = likely IFE
6. **Create fix OR document why no fix needed**
7. **Verify fix doesn't nerf the problem**

After processing all tasks:
8. **Update the fix runner script** if new fix types were used
9. **Test that fixes can be applied** without errors

Remember: Make evaluation FAIR, not EASY. Be THOROUGH in diagnosis.
'''
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Claude Code CLI-based <Benchmark Name> Fixer")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Task ID(s) to process")
    parser.add_argument("--all-ife", action="store_true", help="Process all tasks with IFE detected")
    parser.add_argument("--list-ife-tasks", action="store_true", help="List tasks with IFE detected")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks with existing fixes")
    parser.add_argument("--batch", action="store_true", help="Process all tasks in single session")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Implementation follows pattern from claude_fixer_scicode.py
    # ...

if __name__ == "__main__":
    main()
```

**Key customizations needed:**
1. Update `BENCHMARK`, `RUBRICS_OUTPUT_DIR`, `JUDGE_OUTPUT` constants
2. Adapt `load_task_conversations()` to the benchmark's trace structure
3. Customize `build_claude_prompt_batch()` with:
   - Benchmark-specific harness files to read
   - Evaluation process description
   - Known IFE patterns from trace analysis
   - Fix format specific to the benchmark

---

## Step 6: Create the Fix Runner Script

Create `scripts/run_<benchmark_name>_fixes.py` to apply fixes and re-run evaluations.

**Reference**: Use `scripts/run_scicode_fixes.py` as a template.

### 6.1 Create Model Configuration File

First, create `model_to_baseline_<benchmark_name>.json` to map models to their configurations:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "baseline_trace": "<benchmark>_agent_gpt4120250414_UPLOAD.json",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3-2025-04-16",
    "short_name": "o3",
    "baseline_trace": "<benchmark>_agent_o320250416_UPLOAD.json",
    "reasoning_effort": "medium",
    "max_steps": 5
  },
  "openai/o4-mini-2025-04-16-high": {
    "model_id": "openai/o4-mini-2025-04-16",
    "short_name": "o4-mini-high",
    "baseline_trace": "<benchmark>_agent_o4mini20250416_high_UPLOAD.json",
    "reasoning_effort": "high",
    "max_steps": 5
  },
  "anthropic/claude-sonnet-4-5-20250929": {
    "model_id": "anthropic/claude-sonnet-4-5-20250929",
    "short_name": "sonnet-4.5",
    "baseline_trace": "<benchmark>_agent_claudesonnet45_UPLOAD.json",
    "max_steps": 5
  }
}
```

### 6.2 Create Fix Runner Script

```python
#!/usr/bin/env python3
"""
Apply <Benchmark Name> fix packages and re-run HAL evaluations.

This script:
1. Loads model configurations from model_to_baseline_<benchmark>.json
2. Loads fixes from fixes/<benchmark_name>/<task_id>/
3. Creates a modified dataset with instruction clarifications injected
4. Applies environment overrides (packages, timeouts)
5. Runs HAL evaluation for fixed tasks using the original failing model
6. Outputs traces with a configurable prefix

Usage:
    # List available fixes
    python scripts/run_<benchmark_name>_fixes.py --list-fixes

    # Dry run - see what would happen
    python scripts/run_<benchmark_name>_fixes.py --task-id 11 --dry-run

    # Run fixes for all failed tasks that have fixes (auto-detects model from rubric)
    python scripts/run_<benchmark_name>_fixes.py --prefix iter1_ \\
        --rubric-csv rubrics_output/<benchmark>/<benchmark>_combined.csv --docker

    # Run with specific model override
    python scripts/run_<benchmark_name>_fixes.py --prefix iter1_ --model gpt-4o
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_DIR = REPO_ROOT / "fixes" / "<benchmark_name>"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "model_to_baseline_<benchmark_name>.json"

def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline_<benchmark>.json and return dict mapping model keys to config.

    Format:
    {
      "openai/gpt-4.1-2025-04-14": {
        "model_id": "openai/gpt-4.1-2025-04-14",
        "short_name": "gpt-4.1",
        "baseline_trace": "<benchmark>_..._UPLOAD.json",
        "reasoning_effort": "high",  // optional
        "max_steps": 5
      },
      ...
    }
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))

    # Flat dict format - expected format
    if isinstance(data, dict) and "models" not in data:
        return data

    # Legacy format with "models" array - convert to flat dict
    if "models" in data and isinstance(data["models"], list):
        result = {}
        for model_entry in data["models"]:
            model_id = model_entry.get("model_id")
            if model_id:
                result[model_id] = model_entry
        return result

    return {}

def load_rubric_task_models(csv_path: Path) -> List[Tuple[str, str]]:
    """Load rubric CSV and return list of (task_id, model_id) pairs for failed tasks."""
    task_model_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    if not csv_path.exists():
        return task_model_pairs

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            correct = row.get("correct", "").strip()
            model_run = row.get("model_run", "").strip()

            # Only include failed tasks (correct=0)
            if task_id and correct == "0" and model_run:
                model_id = _extract_model_from_run_name(model_run)
                if model_id and (task_id, model_id) not in seen:
                    task_model_pairs.append((task_id, model_id))
                    seen.add((task_id, model_id))

    return task_model_pairs

def _extract_model_from_run_name(model_run: str) -> Optional[str]:
    """Extract model config key from model_run column.

    Customize this function for your benchmark's naming conventions.
    """
    # Add patterns specific to your benchmark
    patterns = [
        (r"gpt[\-_]?4[\-_]?1[\-_]?2025", "openai/gpt-4.1-2025-04-14"),
        (r"o3[\-_]?2025", "openai/o3-2025-04-16"),
        (r"o4[\-_]?mini.*high", "openai/o4-mini-2025-04-16-high"),
        (r"o4[\-_]?mini.*low", "openai/o4-mini-2025-04-16-low"),
        (r"claude[\-_]?sonnet[\-_]?4[\-_]?5", "anthropic/claude-sonnet-4-5-20250929"),
    ]

    for pattern, config_key in patterns:
        if re.search(pattern, model_run, re.IGNORECASE):
            return config_key
    return None

def load_fix(task_id: str) -> Dict[str, Any]:
    """Load all fix files for a task."""
    fix_dir = FIXES_DIR / task_id
    fix = {"task_id": task_id, "exists": fix_dir.exists()}

    if fix_dir.exists():
        for fix_file in ["env_override.json", "instruction_override.json",
                         "evaluation_override.json"]:
            path = fix_dir / fix_file
            if path.exists():
                fix[fix_file.replace(".json", "")] = json.loads(path.read_text())

    return fix

def apply_env_override(fix: Dict[str, Any]) -> Dict[str, str]:
    """Convert env_override to environment variables."""
    env = os.environ.copy()
    if "env_override" in fix:
        override = fix["env_override"]
        for key in ["HAL_CONDA_PACKAGES", "HAL_PIP_PACKAGES", "HAL_APT_PACKAGES",
                    "HAL_CONDA_CHANNELS", "HAL_TIMEOUT_SECONDS"]:
            if key in override:
                env[key] = str(override[key])
    return env

def inject_instruction_clarifications(task_data: Dict, fix: Dict) -> Dict:
    """Inject instruction clarifications into task prompt."""
    if "instruction_override" not in fix:
        return task_data

    clarifications = fix["instruction_override"].get("clarifications", [])
    additional_context = fix["instruction_override"].get("additional_context", "")

    if not clarifications and not additional_context:
        return task_data

    # Build clarification text
    clarification_text = ""
    if clarifications:
        clarification_text += "\n\n**IMPORTANT CLARIFICATIONS:**\n" + \
                            "\n".join(f"- {c}" for c in clarifications)
    if additional_context:
        clarification_text += f"\n\n**ADDITIONAL CONTEXT:**\n{additional_context}"

    # Modify task_data based on benchmark structure (customize for your benchmark)
    if "problem_statement" in task_data:
        task_data["problem_statement"] += clarification_text
    elif "instructions" in task_data:
        task_data["instructions"] += clarification_text
    elif "task_inst" in task_data:  # ScienceAgentBench format
        task_data["task_inst"] += clarification_text

    return task_data

def run_hal_evaluation(task_id: str, model_config: Dict[str, Any], fix: Dict[str, Any],
                       prefix: str, docker: bool = False) -> int:
    """Run HAL evaluation with fixes applied for a single task+model."""
    model_id = model_config.get("model_id", "gpt-4o")
    short_name = model_config.get("short_name", "model")
    reasoning_effort = model_config.get("reasoning_effort")
    max_steps = model_config.get("max_steps", 5)

    # Build HAL command
    cmd = [
        "hal-eval",
        "--benchmark", "<benchmark_name>",
        "--agent_dir", "agents/<agent_dir>/",
        "--agent_function", "<function_name>",
        "--agent_name", f"Fixed Agent ({short_name})",
        "-A", f"model_name={model_id}",
        "-A", f"max_steps={max_steps}",
        "--run_id", f"{prefix}{short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--task_ids", task_id,
    ]

    if reasoning_effort:
        cmd.extend(["-A", f"reasoning_effort={reasoning_effort}"])

    if docker:
        cmd.append("--docker")

    # Apply environment overrides from fix
    env = apply_env_override(fix)

    # Run evaluation
    result = subprocess.run(cmd, cwd=HAL_HARNESS, env=env)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run <Benchmark Name> with fixes applied")
    parser.add_argument("--list-fixes", action="store_true", help="List available fixes")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Task IDs to run")
    parser.add_argument("--rubric-csv", type=str, help="Rubric CSV to get failed task/model pairs")
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG),
                        help="Path to model configuration JSON")
    parser.add_argument("--prefix", type=str, default="fixed_", help="Output prefix")
    parser.add_argument("--model", type=str, help="Override model for all tasks")
    parser.add_argument("--docker", action="store_true", help="Use Docker execution")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")

    args = parser.parse_args()

    # Load model configurations
    model_configs = load_model_config(Path(args.model_config))

    if args.list_fixes:
        for fix_dir in sorted(FIXES_DIR.iterdir()):
            if fix_dir.is_dir():
                fix = load_fix(fix_dir.name)
                print(f"{fix_dir.name}: {list(fix.keys())}")
        return

    # Determine task/model pairs to run
    if args.rubric_csv:
        # Get failed task/model pairs from rubric
        task_model_pairs = load_rubric_task_models(Path(args.rubric_csv))
        # Filter to only tasks with fixes
        task_model_pairs = [(tid, mid) for tid, mid in task_model_pairs
                           if (FIXES_DIR / tid).exists()]
    elif args.task_ids:
        # Use specified tasks with default or override model
        default_model = args.model or list(model_configs.keys())[0] if model_configs else "gpt-4o"
        task_model_pairs = [(tid, default_model) for tid in args.task_ids]
    else:
        # All fixes with default model
        default_model = args.model or list(model_configs.keys())[0] if model_configs else "gpt-4o"
        task_model_pairs = [(d.name, default_model) for d in FIXES_DIR.iterdir() if d.is_dir()]

    if args.dry_run:
        print(f"Would run {len(task_model_pairs)} task/model pairs:")
        for tid, mid in task_model_pairs:
            fix = load_fix(tid)
            config = model_configs.get(mid, {"short_name": mid})
            print(f"  {tid} with {config.get('short_name', mid)}: {list(fix.keys())}")
        return

    # Run evaluations
    for task_id, model_id in task_model_pairs:
        fix = load_fix(task_id)
        config = model_configs.get(model_id, {"model_id": model_id, "short_name": model_id})
        print(f"Running {task_id} with {config.get('short_name', model_id)}...")
        run_hal_evaluation(task_id, config, fix, args.prefix, args.docker)

if __name__ == "__main__":
    main()
```

**Key customizations needed:**
1. Update `<benchmark_name>`, agent directory, function name
2. Update `_extract_model_from_run_name()` patterns for your benchmark's naming
3. Adapt `inject_instruction_clarifications()` for benchmark's task structure (key names)
4. Create corresponding `model_to_baseline_<benchmark>.json` configuration file
5. Handle evaluation override parameters if needed

---

## Step 7: Create Output Directories

```bash
mkdir -p rubrics_output/<benchmark_name>
mkdir -p fixes/<benchmark_name>
```

---

## Step 8: Update CLAUDE.md

### 8.1 Add to Supported Benchmarks Table

Find the "Supported Benchmarks" table and add:

```markdown
| **<Benchmark Name>** | `agents/<agent_dir>/` | `rubric_templates/<benchmark>.txt` | Ready |
```

### 8.2 Add Benchmark-Specific Details Section

Add a new section under "Benchmark-Specific Details":

```markdown
### <Benchmark Name>
**Purpose**: <One line description>

**Agent**: `hal-harness/agents/<agent_dir>/`
- <Key feature 1>
- <Key feature 2>

**Common Issues**:
- <Issue 1>
- <Issue 2>

**Known Research-Documented Issues** (if any):
- <Issue from papers/analysis>
- <Issue from trace analysis>

**Rubric Features** (`rubric_templates/<benchmark>.txt`):
- <Key rubric feature 1>
- <Key rubric feature 2>

**Fixer Script**: `scripts/claude_fixer_<benchmark>.py`
- Diagnoses IFEs from rubric evaluations and traces
- Creates fixes in `fixes/<benchmark>/<task_id>/`

**Evaluation**: <How tasks are evaluated>

**HAL Command**:
```bash
hal-eval --benchmark <benchmark_name> \
    --agent_dir agents/<agent_dir>/ \
    --agent_function <function_name> \
    --agent_name "<Agent Name>" \
    -A model_name=gpt-4o
```
```

### 8.3 Add to Quick Reference Table

Find the "Quick Reference" table and add:

```markdown
| <Benchmark Name> | `<benchmark>.txt` | <Key focus areas> |
```

---

## Step 9: Test the Setup

### 9.1 Run Rubric Evaluation

```bash
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-4o \
    --failed-only -y
```

### 9.2 Aggregate Verdicts

```bash
python scripts/judge.py \
    --rubric-dir rubrics_output/<benchmark> \
    --output judge_output/<benchmark>_verdict.csv
```

### 9.3 Test Fixer Script

```bash
# List IFE tasks
python scripts/claude_fixer_<benchmark>.py --list-ife-tasks

# Test with one task
python scripts/claude_fixer_<benchmark>.py --task-id <some_id>
```

### 9.4 Test Fix Runner Script

```bash
# List available fixes
python scripts/run_<benchmark>_fixes.py --list-fixes

# Dry run
python scripts/run_<benchmark>_fixes.py --task-id <some_id> --dry-run

# Run with fixes applied
python scripts/run_<benchmark>_fixes.py --task-id <some_id> --prefix fixed_
```

---

## Checklist

### Rubric Setup
- [ ] Explored benchmark (agent location, evaluation method)
- [ ] Researched known issues (papers, GitHub issues, community)
- [ ] Analyzed traces (if available) for failure patterns
- [ ] Created `rubric_templates/<benchmark>.txt` with:
  - [ ] Benchmark-specific deficiency categories
  - [ ] Agent capability exclusions
  - [ ] Cross-run analysis guidelines
  - [ ] Exploratory analysis section
  - [ ] Known problematic task patterns

### Fixer Script Setup
- [ ] Created `scripts/claude_fixer_<benchmark>.py` with:
  - [ ] Correct constants (BENCHMARK, paths)
  - [ ] Adapted trace loading for benchmark structure
  - [ ] Customized prompt with benchmark-specific context
  - [ ] Thorough error analysis checklist in prompt
  - [ ] Fix format appropriate for benchmark
  - [ ] Fix runner generation instructions in prompt

### Fix Runner Script Setup
- [ ] Created `model_to_baseline_<benchmark>.json` with:
  - [ ] Model ID mappings for all tested models
  - [ ] Baseline trace file references
  - [ ] Reasoning effort settings (for o-series models)
  - [ ] Max steps and other agent parameters
- [ ] Created `scripts/run_<benchmark>_fixes.py` with:
  - [ ] Load model config from `model_to_baseline_<benchmark>.json`
  - [ ] Load fixes from `fixes/<benchmark>/<task_id>/`
  - [ ] Apply environment overrides
  - [ ] Inject instruction clarifications
  - [ ] Apply evaluation parameter adjustments
  - [ ] Run HAL evaluation with modified config
  - [ ] Output traces with configurable prefix
  - [ ] Support `--rubric-csv` for auto-detecting failed task/model pairs

### Documentation
- [ ] Updated CLAUDE.md: Supported Benchmarks table
- [ ] Updated CLAUDE.md: Benchmark-Specific Details section
- [ ] Updated CLAUDE.md: Quick Reference table
- [ ] Updated CLAUDE.md: Fixer Scripts section

### Directories
- [ ] Created `rubrics_output/<benchmark>/`
- [ ] Created `fixes/<benchmark>/`

---

## Reference: Existing Implementations

Use these as templates:

| Benchmark | Rubric Template | Fixer Script | Fix Runner | Key Features |
|-----------|----------------|--------------|------------|--------------|
| SciCode | `scicode.txt` | `claude_fixer_scicode.py` | `run_scicode_fixes.py` | Dependency constraints, test tolerance |
| SWE-bench | `swebench.txt` | N/A | N/A | Cross-run validation, known flaky tasks |
| AssistantBench | `assistantbench.txt` | N/A | N/A | Bot detection, environment blocking |
| ScienceAgentBench | `scienceagentbench.txt` | `claude_fixer_scienceagentbench.py` | `run_scienceagentbench_fixes.py` | Domain libraries, figure evaluation |
| CoreBench | `corebench.txt` | N/A | `run_corebench_fixes.py` | Docker/container issues |
| USACO | `usaco.txt` | N/A | `run_usaco_fixes.py` | Algorithm correctness, time limits |

---

## Notes

- The unified schema (`rubric_templates/rubric.schema.json`) is automatically used
- Focus deficiency categories on issues specific to the benchmark's domain
- Err on the side of classifying issues as agent capability problems (Score 0)
- Use exploratory prompts (suggestions) rather than strict rules in rubrics
- Key principle for fixers: Make evaluation FAIR, not EASY
