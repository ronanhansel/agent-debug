# Instructions: Adding a New Benchmark to the Item Fixing Pipeline

This document provides step-by-step instructions for Claude Code to set up a new benchmark for the Item Fixing Pipeline.

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

---

## Step 2: Create the Rubric Template

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

[Add 4-6 benchmark-specific categories]

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
```

---

## Step 3: Update CLAUDE.md

### 3.1 Add to Supported Benchmarks Table

Find the "Supported Benchmarks" table and add:

```markdown
| **<Benchmark Name>** | `agents/<agent_dir>/` | `rubric_templates/<benchmark>.txt` | Ready |
```

### 3.2 Add Benchmark-Specific Details Section

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

### 3.3 Add to Quick Reference Table

Find the "Quick Reference" table at the bottom and add:

```markdown
| <Benchmark Name> | `<benchmark>.txt` | <Key focus areas> |
```

---

## Step 4: Create Output Directory

```bash
mkdir -p rubrics_output/<benchmark_name>
```

---

## Checklist

- [ ] Explored benchmark (agent location, evaluation method)
- [ ] Created `rubric_templates/<benchmark>.txt` with benchmark-specific categories
- [ ] Updated CLAUDE.md: Supported Benchmarks table
- [ ] Updated CLAUDE.md: Benchmark-Specific Details section
- [ ] Updated CLAUDE.md: Quick Reference table
- [ ] Created `rubrics_output/<benchmark>/` directory

---

## Notes

- The unified schema (`rubric_templates/rubric.schema.json`) is automatically used - no need to create a separate schema file
- Focus deficiency categories on issues specific to the benchmark's domain
- Err on the side of classifying issues as agent capability problems (Score 0)
- Reference existing rubrics (`scicode.txt`, `swebench.txt`, etc.) for examples
