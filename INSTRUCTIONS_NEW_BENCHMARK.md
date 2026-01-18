# Instructions: Adding a New Benchmark to the Item Fixing Pipeline

This document provides step-by-step instructions for Claude Code to set up the Item Fixing Pipeline for a new benchmark.

## Prerequisites

Before starting, ensure you have:
1. Access to the benchmark's agent implementation in `hal-harness/agents/`
2. Access to trace files from failed runs (JSON format)
3. Understanding of the benchmark's evaluation criteria

---

## Step 1: Explore the Benchmark

First, understand the benchmark structure:

```bash
# Find the agent implementation
ls -la hal-harness/agents/ | grep -i <benchmark_name>

# Find the benchmark definition
ls -la hal-harness/hal/benchmarks/ | grep -i <benchmark_name>
```

Read and understand:
1. **Agent structure**: How does the agent work? What files does it have?
2. **Benchmark evaluation**: How are tasks evaluated? Docker? Direct execution?
3. **Common failure patterns**: What types of errors occur?

### Key Questions to Answer:
- What is the agent's entry point? (e.g., `main.py`, `agent.py`)
- What function is called? (e.g., `run`, `main.run`)
- Does it require Docker or special execution environment?
- What are the task IDs format?
- What does success/failure look like in traces?

---

## Step 2: Create the Rubric Template

Create a new rubric template file:

```bash
# Create the rubric file
touch rubric_templates/<benchmark_name>.txt
```

### Rubric Template Structure

Use this template and customize for your benchmark:

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

### 1. <Category 1 Name>
<Description of what this deficiency looks like>
- <Specific indicator>
- <Specific indicator>

### 2. <Category 2 Name>
<Description>
- <Specific indicator>

### 3. <Category 3 Name>
<Description>
- <Specific indicator>

[Add 4-6 categories specific to this benchmark]

---

## CRITICAL EXCLUSIONS: Agent Capability Issues (Score 0)

Do NOT classify the following as benchmark deficiencies:

### 1. <Agent Issue Category 1>
- <Specific example>
- <Specific example>
- These are agent capability issues, not benchmark defects

### 2. <Agent Issue Category 2>
- <Specific example>

[Add 4-6 categories of agent issues specific to this benchmark]

---

## Evidence Requirements

For Score 1, you MUST provide:
1. **Specific defect identification**: Quote the exact issue in the benchmark
2. **Impossibility proof**: Explain why NO agent could overcome this
3. **Exclusion of alternatives**: Show that no workaround exists

For Score 0, explain:
1. What the agent did wrong
2. What a capable agent could have done instead
3. Why the benchmark allowed for success

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
- <Pattern 1>
- <Pattern 2>
- <Pattern 3>

### Potential Benchmark Issues (Score 1):
- <Pattern 1>
- <Pattern 2>
- <Pattern 3>
```

### Tips for Writing Good Rubrics:
1. **Be specific**: Use concrete examples from actual traces
2. **Err on the side of agent issues**: Most failures ARE agent issues
3. **Require strong evidence for IFE**: The bar should be high
4. **Include benchmark-specific terminology**: Use terms agents/evaluators will recognize

---

## Step 3: Create Output Directory

```bash
mkdir -p rubrics_output/<benchmark_name>
```

---

## Step 4: Update CLAUDE.md

Add the new benchmark to CLAUDE.md:

### 4.1 Add to Supported Benchmarks Table

```markdown
| **<Benchmark Name>** | `agents/<agent_dir>/` | `rubric_templates/<benchmark>.txt` | Ready |
```

### 4.2 Add Benchmark-Specific Details Section

```markdown
### <Benchmark Name>
**Purpose**: <One line description>

**Agent**: `hal-harness/agents/<agent_dir>/`
- <Key feature 1>
- <Key feature 2>
- <Key feature 3>

**Common Issues**:
- <Issue 1>
- <Issue 2>
- <Issue 3>

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

### 4.3 Add to Quick Reference Table

```markdown
| <Benchmark Name> | `<benchmark>.txt` | <Key focus areas> |
```

---

## Step 5: Run Initial Evaluation

### 5.1 Collect Traces (if not already available)

```bash
cd hal-harness
hal-eval --benchmark <benchmark_name> \
    --agent_dir agents/<agent_dir>/ \
    --agent_function <function>.run \
    --agent_name "<Agent Name>" \
    -A model_name=gpt-4o
```

### 5.2 Run Rubric Evaluation

```bash
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-4o \
    --failed-only -y
```

### 5.3 Review Results

```bash
# Check rubric output
cat rubrics_output/<benchmark>/*.csv

# Count Grade=1 (potential IFEs)
grep ",1," rubrics_output/<benchmark>/*.csv | wc -l

# View specific Grade=1 tasks
grep ",1," rubrics_output/<benchmark>/*.csv
```

---

## Step 6: Aggregate Verdicts

```bash
python scripts/judge.py \
    --rubric-dir rubrics_output/<benchmark> \
    --output judge_output/<benchmark>_verdict.csv
```

---

## Step 7: Investigate Grade=1 Tasks

For each task marked as Grade=1:

1. **Read the trace**: Understand what happened
2. **Identify root cause**: Is it really a benchmark issue?
3. **Classify the fix type**:
   - Agent configuration (shims, patches, imports)
   - Prompt template updates
   - Rubric clarification

### Common Fix Types:

| Issue Type | Fix Location | Example |
|------------|--------------|---------|
| Deprecated API | `agent.py` | Add compatibility shim |
| Missing import | `agent.py` | Add to AUTHORIZED_IMPORTS |
| Operator not supported | `agent.py` | Patch interpreter |
| Code parsing | `agent.py` | Set `code_block_tags` |
| Missing guidance | `prompt_template.txt` | Add COMPATIBILITY NOTES |
| Grader confusion | `rubric_templates/*.txt` | Add CRITICAL EXCLUSION |

---

## Step 8: Apply Fixes

### 8.1 Agent Configuration Fixes

Edit the agent's main file (e.g., `agent.py`):

```python
# Example: Compatibility shim
try:
    from some_module import old_function
except ImportError:
    from some_module import new_function as old_function

# Example: Add to authorized imports
AUTHORIZED_IMPORTS = [
    "existing_import",
    "new_import",  # Added for task X
]
```

### 8.2 Prompt Template Fixes

Edit prompt templates:

```
COMPATIBILITY NOTES:
- <Guidance 1>
- <Guidance 2>
```

### 8.3 Rubric Clarifications

Add to rubric template:

```
## CRITICAL EXCLUSION: <Category>

Do NOT classify the following as benchmark deficiencies:
1. <Specific exclusion>
2. <Specific exclusion>
```

---

## Step 9: Re-run Evaluation

Use a new prefix to distinguish fixed runs:

```bash
# Re-run with fixes applied
hal-eval --benchmark <benchmark_name> \
    --agent_dir agents/<agent_dir>/ \
    --agent_function <function>.run \
    --agent_name "<Agent Name> (fixed)" \
    -A model_name=gpt-4o

# Re-evaluate with rubrics
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*_fixed_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-4o \
    --failed-only -y
```

---

## Step 10: Compare Before/After

```bash
# Count defects before
echo "Before:"
grep ",1," rubrics_output/<benchmark>/*baseline*.csv | wc -l

# Count defects after
echo "After:"
grep ",1," rubrics_output/<benchmark>/*fixed*.csv | wc -l
```

---

## Checklist

- [ ] Explored benchmark structure (agent, evaluation, task format)
- [ ] Created rubric template with benchmark-specific categories
- [ ] Created output directory
- [ ] Updated CLAUDE.md with benchmark details
- [ ] Ran initial rubric evaluation
- [ ] Aggregated verdicts
- [ ] Investigated Grade=1 tasks
- [ ] Applied fixes (agent config, prompts, rubric)
- [ ] Re-ran evaluation with fixes
- [ ] Compared before/after results

---

## Example: Adding "COL" Benchmark

```bash
# 1. Explore
ls hal-harness/agents/ | grep -i col
cat hal-harness/hal/benchmarks/col*.py

# 2. Create rubric
cat > rubric_templates/col.txt << 'EOF'
# COL Intrinsic Formation Error Detection Rubric
... (fill in based on benchmark specifics)
EOF

# 3. Create output dir
mkdir -p rubrics_output/col

# 4. Update CLAUDE.md (manually)

# 5. Run evaluation
python scripts/eval_rubric.py \
    --trace-file traces/col_*.json \
    --rubric rubric_templates/col.txt \
    --rubric-model openai:gpt-4o \
    --failed-only -y

# 6. Review
cat rubrics_output/col/*.csv
grep ",1," rubrics_output/col/*.csv
```

---

## Notes

- The unified schema (`rubric_templates/rubric.schema.json`) is automatically used
- No need to create benchmark-specific `.schema.json` files
- Run prefixes: `potato/kiwi` = before, `apple` = intermediate, `honey/tomato` = after
- Always preserve benchmark source code - only modify agent configs and prompts
