# Task 74 - OGGM Glacier Area and Thickness Change

## Root Cause Analysis

**Error**: `ImportError: cannot import name 'distribute_2d' from 'oggm.core.flowline'`

The agent's generated code attempts to import `distribute_2d` from the wrong location:
```python
from oggm.core.flowline import distribute_2d  # WRONG
```

The correct import path is:
```python
from oggm.sandbox import distribute_2d  # CORRECT
```

### Why This Happened

1. **Agent Sandbox Limitation**: The smolagents CodeAgent runs in a sandboxed Python environment with a restricted import allowlist. This sandbox does NOT allow importing `oggm`, preventing the agent from testing its code during development.

2. **Incorrect Guess**: Without ability to test imports, the agent guessed `oggm.core.flowline` which seems logical but is incorrect. The `distribute_2d` module is actually in the `oggm.sandbox` namespace because it's experimental functionality.

### Is This an IFE?

**Partially Yes**: The agent sandbox restriction prevented the agent from discovering the correct import path through testing. This is an infrastructure limitation that affects all agents equally.

**However**: The OGGM documentation does document the correct import path, so a sufficiently capable agent could find this information without testing.

## Fix Applied

**Instruction Clarification**: Added a note about the correct OGGM API structure to help agents find the right import paths.

This fix is FAIR because:
- It doesn't give away the solution algorithm
- It only clarifies API structure that's publicly documented
- It compensates for the agent's inability to test imports in the sandbox

## Expected Outcome

After applying this fix, agents should be able to:
1. Find the correct import path for `distribute_2d`
2. Successfully generate glacier thickness plots
3. Pass the figure evaluation

## Files Changed

- `instruction_override.json`: Added clarification about OGGM's module structure
