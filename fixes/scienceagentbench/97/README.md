# Task 97 Fix: DeepChem CGCNN Model for Formation Energy Prediction

## Root Cause Analysis

**Task Requirement**: Train a formation energy prediction model using DeepChem's CGCNN (Crystal Graph Convolutional Neural Network) model on perovskite structure data stored in `.pkl` files.

**Identified IFE**: The agent development sandbox (smolagents) blocks importing `deepchem` and `pickle` modules, preventing agents from developing and testing code that uses these required libraries. All 4 models (GPT-4.1, O3, O4-mini-high, O4-mini-low) failed with the same import restriction errors.

**Evidence from rubric evaluations**:
- "Import of deepchem is not allowed. Authorized imports are: [...]"
- "Import of pickle is not allowed. Authorized imports are: [...]"
- Dataset files are `.pkl` format requiring unpickling with DeepChem classes

## Analysis of Docker Evaluation Environment

The Docker harness (`dockerfiles.py`) already has special handling for DeepChem:
```bash
if echo "$extracted_pkgs" | grep -q 'deepchem'; then \
    /opt/miniconda3/bin/pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html; \
fi;
```

This means IF an agent can produce code that imports deepchem, the Docker evaluation will properly install DGL for CGCNN.

## Fix Applied

**Environment Override** (`env_override.json`): Ensures deepchem and required dependencies are properly installed in the Docker evaluation container.

## Why This Preserves Task Difficulty

This fix does NOT:
- Provide hints about how to solve the scientific problem
- Simplify the CGCNN model architecture
- Pre-compute any results
- Reduce the complexity of the formation energy prediction task

This fix ONLY ensures:
- The required packages are available in the evaluation environment
- The task can be fairly evaluated if an agent produces correct code

## Expected Outcome

With proper package availability, an agent that correctly:
1. Loads the pickled NumpyDataset
2. Initializes a DeepChem CGCNNModel
3. Trains the model
4. Generates predictions

Should have their code properly evaluated rather than failing due to missing dependencies.

## Note on Agent Sandbox

The agent sandbox (smolagents) import restrictions are a separate infrastructure issue that prevents agents from developing code interactively. This fix addresses the Docker evaluation environment only.

## Implementation Notes

For this fix to take effect, the Docker harness (`dockerfiles.py`) should be modified to pre-install deepchem and DGL unconditionally for tasks that require them, OR the agent sandbox should be configured to allow these imports.

Currently, the harness installs deepchem dependencies only when `pipreqs` detects `deepchem` in the agent's submitted code - but agents cannot write code with deepchem imports because their sandbox blocks the import during development.

**Recommended Docker harness change**: Add task-specific package pre-installation based on task requirements, not just code analysis.
