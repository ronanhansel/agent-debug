# Task 12: DAVIS DTI Drug Repurposing - IFE Analysis

## Task Description
Train a drug-target interaction model using the DAVIS dataset to predict binding affinities between antiviral drugs and COVID-19 targets using DeepPurpose library.

## Root Cause Analysis

### Docker Execution Error
```
ImportError: cannot import name 'models' from 'DeepPurpose'
(/opt/miniconda3/lib/python3.10/site-packages/DeepPurpose/__...)
```

### Analysis

**This IS an Intrinsic Formation Error (IFE)** - a Docker environment issue.

The error shows that DeepPurpose is installed but its `models` submodule cannot be imported. This is a known issue with the DeepPurpose package structure.

### Root Cause
DeepPurpose has an unusual package structure. The standard import `from DeepPurpose import utils, models` may fail due to:
1. Missing dependencies during installation (descriptastorus, etc.)
2. Import order issues within the package
3. PyTorch/DGL version incompatibilities

The Dockerfile attempts to install `descriptastorus` when DeepPurpose is detected:
```dockerfile
if echo "$extracted_pkgs" | grep -q 'DeepPurpose'; then \
    /opt/miniconda3/bin/pip install git+https://github.com/bp-kelley/descriptastorus; \
fi;
```

But this may not be sufficient.

## Fix Applied

### Environment Override (`env_override.json`)
Ensures proper DeepPurpose installation with all dependencies.

## Preserves Scientific Rigor
- The task still requires training a DTI model from scratch
- No hints about model architecture or hyperparameters
- No pre-computed embeddings or results
- Only fixes package installation, not the scientific problem

## Expected Outcome
After fix, `from DeepPurpose import utils, models` should work, allowing agents to implement the full DTI pipeline.
