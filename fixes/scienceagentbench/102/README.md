# Task 102 Fix: MODNet Model for Refractive Index Prediction

## Root Cause Analysis

**Task Requirement**: Train a MODNet (Material Optimal Descriptor Network) model for predicting the refractive index of materials using the provided training data, then predict refractive index for materials in the MP_2018.6 dataset.

**Identified IFE**:
1. The agent development sandbox (smolagents) blocks importing `modnet`, `pickle`, and even `glob` modules
2. **CRITICAL**: The Docker harness (`dockerfiles.py`) has NO special handling for MODNet, unlike DeepChem, Scanpy, etc.
3. The dataset files are pickled objects that require MODNet classes to deserialize (pandas.read_pickle fails with "No module named 'modnet'")

**Evidence from rubric evaluations**:
- "Import from modnet.models is not allowed"
- "Import of pickle is not allowed"
- "Import of glob is not allowed"
- "Could not read train.pkl with pandas: No module named 'modnet'" - showing the pickle contains MODNet-specific objects
- Even PyTorch falls back failed: "ImproperlyConfigured: Requested settings, but settings are not configured" (Django settings error from torch import)

## Analysis of Docker Evaluation Environment

**This is the most significant IFE among the four tasks.**

Unlike DeepChem and Scanpy which have special handling in `dockerfiles.py`, MODNet has NONE:

```bash
# DeepChem handling exists:
if echo "$extracted_pkgs" | grep -q 'deepchem'; then \
    /opt/miniconda3/bin/pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html; \
fi;

# Scanpy handling exists:
if echo "$extracted_pkgs" | grep -q 'scanpy'; then \
    echo 'scikit-misc' >> /testbed/instance_requirements.txt && \
    echo 'leidenalg' >> /testbed/instance_requirements.txt; \
fi;

# MODNet handling: DOES NOT EXIST
```

MODNet requires:
- `modnet` package from PyPI
- `pymatgen` for materials structure handling
- `matminer` for feature extraction
- TensorFlow/Keras backend

## Fix Applied

**Environment Override** (`env_override.json`): Adds MODNet and its dependencies to ensure proper installation in the Docker evaluation container.

**This fix requires corresponding changes to the Docker harness** to add MODNet-specific handling similar to DeepChem and Scanpy.

## Why This Preserves Task Difficulty

This fix does NOT:
- Provide hints about MODNet architecture or hyperparameters
- Pre-compute any material features
- Simplify the refractive index prediction task
- Give the agent the model configuration (300 input features, 128/64/32 neurons, 'elu' activation)

This fix ONLY ensures:
- MODNet package is installed in the evaluation environment
- The pickled MODNet data objects can be properly deserialized
- The task can be fairly evaluated if an agent produces correct code

## Expected Outcome

With MODNet properly installed, an agent that correctly:
1. Loads the MODNet training data from md_ref_index_train
2. Initializes a MODNetModel with specified architecture (300 features, 128/64/32 neurons, elu activation)
3. Trains the model
4. Predicts refractive index for MP_2018.6 dataset
5. Saves results to pred_results/ref_index_predictions_pred.csv

Should have their code properly evaluated rather than failing due to missing dependencies.

## Required Harness Modification

The Docker harness should be updated to include:

```bash
if echo "$extracted_pkgs" | grep -q 'modnet'; then \
    echo 'pymatgen' >> /testbed/instance_requirements.txt && \
    echo 'matminer' >> /testbed/instance_requirements.txt; \
fi;
```
