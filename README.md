# Agent debug

Setting up environment

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n agent python=3.12 -y
conda activate agent
```

Install all requirements (this will automatically install packages from submodules `docent` and `hal-harness`)

```bash
pip install -r requirements.txt # From agent-debug
pip install -e ./docent      # Editable install of docent
pip install -e ./hal-harness  # Editable install of hal-harness
pip install --upgrade --force-reinstall certifi click
```

Build docker env for hal harness

```bash
cd hal-harness
docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .
```

Make sure to place environment variables in repo root (same directory you run the scripts from)

```bash
cp .env.example .env
# edit .env to add your variables
```

If you run from a different working directory, set `HAL_DOTENV_PATH=/path/to/.env`.

If Docker runs can upload W&B/Weave projects but fail calling the model with `Connection refused`, check:

- If you use a host-local OpenAI/LiteLLM proxy (e.g. `OPENAI_BASE_URL=http://localhost:4000`), Docker will see `localhost` as the container. Use either:
  - `HAL_DOCKER_NETWORK_MODE=host` (Linux) (alias: `HAL_DOCKER_NETWORK=host`), or
  - `OPENAI_BASE_URL=http://host.docker.internal:4000` (requires host gateway mapping; enabled by default in our runner).
- Enable a quick connectivity printout with `HAL_DOCKER_PREFLIGHT_NETWORK=1`.

Weave project naming in Docker:

- Docker runs upload traces to a single Weave/W&B project named `<prefix>_<benchmark>` (inferred from `run_id` + benchmark), and each task is recorded under an op named by `task_id` with attributes including `trace_file=<run_id>_UPLOAD.json`.

Reducing environment-barrier failures in Docker:

- The Docker image now preinstalls a baseline toolchain in conda `base`: `r-base`, `pandoc`, `rmarkdown/knitr`, `texlive-core`. This prevents common `Rscript: not found` / PDF-render failures that otherwise dominate CoreBench debugging.
- Agents should avoid `apt-get` inside tasks (CoreBench blocks it); use conda or task-provided environments instead.
- CoreBench capsule contents are staged under `/workspace/environment` inside the container (mirroring `/root/environment` in the original harness). The runner `chdir`s there before executing the agent so task-relative paths resolve correctly.

## To Grade Rubrics

Iterating through every `corebench_*_UPLOAD_*.json` pattern

```bash
for trace in traces/corebench_*_UPLOAD*.json; do
        /opt/homebrew/Caskroom/miniconda/base/envs/hal/bin/python \
          main.py evaluate \
          --trace-file "$trace" \
          --rubrics-dir rubrics \
          --output-dir rubrics_output \
          --rubric-model gpt-5.2 \
          --output-mode csv \
          --failed-only \
          --json-mode \
          --yes # Bypass confirmation screen rubrics evaluation
done
```

Merge generated rubrics

```bash
python scripts/merge_rubric_csvs.py \
      --rubrics-root rubrics_output \
      --output rubrics_output/merged_rubrics.csv --criteria environmental_barrier --criteria instruction_error --model-run-substring corebench_hard
```

print items with rubric score = 1 & correct = 0

```python
import csv
from pathlib import Path

csv_path = Path("rubrics_output/merged_rubrics.csv")
with csv_path.open() as fh:
    reader = csv.DictReader(fh)
    flagged = sorted(
        row["task_id"]
        for row in reader
        if row.get("criteria") == "environmental_barrier"
        and row.get("grade") == "1.00"
        and row.get("correct") == "0"
    )

for task in flagged:
    print(task)
```

## Generate fixing instruction (inspector)

```bash
./scripts/fixing_pipeline.sh \
    --benchmark-name corebench_hard \
    --inspector-model azure/o3-mini \
    --skip-runner \
    --skip-codex \
    --skip-rubric-eval
```

and sequentially fixing all of the rubric items using OpenAI `codex`, skipping rerun of the agent.

```bash
./scripts/fixing_pipeline.sh \
    --trace-dir traces \
    --rubrics-dir rubrics \
    --rubrics-output-dir rubrics_output \
    --benchmark-name corebench_hard \
    --skip-runner \
    --skip-rubric-eval \
    --skip-inspector
   #  --defer-rubric-eval \ # Run rubric eval after the rerun is completed
    --task-id capsule-2345790   # optional: repeat to limit to specific IDs, remove to fix all
```

- `--skip-runner`: skips re-running the agent after applying fixes
- `--skip-rubric-eval`: skips re-evaluating rubrics after applying fixes
- `--skip-inspector`: skips generating fixing instructions, useful if instructions are already generated

Running the corebench with fixes in docker. The runner model is defined in the agent args json file.

```bash
python scripts/run_corebench_fixes.py \
         --fixes-root fixes/corebench_hard \
         --agent-dir hal-harness/agents/hal_generalist_agent \
         --agent-args agent_args.azure.json \
         --rubric-model o3-mini \
         --benchmark corebench_hard \
         --skip-install \
         --docker \
         --task-id capsule-0851068 \
         --task-id capsule-0921079 \
         --task-id ...
```

downloading COREBench test set

```bash
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
      --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
```

When prompted for the passphrase, use: `reproducibility` (this is also hardcoded as the hint in hal-harness/hal/
benchmarks/corebench.py).

Rerun only the failed baselines from the mapping file using the fixed code withou

```bash
python scripts/master_rerun_corebench_fixes.py \
    --mapping-file model_to_baseline.json \
    --max-parallel 5 \
    --max-parallel-capsules 5 \
    --prefix lime \
    --wandb-mode online \
    --docker \
    --skip-rubrics
```

- `--max-parallel`: max number of parallel processes to run
- `--max-parallel-capsules`: max number of capsules to run in parallel per process (agent)

- `--prefix`: prefix for the wandb run name

```bash
python scripts/run_corebench_fixes.py \
    --fixes-root fixes/corebench_hard \
    --agent-dir hal-harness/agents/hal_generalist_agent \
    --agent-args agent_args.azure.json \
    --rubric-model openai/o3-mini \
    --benchmark corebench_hard \
    --task-id capsule-1394704 \
    --wandb-mode online \
    --keep-temp
```

## Note

Watch out for package conflicts when installing

If encountered TLS error, reinstall certifi, click

```bash
pip install --upgrade --force-reinstall certifi click
```

Make sure
