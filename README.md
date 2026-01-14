# Agent debug

Setting up environment

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n hal python=3.12 -y
conda activate hal
```

Install all requirements (this will automatically install packages from submodules `docent` and `hal-harness`)

```bash
pip install -r requirements.txt
```

Build docker env for hal harness

```bash
cd hal-harness
docker build -t hal-agent -f hal/utils/docker/Dockerfile .
```

Make sure to place environment variables in place `hal-harness/` directory

```bash
cp hal-harness/.env.example .env
# edit .env to add your variables
```

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
    --rubric-model openai/o3-mini \
    --max-parallel 5 \
    --wandb-mode online
```

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
