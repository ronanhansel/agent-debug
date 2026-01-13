# Agent Trace Uploader for Lunette

This project uploads agent evaluation traces to [Lunette](https://lunette.dev/) for analysis and investigation.

## Overview

The script reads SWE-bench evaluation results from JSON files and uploads them as trajectories to Lunette, allowing you to:

- View agent traces in the Lunette web interface
- Analyze agent behavior and performance
- Debug failed tasks
- Compare different runs

## Setup

1. **Install dependencies:**

   ```bash
   source venv/bin/activate
   pip install lunette-sdk anthropic python-dotenv
   ```

2. **Configure API keys:**

   Create a `.env` file with your Lunette API key:

   ```bash
   LUNETTE_API_KEY="your-lunette-api-key"
   ```

   Get your API key from [lunette.dev](https://lunette.dev/)

## Usage

## Unified Evaluation + Debug Pipeline

Use the redesigned `main.py` to drive rubric evaluation, inspection, and reruns without juggling multiple commands:

```bash
# Full unattended run: evaluate traces, then run the inspector (default) or runner
python main.py pipeline \
  --trace-file traces/swebench_verified_mini_hal_generalist_agent_o3mini20250131_low_1744603516_UPLOAD.json \
  --rubrics-dir rubrics \
  --output-dir rubrics_output \
  --traces-dir traces \
  --agent-dir hal-harness/agents/hal_generalist_agent \
  --agent-args agent_args.azure.json \
  --benchmark-name swebench_verified_mini \
  --reasoning-effort medium \  # optional: applies to AutoInspector
  --debug-mode inspect \
  --yes
```

Add `--debug-mode run` when you want the pipeline to evaluate the traces *and* immediately replay any fixes in Docker. Every invocation logs to `hal-harness/log/pipeline-run-<timestamp>/pipeline.log`, so you can leave the process unattended and review the entire transcript later.

Key stages:

1. **Rubric evaluation (`main.py evaluate ...`)** – reads traces, runs Docent rubrics, and writes `rubrics_output/<rubric>.csv` (no timestamp). Use `--output-mode stdout` to check fixes without overwriting CSVs.
2. **Inspector (`main.py debug --debug-mode inspect ...`)** – generates structured guidance per task under `hal-harness/debug/<task>/<run>/inspection_report.json`.
3. **Runner (`main.py debug --debug-mode run ...`)** – replays fixes from `hal-harness/fixes/<task_id>/`, saves rerun summaries, and emits synthetic traces under `traces/debug_runs/` so the evaluator can recheck resolved tasks in “stdout only” mode.

Every debugger invocation emits a consolidated log folder at `hal-harness/log/pipeline-run-<timestamp>`; when chaining commands through `main.py pipeline`, both stages share the same execution context so you can copy/paste the resulting logs for reports.

## Output

You'll see output like:

```
📂 Loading trace data from swebench_verified_mini_hal_generalist_agent_o3mini20250131_low_1744603516_UPLOAD.json...

📊 Dataset Information:
   Agent: hal_generalist_agent_o3mini20250131_low
   Model: o3-mini-2025-01-31
   Benchmark: swebench_verified_mini

📈 Results Summary:
   Accuracy: 0.0%
   Total Cost: $11.08
   Failed Tasks: 50

🚀 Initializing Lunette tracer...
   Processing 840 trace entries...
   Found 50 unique tasks

📝 Creating trajectories for 50 tasks...
   ✓ Completed processing 50 tasks

☁️  Uploading traces to Lunette...

✅ Upload complete!
   Run ID: 8d6a4194-7ab5-4166-8404-395acc7e2a09
   Trajectory IDs: 50 trajectories

🔗 View your traces at: https://lunette.dev/
```

## Trace File Structure

The script expects JSON files in the `traces/` directory with the following structure:

```json
{
  "config": {
    "agent_name": "...",
    "benchmark_name": "...",
    "agent_args": {
      "model_name": "..."
    }
  },
  "results": {
    "accuracy": 0.0,
    "total_cost": 11.08,
    "failed_tasks": [...]
  },
  "raw_logging_results": [
    {
      "id": "...",
      "attributes": {
        "weave_task_id": "task-id"
      },
      "inputs": {
        "messages": [...]
      },
      "output": {...},
      "started_at": "...",
      "ended_at": "..."
    }
  ]
}
```

## How It Works

1. **Parse Trace Data**: The script reads the JSON file and extracts:

   - Agent configuration (name, model, benchmark)
   - Evaluation results (accuracy, cost)
   - Raw logging results with messages and outputs

2. **Group by Task**: Trace entries are grouped by `weave_task_id` to create one trajectory per unique task

3. **Create Trajectories**: For each task, a Lunette trajectory is created with:

   - Sample ID (task ID)
   - Metadata (number of entries, failure status)
   - Context for any API calls made within the trajectory

4. **Upload**: The tracer uploads all trajectories to Lunette and returns a run ID

## Lunette Documentation

- [Quickstart](https://docs.lunette.dev/quickstart/)
- [Tracer API](https://docs.lunette.dev/api/tracer/)
- [Trajectory API](https://docs.lunette.dev/api/trajectory/)
- [Sandbox API](https://docs.lunette.dev/api/sandbox/)

## Notes

- The script processes all tasks found in the trace file (default: 50 tasks)
- Each trajectory is created with metadata about the task
- No actual LLM API calls are made during upload - we're just organizing and uploading existing trace data
- The Lunette tracer automatically captures and formats the data for upload
  Steps to Build Docker Image

```bash
set -a; source .env; set +a
docker build -t hal-agent-runner:latest hal/utils/docker
```

Fix for Apple Silicon (M1/M2) Macs

1. Force the build to run as linux/amd64 (so the base image plus Miniconda match):

   cd /Users/ronan/Developer/agent-debug/hal-harness
   docker build --platform=linux/amd64 -t hal-agent-runner:latest hal/utils/docker
   This tells Docker Desktop to emulate x86 for the entire image. The build will be slower but succeeds without changing the Dockerfile.
   This tells Docker Desktop to emulate x86 for the entire image. The build will be slower but succeeds without changing the Dockerfile.

2. Stay on arm64 and install the arm64 Miniconda tarball by editing hal/utils/docker/Dockerfile:

   - RUN wget <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh> -O /tmp/miniconda.sh \
   - RUN wget <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh> -O /tmp/miniconda.sh \
      && bash /tmp/miniconda.sh -b -p /opt/conda \
      ...

   Rebuild normally (docker build -t hal-agent-runner:latest hal/utils/docker). This avoids x86 emulation entirely.

```bash
python scripts/auto_debug_batch.py \
        --rubrics-csv ../output/environmental_barrier_rubrics.csv \
        --traces-dir ../traces \
        --agent-dir agents/hal_generalist_agent \
        --agent-args ../agent_args.azure.json \
        --agent-function main.run \
        --benchmark-name swebench_verified
```

RUBRICS

```bash
cd docent/
pip install -e .
```
