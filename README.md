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

Run the script to upload traces:

```bash
python main.py
```

The script will:

1. Load the trace data from `traces/` directory
2. Extract task information, messages, and metadata
3. Create trajectories for each unique task
4. Upload everything to Lunette
5. Display a summary with the run ID

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
