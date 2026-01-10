import asyncio
import json
from pathlib import Path
import os
import dotenv

dotenv.load_dotenv()

# Check for required API keys
if not os.getenv("LUNETTE_API_KEY"):
    print("⚠️  Warning: LUNETTE_API_KEY not set. Get your API key from https://lunette.dev/")
    print("   Set it with: export LUNETTE_API_KEY='your-api-key-here'")
    print("   Or add it to a .env file")
    print()


def extract_task_info(entry):
    """Extract relevant information from a trace entry."""
    task_id = entry.get("attributes", {}).get("weave_task_id", "unknown")
    messages_data = entry.get("inputs", {}).get("messages", [])
    output = entry.get("output", {})
    started_at = entry.get("started_at", "")
    ended_at = entry.get("ended_at", "")
    
    return {
        "task_id": task_id,
        "messages": messages_data,
        "output": output,
        "started_at": started_at,
        "ended_at": ended_at,
        "entry": entry
    }


async def main():
    # Find all trace files
    traces_dir = Path("traces")
    trace_files = list(traces_dir.glob("*.json"))
    
    if not trace_files:
        print("❌ No trace files found in traces/ directory")
        return
    
    print(f"📂 Found {len(trace_files)} trace file(s):")
    for tf in trace_files:
        print(f"   • {tf.name}")
    
    # For now, process the first file (can be extended to process all)
    trace_file = trace_files[0]
    print(f"\n📂 Processing: {trace_file.name}")
    
    print(f"   Loading trace data...")
    with open(trace_file, "r") as f:
        data = json.load(f)
    
    # Extract config information
    config = data.get("config", {})
    agent_name = config.get("agent_name", "hal_generalist_agent")
    model_name = config.get("agent_args", {}).get("model_name", "o3-mini-2025-01-31")
    benchmark_name = config.get("benchmark_name", "swebench_verified_mini")
    
    print(f"\n📊 Dataset Information:")
    print(f"   Agent: {agent_name}")
    print(f"   Model: {model_name}")
    print(f"   Benchmark: {benchmark_name}")
    
    # Extract evaluation results
    results = data.get("results", {})
    accuracy = results.get("accuracy", 0)
    total_cost = results.get("total_cost", 0)
    failed_tasks = results.get("failed_tasks", [])
    
    print(f"\n📈 Results Summary:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Total Cost: ${total_cost:.2f}")
    print(f"   Failed Tasks: {len(failed_tasks)}")
    
    # Process raw logging results
    raw_logging_results = data.get("raw_logging_results", [])
    print(f"   Processing {len(raw_logging_results)} trace entries...")
    
    # Group trace entries by task
    tasks_by_id = {}
    for entry in raw_logging_results:
        task_info = extract_task_info(entry)
        task_id = task_info["task_id"]
        if task_id and task_id != "unknown":
            if task_id not in tasks_by_id:
                tasks_by_id[task_id] = []
            tasks_by_id[task_id].append(task_info)
    
    print(f"   Found {len(tasks_by_id)} unique tasks")
    
    # Analyze the traces locally without making API calls
    print(f"\n📊 Analyzing Traces:")
    print(f"   Total trace entries: {len(raw_logging_results)}")
    print(f"   Unique tasks: {len(tasks_by_id)}")
    
    # Sample analysis of first few tasks
    print(f"\n🔍 Sample Task Details:")
    for idx, (task_id, task_entries) in enumerate(list(tasks_by_id.items())[:5]):
        print(f"\n   Task {idx + 1}: {task_id}")
        print(f"      Status: {'❌ Failed' if task_id in failed_tasks else '✅ Success'}")
        print(f"      Trace entries: {len(task_entries)}")
        
        # Get message count from first entry
        if task_entries:
            first_entry = task_entries[0]
            messages_data = first_entry.get("messages", [])
            print(f"      Messages in first entry: {len(messages_data)}")
            
            # Show first message if available
            if messages_data:
                first_msg = messages_data[0]
                content = first_msg.get("content", "")
                if isinstance(content, list) and content:
                    preview = content[0].get("text", "")[:100] if isinstance(content[0], dict) else str(content[0])[:100]
                else:
                    preview = str(content)[:100]
                print(f"      First message preview: {preview}...")
    
    print(f"\n📋 Summary:")
    print(f"   ✓ Successfully parsed {len(raw_logging_results)} trace entries")
    print(f"   ✓ Found {len(tasks_by_id)} unique tasks")
    print(f"   • Model: {model_name}")
    print(f"   • Benchmark: {benchmark_name}")
    print(f"   • Accuracy: {accuracy:.1%}")
    print(f"   • Total Cost: ${total_cost:.2f}")
    print(f"   • Successful Tasks: {len([t for t in tasks_by_id.keys() if t not in failed_tasks])}")
    print(f"   • Failed Tasks: {len(failed_tasks)}")
    
    # Upload to Lunette if API key is set
    if os.getenv("LUNETTE_API_KEY"):
        # Check for Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not azure_endpoint or not azure_api_key:
            print(f"\n⚠️  Error: Azure OpenAI configuration missing.")
            print(f"   Required environment variables:")
            print(f"   - AZURE_OPENAI_ENDPOINT (e.g., https://your-resource.cognitiveservices.azure.com/)")
            print(f"   - AZURE_OPENAI_API_KEY")
            print(f"   - AZURE_OPENAI_DEPLOYMENT (e.g., gpt-4o-mini) [optional, defaults to gpt-4o-mini]")
            print(f"   - AZURE_OPENAI_API_VERSION [optional, defaults to 2024-12-01-preview]")
            return
        
        from lunette import LunetteTracer
        from openai import AzureOpenAI
        
        print(f"\n🚀 Initializing Lunette tracer...")
        tracer = LunetteTracer(task=benchmark_name, model=model_name)
        
        # Initialize Azure OpenAI client
        print(f"   Connecting to Azure OpenAI...")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {azure_deployment}")
        openai_client = AzureOpenAI(
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )
        
        # Process tasks and create trajectories
        max_tasks = min(1, len(tasks_by_id))  # Start with 1 task to test
        print(f"   Creating trajectories for {max_tasks} tasks...")
        
        for idx, (task_id, task_entries) in enumerate(list(tasks_by_id.items())[:max_tasks]):
            if idx > 0 and idx % 5 == 0:
                print(f"   Progress: {idx}/{max_tasks} tasks uploaded...")
            
            # Create a trajectory for this task
            async with tracer.trajectory(
                sample=task_id,
                metadata={
                    "num_entries": len(task_entries),
                    "task_id": task_id,
                    "failed": task_id in failed_tasks,
                    "agent": agent_name
                }
            ):
                # Process all entries from this task to create the complete trajectory
                for entry_idx, task_entry in enumerate(task_entries):  # Process all entries
                    messages_data = task_entry.get("messages", [])
                    
                    if not messages_data:
                        continue
                    
                    # Convert to OpenAI message format
                    openai_messages = []
                    for msg in messages_data:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        
                        # Handle content that can be a list or string
                        if isinstance(content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            text_content = "\n".join(text_parts) if text_parts else str(content)
                        else:
                            text_content = str(content)
                        
                        if role in ["user", "assistant", "system"]:
                            openai_messages.append({
                                "role": role,
                                "content": text_content  # Full content for complete traces
                            })
                    
                    # Only make API call if we have valid messages
                    if openai_messages and len(openai_messages) >= 2:
                        try:
                            # Filter out system messages for the API call
                            api_messages = [m for m in openai_messages if m["role"] != "system"]
                            if not api_messages:
                                continue
                            
                            # Make the API call - this will be captured by Lunette
                            response = openai_client.chat.completions.create(
                                model=azure_deployment,
                                max_completion_tokens=100,  # Keep it minimal for cost
                                messages=api_messages,  # All messages for complete trace
                            )
                            # Response is automatically captured by Lunette via OpenTelemetry
                        except Exception as e:
                            # Continue on error (rate limits, etc.)
                            if idx == 0 and entry_idx == 0:
                                print(f"   Note: API call error (continuing): {str(e)[:50]}")
                            pass
        
        print(f"   ✓ Completed processing {max_tasks} tasks")
        
        # Close tracer and upload
        print(f"\n☁️  Uploading traces to Lunette...")
        result = await tracer.close()
        
        print(f"\n✅ Upload complete!")
        if result and result.get('run_id'):
            print(f"   Run ID: {result['run_id']}")
            print(f"   Trajectory IDs: {len(result.get('trajectory_ids', []))} trajectories")
            print(f"\n🔗 View your traces at: https://lunette.dev/")
        else:
            print(f"   Warning: No run_id returned. Check if trajectories were created.")
    else:
        print(f"\n💡 To upload to Lunette for visualization:")
        print(f"   1. Get your API key from https://lunette.dev/")
        print(f"   2. Set Azure OpenAI environment variables")
        print(f"   3. Run the script again")
        print(f"\n   Example .env file:")
        print(f"   LUNETTE_API_KEY=your-lunette-key")
        print(f"   AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/")
        print(f"   AZURE_OPENAI_API_KEY=your-azure-api-key")
        print(f"   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        print(f"   AZURE_OPENAI_API_VERSION=2024-12-01-preview")


if __name__ == "__main__":
    asyncio.run(main())