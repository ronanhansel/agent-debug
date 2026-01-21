#!/usr/bin/env python3
"""Test Weave query to diagnose why ColBench traces aren't found."""

import os
import sys

# Check for API key
if not os.environ.get("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY not set")
    print("Run: export WANDB_API_KEY=your_key_here")
    sys.exit(1)

import weave

# Initialize Weave
project = "ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming"
print(f"Initializing Weave with project: {project}...")
client = weave.init(project)

print(f"Testing Weave queries for project: {project}")
print("=" * 80)

# Test 1: Get all calls without filter
print("\nTest 1: Get all calls (limit 5)...")
try:
    all_calls = list(client.get_calls(limit=5))
    print(f"Found {len(all_calls)} calls")
    for i, call in enumerate(all_calls):
        attrs = call.attributes if hasattr(call, 'attributes') else {}
        run_id = attrs.get('run_id') if isinstance(attrs, dict) else None
        print(f"  Call {i+1}:")
        print(f"    id: {call.id if hasattr(call, 'id') else 'N/A'}")
        print(f"    op_name: {call.op_name if hasattr(call, 'op_name') else 'N/A'}")
        print(f"    attributes.run_id: {run_id}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Query with prefix
print("\nTest 2: Query with prefix 'col_ivy_gpt-4_1'...")
query = {
    "$expr": {
        "$contains": {
            "input": {"$getField": "attributes.run_id"},
            "substr": {"$literal": "col_ivy_gpt-4_1"},
            "case_insensitive": False,
        }
    }
}
try:
    filtered = list(client.get_calls(query=query, limit=5))
    print(f"Found {len(filtered)} calls with prefix 'col_ivy_gpt-4_1'")
    for i, call in enumerate(filtered):
        attrs = call.attributes if hasattr(call, 'attributes') else {}
        run_id = attrs.get('run_id') if isinstance(attrs, dict) else None
        print(f"  Call {i+1}: run_id={run_id}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check if run_id is in different location
print("\nTest 3: Inspecting first call structure...")
try:
    first_call = next(iter(client.get_calls(limit=1)))
    print(f"Call attributes: {dir(first_call)}")
    if hasattr(first_call, '__dict__'):
        print(f"Call dict keys: {list(first_call.__dict__.keys())}")
except Exception as e:
    print(f"Error: {e}")
