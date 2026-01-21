#!/usr/bin/env python3
"""Diagnose why ColBench Weave extraction finds 0 calls."""

import json
import os
import sys

if not os.environ.get("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY not set")
    sys.exit(1)

print("=" * 80)
print("COLBENCH WEAVE EXTRACTION DIAGNOSIS")
print("=" * 80)

# 1. Check local trace format
print("\n1. Checking local trace file structure...")
local_trace = "traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_15_colbench_backend_programming_20260120_174752_UPLOAD.json"
try:
    with open(local_trace) as f:
        data = json.load(f)

    config = data.get("config", {})
    print(f"   File: {local_trace}")
    print(f"   Config run_id: {config.get('run_id')}")
    print(f"   Has raw_logging_results: {len(data.get('raw_logging_results', []))} entries")
    print(f"   Has raw_eval_results: {data.get('raw_eval_results')}")
    print(f"   wandb_run_id in config: {config.get('wandb_run_id')}")
except Exception as e:
    print(f"   Error reading local trace: {e}")

# 2. Query Weave with different prefix patterns
import weave

project = "ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming"
print(f"\n2. Testing Weave queries on project: {project}")

try:
    client = weave.init(project)
    print("   ✓ Weave initialized")

    # Test query 1: Full exact prefix
    test_prefixes = [
        "col_ivy_gpt-4_1_2025-04-14",
        "col_ivy_gpt-4_1",
        "col_ivy",
    ]

    for prefix in test_prefixes:
        query = {
            "$expr": {
                "$contains": {
                    "input": {"$getField": "attributes.run_id"},
                    "substr": {"$literal": prefix},
                    "case_insensitive": False,
                }
            }
        }
        print(f"\n   Testing prefix: '{prefix}'")
        try:
            calls = list(client.get_calls(query=query, limit=3))
            print(f"     Found: {len(calls)} calls")
            if calls:
                for call in calls[:2]:
                    attrs = call.attributes if hasattr(call, 'attributes') else {}
                    run_id = attrs.get('run_id') if isinstance(attrs, dict) else None
                    print(f"       - run_id: {run_id}")
        except Exception as e:
            print(f"     Error: {e}")

    # Test: Get ANY calls without filter
    print(f"\n   Getting all calls (no filter, limit 5)...")
    try:
        all_calls = list(client.get_calls(limit=5))
        print(f"     Found: {len(all_calls)} calls")
        if all_calls:
            for call in all_calls[:3]:
                attrs = getattr(call, 'attributes', {})
                run_id = attrs.get('run_id') if isinstance(attrs, dict) else 'N/A'
                op_name = getattr(call, 'op_name', 'N/A')
                print(f"       - op: {op_name}, run_id: {run_id}")
    except Exception as e:
        print(f"     Error: {e}")

except Exception as e:
    print(f"   ✗ Weave initialization failed: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
