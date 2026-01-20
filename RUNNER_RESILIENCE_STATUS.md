# Runner Script Resilience - Implementation Status

## ✅ COMPLETED

### SciCode (`scripts/run_scicode_fixes.py`)
**Fully Implemented** - Production ready

**Features:**
- ✅ Enhanced retry patterns (401/unauthorized/expired token detection)
- ✅ Exponential backoff (up to 3 retries, increasing timeouts)
- ✅ Failed task tracking (`.tmp/scicode_failed_tasks.json`)
- ✅ Restarts entire hal-eval process on token expiration
- ✅ Fresh MSAL token acquisition in new Docker container

**Key Patterns Detected:**
- 401, "unauthorized", "expired token", "invalid or expired token"
- 503, 504, 502 (TRAPI timeouts)
- 429 (rate limits)
- Connection errors, network errors

## 🔄 REMAINING WORK

Need to copy the same retry system to 4 scripts:

1. **`scripts/run_scienceagentbench_fixes.py`**
   - Add: FAILED_TASKS_FILE, is_retryable_error(), save_failed_task(), run_with_retry()
   - Replace: subprocess.run() call with run_with_retry()
   - Failed file: `.tmp/scienceagentbench_failed_tasks.json`

2. **`scripts/run_corebench_fixes.py`**
   - Same pattern as above
   - Failed file: `.tmp/corebench_failed_tasks.json`

3. **`scripts/run_colbench_fixes.py`**
   - Same pattern as above
   - Failed file: `.tmp/colbench_failed_tasks.json`

4. **`scripts/run_usaco_fixes.py`**
   - Same pattern as above
   - Failed file: `.tmp/usaco_failed_tasks.json`

## How It Solves Your Problem

**Your Concern:**
```
openai.AuthenticationError: Error code: 401 - 
{'statusCode': 401, 'message': 'TRAPI: Unauthorized. Invalid or expired token.'}
```

**Solution:**
1. Runner script detects "401" or "unauthorized" or "expired token" in subprocess output
2. Waits 2-7 seconds (with jitter to avoid thundering herd)
3. **Restarts entire hal-eval command**
4. hal-eval spawns **new Docker container**
5. Docker container acquires **fresh MSAL token** from ~/.azure/msal_token_cache.json
6. Task continues with valid token

**No manual intervention needed!**

## Quick Implementation Guide

For each of the 4 remaining scripts:

### Step 1: Add after `def log()` function

```python
FAILED_TASKS_FILE = REPO_ROOT / ".tmp" / "<benchmark>_failed_tasks.json"

# Copy is_retryable_error() from scicode script (lines 81-119)
# Copy save_failed_task() from scicode script (lines 122-134)
# Copy run_with_retry() from scicode script (lines 137-202)
```

### Step 2: Find and replace subprocess.run()

**Find this pattern:**
```python
result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, timeout=XXXX)
```

**Replace with:**
```python
success, error_msg, result = run_with_retry(
    cmd=cmd, env=env, cwd=REPO_ROOT,
    task_id=task_id, model_id=model_id,
    max_retries=3, base_timeout=XXXX,
)
if not success:
    log(f"Failed after retries: {error_msg}", "hal")
    return False, error_msg, None
```

## Testing

```bash
# Test SciCode (working now)
python scripts/run_scicode_fixes.py --list-fixes | head -5
# Should show: [INFO] Direct Azure mode: removed proxy URLs from environment

# After implementing others, test each:
python scripts/run_scienceagentbench_fixes.py --list-fixes | head -5
python scripts/run_corebench_fixes.py --list-fixes | head -5
python scripts/run_colbench_fixes.py --list-fixes | head -5
python scripts/run_usaco_fixes.py --list-fixes | head -5
```

## Current CoreBench Issue

Your CoreBench run is stuck because:
1. **Too many parallel capsules** (20) = ~60 Docker containers
2. Most containers are idle (0.01% CPU), waiting in queue
3. Only 3-4 containers actively working
4. Docker daemon overhead is significant

**Recommendation:**
```bash
# Kill current run
pkill -f "run_corebench_fixes.py"

# Restart with lower parallelism
python scripts/run_corebench_fixes.py \
    --all-models \
    --prefix cb_correct \
    --docker \
    --skip-rubrics \
    --max-parallel-capsules 5  # Reduced from 20
```

With 5 capsules, you'll have ~15 containers instead of 60, and most will be actively working. This will be **faster overall** due to reduced overhead.

