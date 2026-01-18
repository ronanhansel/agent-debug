# Task 64: OGGM Flowline Mass Balance Plotting Fix

## Root Cause Analysis

### Observed Failure
- **valid_program**: 0 (code failed to execute)
- **success_rate**: 0
- **Log**: Shows OGGM trying to download from `https://github.com/OGGM/oggm-sample-data/archive/...`

### Failure Reason
The agent's code triggered OGGM's default behavior of downloading sample data from GitHub, rather than using the locally provided pre-computed glacier data (`RGI60-11.00001.tar.gz`).

The execution log was cut off, but this typically indicates:
1. Network download timeout/failure in Docker
2. Missing configuration to point OGGM to local data directory
3. OGGM requires internet access for climate data even with local glacier data

### Is This an IFE?

**Yes** - This is an infrastructure/configuration issue:
1. The task provides local glacier data but doesn't specify how to configure OGGM to use it
2. OGGM's default behavior downloads data from the internet
3. Docker containers may have network restrictions
4. The domain knowledge doesn't explain OGGM configuration for offline/local data use

### Fix Type: Instruction Clarification + Environment

Add clarification about configuring OGGM to use local data and ensure network access.

## Why This Preserves Task Difficulty

- The scientific computation remains unchanged (calculate mass balance, create plot)
- The fix only provides configuration guidance, not solution implementation
- The agent still needs to understand OGGM API and create the correct visualization

## Expected Outcome

After clarification, agents should:
1. Configure OGGM working directory to point to the provided data
2. Or use cfg.PATHS to specify local directories
3. Extract and use the provided tar.gz data files
