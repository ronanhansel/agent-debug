# Docker /Data Storage Solution

## Problem
Your root partition (`/`) is 100% full (432GB used of 432GB), causing Docker operations to fail. The issue is that Docker stores all its data (images, containers, build cache) in `/new_dir_structure/docker` which is on the root partition.

**Without sudo access**, you cannot change Docker's data root directory, but you can work around this issue.

## Your Storage Situation

| Partition | Size | Used | Available | Usage |
|-----------|------|------|-----------|-------|
| Root (`/`) | 432GB | 432GB | 0GB | **100%** ❌ |
| Data (`/Data`) | 14TB | 3.0TB | 11TB | 23% ✅ |

Docker is currently using ~400GB on root:
- Images: 168.7GB (108.2GB reclaimable)
- Containers: 213.4GB (40.6GB reclaimable)
- Volumes: 17.4GB (0.4GB reclaimable)

## Solution: Two-Part Fix

### Part 1: Free Up Space on Root (One-Time Cleanup)

Run the cleanup script to remove unused Docker resources:

```bash
cd /Data/home/v-qizhengli/workspace/agent-debug
./cleanup_docker.sh
```

This will:
1. Remove stopped containers (7 found)
2. Remove dangling images (4 found)
3. Optionally remove unused images (will prompt you)
4. Run Docker system prune

**Expected space freed:** ~150-200GB

### Part 2: Use /Data for Temporary Files (Always)

Instead of running your command directly:

```bash
# ❌ DON'T run this directly anymore:
python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs \
    --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 25
```

Use the wrapper script that redirects temp files to `/Data`:

```bash
# ✅ Run through wrapper instead:
./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs \
    --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 25
```

The wrapper script:
- Sets `TMPDIR=/Data/home/v-qizhengli/tmp` (redirects Python's tempfile to /Data)
- Shows disk usage before starting
- Warns if root partition is >95% full
- Runs your command with proper environment variables

## What Gets Fixed

✅ **Temporary directories**: Python's `tempfile.mkdtemp()` will use `/Data/tmp` instead of `/tmp`
✅ **Agent workspace copies**: 11TB of space available on `/Data`
✅ **Build directories**: Temp build directories go to `/Data`

⚠️ **Still on root** (unavoidable without sudo):
- Docker images storage
- Docker container storage
- Docker volumes

## Regular Maintenance

Run cleanup periodically when disk gets full:

```bash
# Quick check
df -h / /Data

# If root is >90%, clean up:
./cleanup_docker.sh
```

## Alternative: If You Get Sudo Access Later

If you get sudo access in the future, you can permanently fix this by moving Docker's data root:

```bash
# 1. Stop Docker
sudo systemctl stop docker

# 2. Create directory on /Data
sudo mkdir -p /Data/docker
sudo chown root:root /Data/docker

# 3. Update Docker config
sudo cp /Data/home/v-qizhengli/daemon.json /etc/docker/daemon.json

# 4. Optionally move existing data
sudo rsync -aP /new_dir_structure/docker/ /Data/docker/

# 5. Restart Docker
sudo systemctl start docker

# 6. Verify
docker info | grep "Docker Root Dir"
# Should show: Docker Root Dir: /Data/docker
```

## Testing the Solution

Test that everything works:

```bash
# 1. Check disk space
df -h / /Data

# 2. Run cleanup if needed
./cleanup_docker.sh

# 3. Test wrapper with a small job
./run_benchmark_with_data.sh python --version

# 4. Run your actual benchmark
./run_benchmark_with_data.sh python scripts/run_benchmark_fixes.py \
    --benchmark scicode --all-configs --all-tasks \
    --prefix scicode_moon1_ --docker \
    --parallel-models 10 --parallel-tasks 25
```

## Files Created

| File | Purpose |
|------|---------|
| `run_benchmark_with_data.sh` | Wrapper to run commands with /Data temp storage |
| `cleanup_docker.sh` | Clean up Docker to free root partition space |
| `daemon.json` | Docker config for future sudo use |
| `/Data/home/v-qizhengli/tmp/` | Temp directory on /Data partition |

## Monitoring Disk Usage

During long runs, monitor disk usage:

```bash
# Watch disk usage in real-time (updates every 2 seconds)
watch -n 2 'df -h / /Data && echo "" && docker system df'

# Check Docker usage
docker system df

# Check temp directory size
du -sh /Data/home/v-qizhengli/tmp
```

If root still fills up, you may need to:
1. Reduce `--parallel-models` or `--parallel-tasks`
2. Delete unused Docker images manually
3. Request sudo access to move Docker's data root permanently
