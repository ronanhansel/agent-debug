#!/bin/bash
# Docker cleanup script to free up root partition space
# Run this BEFORE starting your benchmark to free space

set -e

echo "=========================================="
echo "Docker Disk Usage BEFORE Cleanup:"
echo "=========================================="
docker system df
echo ""
df -h / | grep -E "Filesystem|/dev"
echo ""

echo "=========================================="
echo "Step 1: Removing stopped containers..."
echo "=========================================="
STOPPED=$(docker ps -a -q -f status=exited -f status=created)
if [ -n "$STOPPED" ]; then
    docker rm $STOPPED
    echo "Removed $(echo $STOPPED | wc -w) stopped containers"
else
    echo "No stopped containers to remove"
fi
echo ""

echo "=========================================="
echo "Step 2: Removing dangling images..."
echo "=========================================="
DANGLING=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING" ]; then
    docker rmi $DANGLING
    echo "Removed dangling images"
else
    echo "No dangling images to remove"
fi
echo ""

echo "=========================================="
echo "Step 3: Removing unused images (optional)..."
echo "=========================================="
echo "This will remove images not used by any container."
echo "Your hal-agent-runner images will be preserved."
read -p "Remove unused images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Only remove images NOT named hal-agent-runner
    docker images --format "{{.Repository}}:{{.Tag}}\t{{.ID}}" | \
        grep -v "hal-agent-runner" | \
        grep -v "<none>" | \
        awk '{print $2}' | \
        xargs -r docker rmi 2>/dev/null || true
    echo "Removed unused images (kept hal-agent-runner)"
fi
echo ""

echo "=========================================="
echo "Step 4: Pruning system..."
echo "=========================================="
docker system prune -f
echo ""

echo "=========================================="
echo "Docker Disk Usage AFTER Cleanup:"
echo "=========================================="
docker system df
echo ""
df -h / | grep -E "Filesystem|/dev"
echo ""
echo "Cleanup complete!"
