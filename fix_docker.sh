#!/bin/bash
set -e

echo "=== Docker Storage Fix Script ==="
echo "Moving Docker to /Data to free up root filesystem"

# Step 1: Stop all running containers
echo ""
echo "Step 1: Stopping running containers..."
docker stop $(docker ps -q) 2>/dev/null || echo "No running containers"

# Step 2: Clean up containers
echo ""
echo "Step 2: Pruning containers..."
docker container prune -f

# Step 3: Clean up dangling images
echo ""
echo "Step 3: Pruning dangling images..."
docker image prune -f

# Step 4: Clean up build cache
echo ""
echo "Step 4: Pruning build cache..."
docker builder prune -af

# Step 5: Show current usage
echo ""
echo "Step 5: Current Docker disk usage:"
docker system df

# Step 6: Check disk space freed
echo ""
echo "Step 6: Current root filesystem:"
df -h /

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Now to move Docker to /Data, run these commands as root:"
echo ""
echo "  sudo systemctl stop docker"
echo "  sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak"
echo "  sudo bash -c 'cat > /etc/docker/daemon.json << EOF"
echo '{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "data-root": "/Data/docker"
}'
echo 'EOF'"
echo "  sudo mkdir -p /Data/docker"
echo "  sudo rsync -aP /new_dir_structure/docker/ /Data/docker/"
echo "  sudo systemctl start docker"
echo ""
echo "Or if /new_dir_structure/docker is empty, just:"
echo "  sudo systemctl stop docker"
echo "  # Edit /etc/docker/daemon.json to use /Data/docker"
echo "  sudo mkdir -p /Data/docker"
echo "  sudo systemctl start docker"
