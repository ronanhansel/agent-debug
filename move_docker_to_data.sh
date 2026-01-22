#!/bin/bash
set -e

# This script moves Docker's data directory to /Data
# Run with sudo: sudo bash move_docker_to_data.sh

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash $0"
    exit 1
fi

echo "=== Moving Docker Data to /Data ==="

# Stop Docker
echo "Stopping Docker..."
systemctl stop docker
systemctl stop docker.socket 2>/dev/null || true

# Backup current config
echo "Backing up daemon.json..."
cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%Y%m%d%H%M%S)

# Create new data directory
echo "Creating /Data/docker..."
mkdir -p /Data/docker

# Update daemon.json
echo "Updating Docker config..."
cat > /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "data-root": "/Data/docker"
}
EOF

# If old docker dir has data, move it
OLD_DOCKER="/new_dir_structure/docker"
if [ -d "$OLD_DOCKER" ] && [ "$(ls -A $OLD_DOCKER 2>/dev/null)" ]; then
    echo "Moving data from $OLD_DOCKER to /Data/docker..."
    rsync -aP "$OLD_DOCKER/" /Data/docker/
    echo "Old data moved. You can delete $OLD_DOCKER after verifying."
else
    echo "No existing Docker data to move (or directory empty)."
fi

# Start Docker
echo "Starting Docker..."
systemctl start docker

# Verify
echo ""
echo "=== Verification ==="
docker info | grep -E "(Docker Root Dir|Storage Driver)"
echo ""
df -h / /Data

echo ""
echo "=== Done! Docker now using /Data/docker ==="
