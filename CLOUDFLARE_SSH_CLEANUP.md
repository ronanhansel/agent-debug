# Cloudflare SSH Tunnel Cleanup Guide

This machine has a persistent SSH tunnel configured via Cloudflare.

## Current Setup

| Component | Value |
|-----------|-------|
| Hostname | `ssh.doanchu.com` |
| Tunnel ID | `9b3b1490-9e9b-4645-8dc1-e5dffc8af47d` |
| Tunnel Name | `persistent-ssh` |
| Service | `cloudflared.service` (systemd) |
| Config | `/etc/cloudflared/config.yml` |

## How to Connect (while active)

```bash
# Requires cloudflared on client machine
ssh ssh.doanchu.com
```

---

## CLEANUP INSTRUCTIONS

### Step 1: Stop and disable the service

```bash
sudo systemctl stop cloudflared
sudo systemctl disable cloudflared
```

### Step 2: Remove the tunnel from Cloudflare

```bash
# Delete the tunnel (this also removes DNS record)
cloudflared tunnel delete persistent-ssh
```

### Step 3: Remove local files

```bash
# Remove config and credentials
sudo rm -rf /etc/cloudflared/
rm -rf ~/.cloudflared/

# Remove the systemd service file
sudo rm /etc/systemd/system/cloudflared.service
sudo systemctl daemon-reload
```

### Step 4: (Optional) Uninstall cloudflared

```bash
sudo apt remove cloudflared
sudo rm /etc/apt/sources.list.d/cloudflared.list
sudo rm /usr/share/keyrings/cloudflare-main.gpg
```

### Step 5: Remove authorized SSH key

```bash
# Remove the key for "ronan" from authorized_keys
sed -i '/ronan$/d' ~/.ssh/authorized_keys
```

### Step 6: Clean up Cloudflare Dashboard

1. Go to https://dash.cloudflare.com/
2. Select domain: `doanchu.com`
3. Go to DNS > Records
4. Delete CNAME record for `ssh` (if still exists)
5. Go to Zero Trust > Networks > Tunnels
6. Delete tunnel `persistent-ssh` (if still exists)

---

## One-Liner Cleanup (run as the user)

```bash
sudo systemctl stop cloudflared && sudo systemctl disable cloudflared && cloudflared tunnel delete persistent-ssh && sudo rm -rf /etc/cloudflared/ && rm -rf ~/.cloudflared/ && sudo rm -f /etc/systemd/system/cloudflared.service && sudo systemctl daemon-reload && sed -i '/ronan$/d' ~/.ssh/authorized_keys && echo "Cleanup complete"
```

---

## Notes

- The tunnel was created on: 2026-01-20
- Setup by: v-tatruong (with Claude Code assistance)
- Domain owner must also revoke the tunnel token from Cloudflare dashboard for full cleanup
