# Cloudflare SSH Tunnel Setup Guide

Set up persistent SSH access to any Linux/Mac server via Cloudflare Tunnel. No public IP needed, works behind NAT/firewall.

## Prerequisites

- A domain on Cloudflare (free tier works)
- Root/sudo access on the server
- `cloudflared` on your client machine (for connecting)

---

## Server Setup (Linux)

### Step 1: Install cloudflared

```bash
# Ubuntu/Debian
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared

# RHEL/CentOS/Fedora
sudo dnf install -y cloudflared
# or
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.rpm -o cloudflared.rpm
sudo rpm -i cloudflared.rpm

# Arch
yay -S cloudflared
```

### Step 2: Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

- Opens a browser URL
- Log into Cloudflare
- Select your domain
- Authorize the tunnel

Credentials saved to `~/.cloudflared/cert.pem`

### Step 3: Create a tunnel

```bash
cloudflared tunnel create my-ssh-tunnel
```

Note the **Tunnel ID** from the output (e.g., `9b3b1490-9e9b-4645-8dc1-e5dffc8af47d`)

### Step 4: Create config file

```bash
# Replace TUNNEL_ID with your actual tunnel ID
# Replace YOURDOMAIN.COM with your domain
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: TUNNEL_ID
credentials-file: /home/YOUR_USER/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: ssh.YOURDOMAIN.COM
    service: ssh://localhost:22
  - service: http_status:404
EOF
```

### Step 5: Create DNS route

```bash
cloudflared tunnel route dns my-ssh-tunnel ssh.YOURDOMAIN.COM
```

### Step 6: Install as system service

```bash
# Copy config to system location
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/
sudo cp ~/.cloudflared/*.json /etc/cloudflared/

# Update credentials path in config
sudo sed -i "s|/home/$USER/.cloudflared/|/etc/cloudflared/|g" /etc/cloudflared/config.yml

# Install and start service
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared

# Verify
sudo systemctl status cloudflared
```

---

## Server Setup (macOS)

### Step 1: Install cloudflared

```bash
brew install cloudflared
```

### Step 2-5: Same as Linux

Follow Steps 2-5 from Linux instructions above.

### Step 6: Install as launchd service

```bash
# Copy config
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/
sudo cp ~/.cloudflared/*.json /etc/cloudflared/

# Update paths
sudo sed -i '' "s|$HOME/.cloudflared/|/etc/cloudflared/|g" /etc/cloudflared/config.yml

# Install service
sudo cloudflared service install
sudo launchctl start com.cloudflare.cloudflared
```

---

## Client Setup (to connect)

### Install cloudflared on client

```bash
# macOS
brew install cloudflared

# Ubuntu/Debian
sudo apt install cloudflared

# Windows
# Download from: https://github.com/cloudflare/cloudflared/releases
```

### Configure SSH

Add to `~/.ssh/config`:

```
Host ssh.YOURDOMAIN.COM
    ProxyCommand cloudflared access ssh --hostname %h
    User YOUR_SERVER_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

### Generate SSH key (if needed)

```bash
ssh-keygen -t ed25519 -C "your-name"
```

### Add public key to server

On the **server**, add your public key:

```bash
echo 'YOUR_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Connect

```bash
ssh ssh.YOURDOMAIN.COM
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `cloudflared tunnel list` | List all tunnels |
| `cloudflared tunnel info NAME` | Show tunnel details |
| `cloudflared tunnel delete NAME` | Delete a tunnel |
| `sudo systemctl status cloudflared` | Check service status (Linux) |
| `sudo journalctl -u cloudflared -f` | View logs (Linux) |

---

## Troubleshooting

### Connection refused
```bash
# Check SSH is running
sudo systemctl status sshd

# Check cloudflared is running
sudo systemctl status cloudflared
```

### Password prompt despite SSH key
- Ensure username is correct (check with `whoami` on server)
- Some systems use full domain: `user@domain.com`
- Verify key is in `~/.ssh/authorized_keys` on server

### Tunnel not starting
```bash
# Check config syntax
cloudflared tunnel --config /etc/cloudflared/config.yml run

# Check credentials file exists
ls -la /etc/cloudflared/*.json
```

---

## Security Notes

- The tunnel credentials (`*.json` file) grant access to your tunnel - keep them secret
- Use SSH keys, not passwords
- Consider Cloudflare Access policies for additional auth (Zero Trust dashboard)
- Regularly rotate SSH keys

---

## Multiple Services (Optional)

You can expose multiple services through one tunnel:

```yaml
tunnel: TUNNEL_ID
credentials-file: /etc/cloudflared/TUNNEL_ID.json

ingress:
  - hostname: ssh.YOURDOMAIN.COM
    service: ssh://localhost:22
  - hostname: web.YOURDOMAIN.COM
    service: http://localhost:8080
  - hostname: api.YOURDOMAIN.COM
    service: http://localhost:3000
  - service: http_status:404
```

Then add DNS routes:
```bash
cloudflared tunnel route dns my-tunnel ssh.YOURDOMAIN.COM
cloudflared tunnel route dns my-tunnel web.YOURDOMAIN.COM
cloudflared tunnel route dns my-tunnel api.YOURDOMAIN.COM
```
