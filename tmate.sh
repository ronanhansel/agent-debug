#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 {start <N>|list|stop <ID>|stop-all|clean}"
    echo ""
    echo "Commands:"
    echo "  start <N>   : Start N new tmate sessions (e.g., '$0 start 5')"
    echo "  list        : List all running tmate sessions and their SSH links"
    echo "  stop <ID>   : Stop a specific session ID (e.g., '$0 stop 3')"
    echo "  stop-all    : Stop all tmate sessions found in /tmp/"
    echo "  clean       : Force kill and remove all socket files (hard reset)"
    exit 1
}

# Function to get SSH link for a specific socket
get_ssh_link() {
    local socket=$1
    # We use env -u TMUX to ensure we don't conflict with current session
    env -u TMUX tmate -S "$socket" display -p '#{tmate_ssh}' 2>/dev/null
}

# 1. START N SESSIONS
cmd_start() {
    COUNT=${1:-1}
    echo "Starting $COUNT tmate sessions..."
    echo "------------------------------------------------------------------------------------------------"
    printf "%-5s | %-50s\n" "ID" "SSH Connection String"
    echo "------------------------------------------------------------------------------------------------"

    for i in $(seq 1 "$COUNT"); do
        SOCKET="/tmp/tmate-$i"
        
        # Start session in detached mode
        env -u TMUX tmate -S "$SOCKET" new-session -d 2>/dev/null
        
        # Wait for readiness
        env -u TMUX tmate -S "$SOCKET" wait tmate-ready 2>/dev/null
        
        # Fetch and print link
        SSH_LINK=$(get_ssh_link "$SOCKET")
        
        if [ -n "$SSH_LINK" ]; then
            printf "%-5s | %-50s\n" "$i" "$SSH_LINK"
        else
            printf "%-5s | %-50s\n" "$i" "Error: Could not start or retrieve link"
        fi
    done
    echo "------------------------------------------------------------------------------------------------"
}

# 2. LIST ALL RUNNING SESSIONS
cmd_list() {
    echo "Scanning for running sessions in /tmp/tmate-* ..."
    echo "------------------------------------------------------------------------------------------------"
    printf "%-15s | %-50s\n" "SOCKET ID" "SSH Connection String"
    echo "------------------------------------------------------------------------------------------------"
    
    found=0
    for SOCKET in /tmp/tmate-*; do
        if [ -S "$SOCKET" ]; then
            found=1
            # Extract ID from filename (e.g., /tmp/tmate-3 -> 3)
            ID=$(basename "$SOCKET" | sed 's/tmate-//')
            
            # Check if session is actually alive
            if env -u TMUX tmate -S "$SOCKET" has-session 2>/dev/null; then
                SSH_LINK=$(get_ssh_link "$SOCKET")
                if [ -z "$SSH_LINK" ]; then SSH_LINK="(No connection / Starting up)"; fi
                printf "%-15s | %-50s\n" "tmate-$ID" "$SSH_LINK"
            else
                printf "%-15s | %-50s\n" "tmate-$ID" "(Dead Socket - Recommend cleanup)"
            fi
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo "No running sessions found."
    fi
    echo "------------------------------------------------------------------------------------------------"
}

# 3. STOP ONE SESSION
cmd_stop_one() {
    ID=$1
    if [ -z "$ID" ]; then
        echo "Error: Please specify an ID to stop (e.g., '$0 stop 3')"
        exit 1
    fi
    
    SOCKET="/tmp/tmate-$ID"
    if [ -S "$SOCKET" ]; then
        env -u TMUX tmate -S "$SOCKET" kill-session 2>/dev/null
        echo "Session $ID (Socket: $SOCKET) stopped."
    else
        echo "Error: Session $ID not found."
    fi
}

# 4. STOP ALL SESSIONS
cmd_stop_all() {
    echo "Stopping all tmate sessions..."
    count=0
    for SOCKET in /tmp/tmate-*; do
        if [ -S "$SOCKET" ]; then
            env -u TMUX tmate -S "$SOCKET" kill-session 2>/dev/null
            echo " - Stopped: $SOCKET"
            count=$((count + 1))
        fi
    done
    
    if [ $count -eq 0 ]; then
        echo "No active sessions found to stop."
    else
        echo "Stopped $count sessions."
    fi
}

# 5. CLEANUP (Force Remove)
cmd_clean() {
    echo "Running hard cleanup..."
    cmd_stop_all
    rm -f /tmp/tmate-*
    echo "Cleanup complete. Removed all socket files in /tmp/tmate-*"
}

# Main Command Switch
case "$1" in
    start)
        cmd_start "$2"
        ;;
    list)
        cmd_list
        ;;
    stop)
        cmd_stop_one "$2"
        ;;
    stop-all)
        cmd_stop_all
        ;;
    clean)
        cmd_clean
        ;;
    *)
        usage
        ;;
esac