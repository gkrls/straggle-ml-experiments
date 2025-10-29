#!/usr/bin/env bash
set -euo pipefail

# Usage: ./check-status.sh <session> <host1> <host2> ... [<hostN>]
#    or: ./check-status.sh <session> -f <hostfile>

[[ $# -ge 2 ]] || { echo "Usage: $0 <session> {<hosts...>|-f <hostfile>}"; exit 1; }

SESSION="$1"; shift
HOSTS=()

# Check if using hostfile
if [[ "${1:-}" == "-f" ]]; then
    shift
    HOSTFILE="${1:-}"
    [[ -n "$HOSTFILE" ]] || { echo "Error: No hostfile specified after -f"; exit 1; }
    [[ -f "$HOSTFILE" ]] || { echo "Error: Hostfile '$HOSTFILE' not found"; exit 1; }
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]] && HOSTS+=("$line")
    done < "$HOSTFILE"
else
    HOSTS=("$@")
fi

echo "Checking session '$SESSION' on ${#HOSTS[@]} hosts:"
echo

for HOST in "${HOSTS[@]}"; do
    echo "=== $HOST ==="
    
    # Check if session exists
    if ssh "$HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
        echo "✓ Session exists"
        
        # Show last 10 lines
        echo "  Last 10 lines:"
        LAST_OUTPUT=$(ssh "$HOST" "tmux capture-pane -t '$SESSION' -p 2>/dev/null" || echo "Could not capture output")
        echo "$LAST_OUTPUT" | tail -10 | sed 's/^/    /'
        
    else
        echo "✗ No session (either completed or failed)"
    fi
    echo
done