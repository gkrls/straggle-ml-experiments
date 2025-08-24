#!/usr/bin/env bash
set -euo pipefail

# Usage: ./kill-session.sh <session> <host1> <host2> ... [<hostN>]
#    or: ./kill-session.sh <session> -f <hostfile>

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

echo "Killing session '$SESSION' on ${#HOSTS[@]} hosts..."

for HOST in "${HOSTS[@]}"; do
    echo -n "  $HOST... "
    
    if ssh "$HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
        ssh "$HOST" "tmux kill-session -t '$SESSION' 2>/dev/null"
        echo "✓"
    else
        echo "-"
    fi
done

echo "Done!"