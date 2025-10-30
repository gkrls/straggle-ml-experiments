#!/usr/bin/env bash
set -euo pipefail

# Usage: ./launch.sh [--keep] session host1 host2 ... -S script [args]
#    or: ./launch.sh [--keep] session host1 host2 ... -- command [args]
#    or: ./launch.sh [--keep] session -f hostfile -S script [args]
#    or: ./launch.sh [--keep] session -f hostfile -- command [args]

KEEP_OPEN=false
if [[ "${1:-}" == "--keep" || "${1:-}" == "--keep-open" ]]; then
    KEEP_OPEN=true
    shift
fi

# Check if next arg is -f (hostfile mode without session name)
if [[ "${1:-}" == "-f" ]]; then
    echo "Error: Session name required. Usage: $0 [--keep] <session> -f <hostfile> {-S <script>|-- <command>} [args...]"
    exit 1
fi

SESSION="${1:-}"
[[ -n "$SESSION" ]] || { echo "Usage: $0 [--keep] <session> {<hosts...>|-f <hostfile>} {-S <script>|-- <command>} [args...]"; exit 1; }
shift

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
    shift
else
    # Collect individual hosts
    while [[ $# -gt 0 && "$1" != "-S" && "$1" != "--" ]]; do
        HOSTS+=("$1")
        shift
    done
fi

[[ ${#HOSTS[@]} -gt 0 ]] || { echo "Error: No hosts specified"; exit 1; }
[[ $# -gt 0 ]] || { echo "Error: No -S or -- specified"; exit 1; }

MODE="$1"
shift

export WORLD_SIZE=${#HOSTS[@]}
echo "Launching on ${#HOSTS[@]} hosts: ${HOSTS[*]}"

for HOST in "${HOSTS[@]}"; do
    echo -n "  $HOST... "
    ssh "$HOST" "tmux kill-session -t '$SESSION' 2>/dev/null || true"
    
    if [[ "$MODE" == "-S" ]]; then
        SCRIPT="${1:-}"
        [[ -n "$SCRIPT" ]] || { echo "Error: No script specified"; exit 1; }
        ARGS="${*:2}"
        
        # Create unique remote script name
        REMOTE_SCRIPT="/tmp/$(basename "$SCRIPT").$$"
        scp -q "$SCRIPT" "$HOST:$REMOTE_SCRIPT"
        
        if [[ "$KEEP_OPEN" == "true" ]]; then
            ssh "$HOST" "tmux new-session -d -s '$SESSION' 'export WORLD_SIZE=$WORLD_SIZE; bash $REMOTE_SCRIPT $ARGS; rm -f $REMOTE_SCRIPT; exec bash'"
        else
            ssh "$HOST" "tmux new-session -d -s '$SESSION' 'export WORLD_SIZE=$WORLD_SIZE; bash $REMOTE_SCRIPT $ARGS; rm -f $REMOTE_SCRIPT'"
        fi
    else
        # -- mode
        COMMAND="$*"
        if [[ "$KEEP_OPEN" == "true" ]]; then
            ssh "$HOST" "tmux new-session -d -s '$SESSION' 'export WORLD_SIZE=$WORLD_SIZE; $COMMAND; exec bash'"
        else
            ssh "$HOST" "tmux new-session -d -s '$SESSION' 'export WORLD_SIZE=$WORLD_SIZE; $COMMAND'"
        fi
    fi
    
    echo "✓"
done

echo "Monitor: ssh HOST tmux attach -t $SESSION"