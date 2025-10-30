#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./kill-session.sh <session> {<hosts...>|-f <hostfile>}
#   ./kill-session.sh --all     {<hosts...>|-f <hostfile>}
#
# <session>: kill one session by name
# --all/-a : kill ALL tmux sessions (tmux kill-server) on each host

[[ $# -ge 2 ]] || { echo "Usage: $0 <session|--all> {<hosts...>|-f <hostfile>}"; exit 1; }

TARGET="$1"; shift
KILL_ALL=false
if [[ "$TARGET" == "--all" || "$TARGET" == "-a" ]]; then
  KILL_ALL=true
else
  SESSION="$TARGET"
fi

HOSTS=()
if [[ "${1:-}" == "-f" ]]; then
  shift
  HOSTFILE="${1:-}"
  [[ -n "$HOSTFILE" ]] || { echo "Error: No hostfile specified after -f"; exit 1; }
  [[ -f "$HOSTFILE" ]] || { echo "Error: Hostfile '$HOSTFILE' not found"; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]] && HOSTS+=("$line")
  done < "$HOSTFILE"
else
  HOSTS=("$@")
fi

if $KILL_ALL; then
  echo "Killing ALL tmux sessions on ${#HOSTS[@]} hosts..."
else
  echo "Killing session '$SESSION' on ${#HOSTS[@]} hosts..."
fi

for HOST in "${HOSTS[@]}"; do
  printf "  %s... " "$HOST"
  if $KILL_ALL; then
    # Only try to kill if a server is running
    if ssh "$HOST" "tmux list-sessions >/dev/null 2>&1"; then
      ssh "$HOST" "tmux kill-server >/dev/null 2>&1"
      echo "✓"
    else
      echo "-"
    fi
  else
    if ssh "$HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
      ssh "$HOST" "tmux kill-session -t '$SESSION' 2>/dev/null"
      echo "✓"
    else
      echo "-"
    fi
  fi
done

echo "Done!"
