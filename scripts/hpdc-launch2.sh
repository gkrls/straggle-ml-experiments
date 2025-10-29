#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 <node_prefix> <1-6> -- <remote command...>"
  echo "  $0 <node_prefix> <1-6> -S <local_script> -- [script args...]"
  exit 1
}

[[ $# -ge 2 ]] || usage
PREFIX="$1"; N="$2"; shift 2
[[ "$N" =~ ^[1-6]$ ]] || { echo "Error: N must be 1..6"; exit 2; }

LOCAL_SCRIPT=""
if [[ "${1-}" == "-S" ]]; then
  LOCAL_SCRIPT="$2"; shift 2
  [[ -r "$LOCAL_SCRIPT" ]] || { echo "Cannot read: $LOCAL_SCRIPT"; exit 2; }
fi
[[ "${1-}" == "--" ]] || usage; shift
ARGS=( "$@" )

LOGDIR="$(mktemp -d -t nodes-run-XXXXXXXX)"
CMDLOG="$LOGDIR/commands.txt"
echo "# commands sent per host" > "$CMDLOG"

pids=(); hosts=(); ranks=()
for ((i=1; i<=N; i++)); do
  host="${PREFIX}${i}"
  node="$i"; rank=$((i-1))
  log="$LOGDIR/${host}.log"

  if [[ -n "$LOCAL_SCRIPT" ]]; then
    echo "$host (RANK=$rank): bash -l -s -- ${ARGS[*]} < $LOCAL_SCRIPT" | tee -a "$CMDLOG"
    ssh "$host" "NODE=$node RANK=$rank bash -l -s -- ${ARGS[*]@Q}" < "$LOCAL_SCRIPT" > "$log" 2>&1 &
  else
    echo "$host (RANK=$rank): bash -l -c ${ARGS[*]}" | tee -a "$CMDLOG"
    ssh "$host" "NODE=$node RANK=$rank bash -l -c ${ARGS[*]@Q}" >"$log" 2>&1 &
  fi
  pids+=($!)
  hosts+=("$host")
  ranks+=("$rank")
done

rc=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  host="${hosts[$idx]}"
  rank="${ranks[$idx]}"
  log="$LOGDIR/${host}.log"
  if wait "$pid"; then
    echo "  -> OK   $host (RANK=$rank, log: $log)"
  else
    echo "  -> FAIL $host (RANK=$rank, log: $log)"
    rc=1
  fi
done

echo "Logs: $LOGDIR"
exit "$rc"
