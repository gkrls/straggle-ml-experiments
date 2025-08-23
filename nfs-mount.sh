#!/usr/bin/env bash
# Usage: bash nfs_mount_min.sh <server> <mountpoint> [export_path=/]

set -euo pipefail

# one-liner help
[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { echo "Usage: $0 <server> <mountpoint> [path=/]"; exit 0; }

server="${1:?server required}"
mnt="${2:?mountpoint required}"
exp="${3:-/}"

sudo mkdir -p "$mnt"

# If something is already mounted at $mnt, do nothing.
if mountpoint -q "$mnt"; then
  echo "[*] Already mounted at $mnt; nothing to do."
  exit 0
fi

# Otherwise mount with your options.
sudo mount -t nfs -o rw,vers=4.2,nconnect=8,rsize=1048576,wsize=1048576,noatime \
  "$server:$exp" "$mnt"
echo "[✓] Mounted $server:$exp -> $mnt"
