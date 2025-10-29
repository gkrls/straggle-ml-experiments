#!/usr/bin/env bash
set -euo pipefail

# one-line help
case "${1-}" in -h|--help) echo "Usage: $0 SRC DST"; exit 0;; esac

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 SRC DST" >&2
  exit 1
fi

SRC="$1"
DST="$2"

if [ ! -e "$SRC" ]; then
  echo "Error: source does not exist: $SRC" >&2
  exit 2
fi

# require rsync
if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync not found (install with: sudo apt install rsync)" >&2
  exit 3
fi

[ -e "$DST" ] && [ ! -d "$DST" ] && { echo "Error: $DST exists and is not a directory" >&2; exit 4; }
mkdir -p $DST

# do the copy
rsync -aH --info=progress2 --partial --inplace --whole-file -- "$SRC" "$DST"
