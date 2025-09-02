#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/datasets/openwebtext"
TAR="$ROOT/openwebtext.tar.xz"
ARC_DIR="$ROOT/openwebtext"       # will contain the many *_data.xz files
TXT_DIR="$ROOT/openwebtext-txt"  # final plain .txt files

# 1) Extract the OUTER tar.xz so we get $XZ_DIR with lots of *_data.xz
rm -rf "$ARC_DIR"
mkdir -p "$ROOT"
if tar --help 2>&1 | grep -q -- "-I"; then
  tar -I xz -xvf "$TAR" -C "$ROOT"
else
  xz -dc "$TAR" | tar -xv -C "$ROOT"
fi

# 2) Extract all INNER *.xz chunks (each is a tar.xz containing many .txt)
rm -rf "$TXT_DIR"
mkdir -p "$TXT_DIR"

if command -v parallel >/dev/null 2>&1; then
  # GNU parallel path
  find "$ARC_DIR" -maxdepth 1 -type f -name '*.xz' -print0 \
  | parallel -0 -j"$(nproc)" 'xz -dc {} | tar -x -C '"$TXT_DIR"
else
  # Portable xargs fallback
  find "$ARC_DIR" -maxdepth 1 -type f -name '*.xz' -print0 \
  | xargs -0 -n1 -P"$(nproc)" sh -c 'xz -dc "$0" | tar -x -C "'"$TXT_DIR"'"'
fi

# 3) Sanity checks
echo "TXT count: $(find "$TXT_DIR" -type f -name '*.txt' | wc -l)"
echo "Sample files:"; find "$TXT_DIR" -type f -name '*.txt' | head -n 5
