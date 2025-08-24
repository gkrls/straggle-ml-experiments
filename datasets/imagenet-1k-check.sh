#!/usr/bin/env bash
# super quick check: count class folders in train/ and val/
# Usage: $0 <IMAGENET_ROOT> [expected_classes=1000]
# Set STRICT=1 to return non-zero if counts ≠ expected.

set -euo pipefail
case "${1-}" in -h|--help) echo "Usage: $0 <IMAGENET_ROOT> [expected_classes=1000]"; exit 0;; esac

ROOT="${1:?Usage: $0 <IMAGENET_ROOT> [expected_classes=1000]}"
EXP="${2:-1000}"

TRAIN="$ROOT/train"
VAL="$ROOT/val"

[ -d "$TRAIN" ] || { echo "[x] Missing $TRAIN"; exit 1; }
[ -d "$VAL"   ] || { echo "[x] Missing $VAL"; exit 1; }

t=$(find "$TRAIN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
v=$(find "$VAL"   -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

echo "Train class dirs: $t"
echo "Val class dirs:   $v"

if [ "${STRICT:-0}" = "1" ]; then
  [ "$t" -eq "$EXP" ] && [ "$v" -eq "$EXP" ] && exit 0 || exit 1
fi
