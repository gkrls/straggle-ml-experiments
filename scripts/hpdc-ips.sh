#!/usr/bin/env bash
set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

"$SCRIPT_DIR/hpdc-launch2.sh" hpdc-gnode 6 -S "$SCRIPT_DIR/ip.sh" --