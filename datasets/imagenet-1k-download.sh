#!/usr/bin/env bash
# Usage: ./imagenet-1k-download.sh /path/to/dir

set -euo pipefail

TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
N_PARTS="${N_PARTS:-12}"
SPLIT_SIZE_MIN="${SPLIT_SIZE_MIN:-1M}"

TARGET_DIR="${1:-.}"
mkdir -p "$TARGET_DIR"

download() {
  local url="$1" out="$2"
  echo "$url -> $out"
  # -x/-s = connections, -k = min split size, -c = resume, -d/-o set dest
  aria2c -c -x"$N_PARTS" -s"$N_PARTS" -k"$SPLIT_SIZE_MIN" \
         -d "$(dirname "$out")" -o "$(basename "$out")" \
         "$url"
}


download "$TRAIN_URL" "$TARGET_DIR/ILSVRC2012_img_train.tar"
download "$VAL_URL" "$TARGET_DIR/ILSVRC2012_img_val.tar"
download "$DEVKIT_URL" "$TARGET_DIR/ILSVRC2012_devkit_t12.tar.gz"

echo "All done."