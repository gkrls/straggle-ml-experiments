#!/usr/bin/env bash
# Usage: ./prep_imagenet.sh [IMAGENET_DIR]
# If IMAGENET_DIR not given, uses current directory.

set -euo pipefail

IMAGENET_DIR="${1:-.}"
cd "$IMAGENET_DIR"

TRAIN_TAR="ILSVRC2012_img_train.tar"
VAL_TAR="ILSVRC2012_img_val.tar"
DEVKIT_TGZ="ILSVRC2012_devkit_t12.tar.gz"

# Basic checks
for f in "$TRAIN_TAR" "$VAL_TAR" "$DEVKIT_TGZ"; do
  [[ -f "$f" ]] || { echo "Missing $f in $PWD"; exit 1; }
done

echo "[*] Removing old train/ and val/ (if any)..."
rm -rf train val devkit
mkdir -p train val devkit

echo "[*] Extracting train meta-archive..."
tar -xf "$TRAIN_TAR" -C train

echo "[*] Expanding per-class train tars into folders..."
# Example file: train/n01440764.tar -> train/n01440764/*
shopt -s nullglob
for t in train/*.tar; do
  d="${t%.tar}"
  mkdir -p "$d"
  tar -xf "$t" -C "$d"
  rm -f "$t"
done
shopt -u nullglob

echo "[*] Extracting validation images..."
tar -xf "$VAL_TAR" -C val

echo "[*] Extracting devkit..."
tar -xzf "$DEVKIT_TGZ" -C devkit

# Find devkit files (be flexible about exact path)
SYNSETS_FILE="$(find devkit -type f -name 'synset_words.txt' | head -n1 || true)"
GT_FILE="$(find devkit -type f -name 'ILSVRC2012_validation_ground_truth.txt' | head -n1 || true)"

if [[ -z "${SYNSETS_FILE}" || -z "${GT_FILE}" ]]; then
  echo "Could not find devkit mapping files in devkit/. Check your devkit archive."
  exit 1
fi

# Build a 1-based list of synset IDs in order
SYNSETS_LIST="synsets.txt"
awk '{print $1}' "$SYNSETS_FILE" > "$SYNSETS_LIST"

echo "[*] Sorting validation images into class folders (50,000 moves)..."
# Each line i of GT_FILE gives a class index k (1..1000) for image ILSVRC2012_val_000000i.JPEG
# We map k -> synset by taking line k of SYNSETS_LIST.
for i in $(seq 1 50000); do
  img="$(printf "ILSVRC2012_val_%08d.JPEG" "$i")"
  k="$(sed -n "${i}p" "$GT_FILE")"
  synset="$(sed -n "${k}p" "$SYNSETS_LIST")"
  mkdir -p "val/$synset"
  mv "val/$img" "val/$synset/"
done

echo "[*] Summary:"
printf "  Train classes: %s\n" "$(find train -mindepth 1 -maxdepth 1 -type d | wc -l)"
printf "  Val classes:   %s\n" "$(find val   -mindepth 1 -maxdepth 1 -type d | wc -l)"
printf "  Train images:  %s\n" "$(find train -type f -iname '*.JPEG' | wc -l)"
printf "  Val images:    %s\n" "$(find val   -type f -iname '*.JPEG' | wc -l)"

echo "[✓] Done. Train = $PWD/train  |  Val = $PWD/val"
