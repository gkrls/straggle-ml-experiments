#!/usr/bin/env bash
set -euo pipefail

# Where to put the prepared dataset
ROOT="$HOME/data/imagenet"          # change if needed
DOWNLOAD="true"                     # set to false if you've already downloaded the tarballs into $ROOT/raw

# URLs you gave me
TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

# --- prep dirs ---
mkdir -p "$ROOT"/{raw,train,val}
cd "$ROOT"

# --- 1) download (resume supported) ---
if [[ "${DOWNLOAD}" == "true" ]]; then
  echo "[*] Downloading ImageNet archives to $ROOT/raw ..."
  wget -c -O raw/ILSVRC2012_img_train.tar "$TRAIN_URL"
  wget -c -O raw/ILSVRC2012_img_val.tar "$VAL_URL"
  wget -c -O raw/ILSVRC2012_devkit_t12.tar.gz "$DEVKIT_URL"
else
  echo "[*] Skipping download (DOWNLOAD=false). Expecting archives in $ROOT/raw"
fi

# Quick sanity check
for f in raw/ILSVRC2012_img_train.tar raw/ILSVRC2012_img_val.tar raw/ILSVRC2012_devkit_t12.tar.gz; do
  [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
done

# --- 2) extract devkit (for meta + val ground truth) ---
echo "[*] Extracting devkit ..."
tar -xzf raw/ILSVRC2012_devkit_t12.tar.gz -C "$ROOT"
DEVKIT_DIR="$(find "$ROOT" -maxdepth 1 -type d -name 'ILSVRC2012_devkit_t12' -print -quit)"
[[ -n "${DEVKIT_DIR:-}" ]] || { echo "Devkit dir not found"; exit 1; }

# --- 3) extract TRAIN split ---
# Step 3a: this creates ~1000 per-class tar files (nXXXX.tar)
echo "[*] Extracting train meta-archive (this may take a while) ..."
tar -xf raw/ILSVRC2012_img_train.tar -C train

# Step 3b: expand each class tar into its folder, then remove the inner tar
echo "[*] Expanding per-class train tars ..."
# Use as many parallel jobs as CPU cores, fall back to 4 if nproc not present.
JOBS="${JOBS:-$(command -v nproc >/dev/null && nproc || echo 4)}"
find train -type f -name "*.tar" -print0 \
  | xargs -0 -I{} -P "${JOBS}" bash -c '
      set -e
      t="{}"
      d="${t%.tar}"
      mkdir -p "$d"
      tar -xf "$t" -C "$d"
      rm -f "$t"
    '

# --- 4) extract VAL split (unsorted) ---
echo "[*] Extracting validation images ..."
tar -xf raw/ILSVRC2012_img_val.tar -C val

# --- 5) sort VAL images into class folders using devkit mapping ---
echo "[*] Sorting validation images into class folders ..."
python3 - << 'PY'
import os, sys, shutil
from pathlib import Path

root = Path(os.environ.get("ROOT", "."))
devkit_dir = next(root.glob("ILSVRC2012_devkit_t12"))

# Files from the devkit
gt_file = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
meta_mat = devkit_dir / "data" / "meta.mat"

# We need scipy to read meta.mat
try:
    from scipy.io import loadmat  # type: ignore
except Exception:
    # Try to install it on the fly
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.io import loadmat  # type: ignore

import numpy as np

# Map: class_index(1..1000) -> wnid (e.g., "n01440764")
m = loadmat(str(meta_mat), squeeze_me=True, struct_as_record=False)
synsets = m["synsets"]
# synsets has 1000+ entries; the 1000 ILSVRC classes have ILSVRC2012_ID in 1..1000
id_to_wnid = {}
for s in np.atleast_1d(synsets):
    ilsvrc_id = int(getattr(s, "ILSVRC2012_ID", 0))
    if 1 <= ilsvrc_id <= 1000:
        id_to_wnid[ilsvrc_id] = str(getattr(s, "WNID"))

# Read ground-truth labels for val images (one label per line)
labels = [int(x.strip()) for x in open(gt_file, "r").read().splitlines()]
assert len(labels) == 50000, f"Expected 50k val labels, got {len(labels)}"

val_dir = root / "val"
# Val image names are 1..50000 in lexical order
# ILSVRC2012_val_00000001.JPEG, ..., ILSVRC2012_val_00050000.JPEG
def val_name(i):
    return f"ILSVRC2012_val_{i:08d}.JPEG"

moved = 0
for i, class_id in enumerate(labels, start=1):
    wnid = id_to_wnid.get(class_id)
    if wnid is None:
        raise RuntimeError(f"No WNID for class id {class_id}")
    dst_dir = val_dir / wnid
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = val_dir / val_name(i)
    if not src.exists():
        raise FileNotFoundError(f"Missing expected val image: {src.name}")
    shutil.move(str(src), str(dst_dir / src.name))
    moved += 1

print(f"Moved {moved} validation images into class folders.")
PY

# --- 6) basic summary ---
echo "[*] Summary:"
echo "  Train classes: $(find train -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "  Val classes:   $(find val   -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "  Train images:  $(find train -type f -name '*.JPEG' | wc -l)"
echo "  Val images:    $(find val   -type f -name '*.JPEG' | wc -l)"

echo "[✓] ImageNet prepared under: $ROOT"
echo "    Train path: $ROOT/train"
echo "    Val path:   $ROOT/val"
