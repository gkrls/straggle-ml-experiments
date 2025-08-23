#!/usr/bin/env bash
# Usage: bash imagenet_prepare_min.sh /path/to/imagenet
# Expects in that folder:
#   ILSVRC2012_img_train.tar
#   ILSVRC2012_img_val.tar
#   ILSVRC2012_devkit_t12.tar.gz
# Produces:
#   train/<wnid>/*.JPEG
#   val/<wnid>/*.JPEG

set -euo pipefail

ROOT="${1:-.}"
mkdir -p "$ROOT/train" "$ROOT/val"

TRAIN_TAR="$ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$ROOT/ILSVRC2012_img_val.tar"
DEVKIT_TGZ="$ROOT/ILSVRC2012_devkit_t12.tar.gz"

[[ -f "$TRAIN_TAR" ]]  || { echo "[x] Missing $TRAIN_TAR"; exit 1; }
[[ -f "$VAL_TAR"   ]]  || { echo "[x] Missing $VAL_TAR"; exit 1; }
[[ -f "$DEVKIT_TGZ" ]] || { echo "[x] Missing $DEVKIT_TGZ"; exit 1; }

log() { echo "[*] $*"; }

# 1) Devkit
if ! find "$ROOT" -maxdepth 1 -type d -name 'ILSVRC2012_devkit*' | grep -q .; then
  log "Extracting devkit ..."
  tar -xzf "$DEVKIT_TGZ" -C "$ROOT"
else
  log "Devkit already extracted; skipping."
fi
DEVKIT_DIR="$(find "$ROOT" -maxdepth 1 -type d -name 'ILSVRC2012_devkit*' -print -quit)"
[[ -n "${DEVKIT_DIR:-}" ]] || { echo "[x] Devkit dir not found after extraction"; exit 1; }

# 2) Train
# 2a) Expand meta-archive into ~1000 per-class tar files
if ! find "$ROOT/train" -maxdepth 1 -type f -name '*.tar' | grep -q .; then
  if find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    log "Train class folders already present; skipping meta extraction."
  else
    log "Extracting train meta-archive (this will take a bit) ..."
    tar -xf "$TRAIN_TAR" -C "$ROOT/train"
  fi
else
  log "Per-class train tar files already present."
fi

# 2b) Expand each per-class tar into its folder, then remove the inner tar
if find "$ROOT/train" -type f -name '*.tar' | grep -q .; then
  JOBS="${JOBS:-$(command -v nproc >/dev/null && nproc || echo 4)}"
  log "Expanding per-class train tars with $JOBS jobs ..."
  find "$ROOT/train" -type f -name '*.tar' -print0 \
    | xargs -0 -n1 -P "$JOBS" bash -c '
        set -euo pipefail
        t="$1"; d="${t%.tar}"
        mkdir -p "$d"
        tar -xf "$t" -C "$d"
        rm -f "$t"
      ' _
  log "Train per-class expansion done."
else
  log "No per-class train tars to expand; assuming already done."
fi

# 3) Val (flat)
if compgen -G "$ROOT/val/ILSVRC2012_val_*.JPEG" >/dev/null; then
  log "Validation flat images already extracted; skipping."
elif find "$ROOT/val" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  log "Validation already sorted; skipping extraction."
else
  log "Extracting validation images ..."
  tar -xf "$VAL_TAR" -C "$ROOT/val"
fi

# 4) Sort val into class folders
if compgen -G "$ROOT/val/ILSVRC2012_val_*.JPEG" >/dev/null; then
  log "Sorting validation images using devkit mapping ..."
  ROOT_ENV="$ROOT" DEVKIT_DIR_ENV="$DEVKIT_DIR" python3 - << 'PY'
import os, sys, shutil
from pathlib import Path

root = Path(os.environ["ROOT_ENV"])
devkit = Path(os.environ["DEVKIT_DIR_ENV"])
val_dir = root / "val"

gt_file  = devkit / "data" / "ILSVRC2012_validation_ground_truth.txt"
meta_mat = devkit / "data" / "meta.mat"
if not gt_file.exists(): 
    sys.exit(f"[x] Missing {gt_file}")

# Try scipy, then h5py. If missing, try pip install quietly.
def ensure_mod(name):
    try:
        return __import__(name)
    except Exception:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])
            return __import__(name)
        except Exception:
            return None

id_to_wnid = {}

# First attempt: scipy.io.loadmat
scipy = ensure_mod("scipy")
if scipy:
    try:
        from scipy.io import loadmat
        m = loadmat(str(meta_mat), squeeze_me=True, struct_as_record=False)
        synsets = m.get("synsets", [])
        for s in synsets:
            try:
                ilsvrc_id = int(getattr(s, "ILSVRC2012_ID", 0))
                wnid = getattr(s, "WNID", None)
                if wnid and 1 <= ilsvrc_id <= 1000:
                    id_to_wnid[ilsvrc_id] = str(wnid)
            except Exception:
                pass
    except Exception:
        pass

# Second attempt: h5py (for v7.3 mat)
if len(id_to_wnid) < 1000:
    h5py = ensure_mod("h5py")
    if h5py:
        import h5py
        try:
            with h5py.File(str(meta_mat), "r") as f:
                synsets = f["synsets"]
                # synsets is a cell array -> iterate rows
                for ref in synsets[0]:
                    obj = f[ref[0]]
                    def read_str(field):
                        arr = obj[field]
                        if arr.dtype.kind == "O":
                            arr = f[arr[0][0]]
                        return "".join(chr(x) for x in arr[:].flatten())
                    def read_int(field):
                        return int(obj[field][:].squeeze())
                    try:
                        ilsvrc_id = read_int("ILSVRC2012_ID")
                        wnid = read_str("WNID")
                        if 1 <= ilsvrc_id <= 1000 and wnid:
                            id_to_wnid[ilsvrc_id] = wnid
                    except Exception:
                        continue
        except Exception:
            pass

if len(id_to_wnid) != 1000:
    sys.exit(f"[x] Could not build id->wnid mapping from meta.mat (got {len(id_to_wnid)}). "
             f"Install scipy or h5py in your Python env and re-run this step.")

labels = [int(x.strip()) for x in gt_file.read_text().splitlines()]
if len(labels) != 50000:
    sys.exit(f"[x] Expected 50000 val labels, got {len(labels)}")

def val_name(i): return f"ILSVRC2012_val_{i:08d}.JPEG"

moved = 0
for i, cls in enumerate(labels, start=1):
    wnid = id_to_wnid[cls]
    dst = val_dir / wnid
    dst.mkdir(exist_ok=True)
    src = val_dir / val_name(i)
    if src.exists():  # skip if already moved
        shutil.move(str(src), str(dst / src.name))
        moved += 1

print(f"Moved {moved} validation images into class folders.")
PY
else
  log "Validation already appears sorted; skipping sorting."
fi

# Summary
log "Summary:"
echo "  Train classes: $(find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "  Val classes:   $(find "$ROOT/val"   -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "  Train images:  $(find "$ROOT/train" -type f -iname '*.jpeg' | wc -l)"
echo "  Val images:    $(find "$ROOT/val"   -type f -iname '*.jpeg' | wc -l)"
echo "[✓] Done. Train = $ROOT/train  |  Val = $ROOT/val"
