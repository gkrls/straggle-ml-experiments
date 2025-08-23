#!/usr/bin/env bash
# Usage: ./prepare_imagenet.sh /path/to/imagenet/root
# Expects these files downloaded:
#   ILSVRC2012_img_train.tar
#   ILSVRC2012_img_val.tar
#   ILSVRC2012_devkit_t12.tar.gz
#
# It will create:
#   $ROOT/train/<wnid>/*.JPEG
#   $ROOT/val/<wnid>/*.JPEG
#
# Idempotent: safe to re-run; it will skip finished steps.

set -euo pipefail

ROOT="${1:-.}"
RAW_DIR="${RAW_DIR:-$ROOT/raw}"
mkdir -p "$ROOT" "$RAW_DIR" "$ROOT/train" "$ROOT/val"

# --- helpers ---
log() { echo "[*] $*"; }
warn() { echo "[!] $*" >&2; }
die() { echo "[x] $*" >&2; exit 1; }

resolve() {
  # find a file by name in ROOT or RAW_DIR
  local name="$1"
  for d in "$ROOT" "$RAW_DIR"; do
    if [[ -f "$d/$name" ]]; then
      printf '%s/%s\n' "$d" "$name"
      return 0
    fi
  done
  return 1
}

# Better CPU core detection across systems
JOBS="${JOBS:-$(
  getconf _NPROCESSORS_ONLN 2>/dev/null || \
  nproc 2>/dev/null || \
  sysctl -n hw.ncpu 2>/dev/null || \
  echo 4
)}"

# --- locate raw archives ---
TRAIN_TAR="${TRAIN_TAR:-$(resolve ILSVRC2012_img_train.tar || true)}"
VAL_TAR="${VAL_TAR:-$(resolve ILSVRC2012_img_val.tar || true)}"
DEVKIT_TGZ="${DEVKIT_TGZ:-$(resolve ILSVRC2012_devkit_t12.tar.gz || true)}"

[[ -n "${TRAIN_TAR:-}" ]] || die "Missing ILSVRC2012_img_train.tar (place in $ROOT or $RAW_DIR)"
[[ -n "${VAL_TAR:-}"   ]] || die "Missing ILSVRC2012_img_val.tar (place in $ROOT or $RAW_DIR)"
[[ -n "${DEVKIT_TGZ:-}" ]] || die "Missing ILSVRC2012_devkit_t12.tar.gz (place in $ROOT or $RAW_DIR)"

# --- 1) extract devkit (for meta + val ground truth) ---
if ! ls -d "$ROOT"/ILSVRC2012_devkit* >/dev/null 2>&1; then
  log "Extracting devkit ..."
  tar -xzf "$DEVKIT_TGZ" -C "$ROOT"
else
  log "Devkit already extracted; skipping."
fi

DEVKIT_DIR=""
for d in "$ROOT"/ILSVRC2012_devkit*; do
  [[ -d "$d" ]] && DEVKIT_DIR="$d" && break
done
[[ -n "$DEVKIT_DIR" ]] || die "Devkit directory not found after extraction."

# --- 2) extract TRAIN split ---
# Step 2a: expand the meta-archive into ~1000 per-class tar files
if ! find "$ROOT/train" -maxdepth 1 -type f -name '*.tar' | grep -q .; then
  if find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    log "Train per-class folders already present; skipping meta-archive extraction."
  else
    log "Extracting train meta-archive (this may take a while) ..."
    tar -xf "$TRAIN_TAR" -C "$ROOT/train"
  fi
else
  log "Per-class train tar files already present; skipping re-extract."
fi

# Step 2b: expand each per-class tar into its folder, then remove the inner tar
if find "$ROOT/train" -type f -name '*.tar' | grep -q .; then
  log "Expanding per-class train tars with $JOBS parallel jobs ..."
  find "$ROOT/train" -type f -name "*.tar" -print0 \
    | xargs -0 -n1 -P "$JOBS" bash -c '
        set -euo pipefail
        t="$1"
        d="${t%.tar}"
        mkdir -p "$d"
        tar -xf "$t" -C "$d"
        rm -f "$t"
      ' _
  log "Finished expanding per-class train tars."
else
  log "No per-class train tars to expand; assuming already done."
fi

# --- 3) extract VAL split (unsorted) ---
# Only extract if the flat images are not already present and val subdirs are empty
if compgen -G "$ROOT/val/ILSVRC2012_val_*.JPEG" >/dev/null; then
  log "Validation flat images already extracted; skipping."
elif find "$ROOT/val" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  log "Validation appears already sorted; skipping extraction."
else
  log "Extracting validation images ..."
  tar -xf "$VAL_TAR" -C "$ROOT/val"
fi

# --- 4) sort VAL images into class folders ---
# We only sort if the flat images exist in $ROOT/val (i.e., not already organized)
if compgen -G "$ROOT/val/ILSVRC2012_val_*.JPEG" >/dev/null; then
  log "Sorting validation images into class folders ..."
  ROOT_ENV="$ROOT" DEVKIT_DIR_ENV="$DEVKIT_DIR" python3 - << 'PY'
import os, sys, shutil
from pathlib import Path

root = Path(os.environ["ROOT_ENV"])
devkit_dir = Path(os.environ["DEVKIT_DIR_ENV"])
val_dir = root / "val"

gt_file = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
map_file = devkit_dir / "data" / "map_clsloc.txt"  # preferred; text mapping
meta_mat = devkit_dir / "data" / "meta.mat"        # fallback; matlab struct

if not gt_file.exists():
    raise SystemExit(f"Missing ground truth: {gt_file}")

# Build mapping: class_index (1..1000) -> wnid (e.g. 'n01440764')
id_to_wnid = {}

def try_map_clsloc():
    # Accept either "wnid idx" OR "idx wnid" per line.
    mapping = {}
    with map_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a.startswith("n") and b.isdigit():
                mapping[int(b)] = a
            elif a.isdigit() and b.startswith("n"):
                mapping[int(a)] = b
    return mapping

def try_meta_mat():
    # Best-effort fallback using scipy or h5py if installed.
    # Avoids forcing installs in a cluster environment.
    try:
        from scipy.io import loadmat
        m = loadmat(str(meta_mat), squeeze_me=True, struct_as_record=False)
        synsets = m.get("synsets", [])
        mapping = {}
        for s in synsets:
            ilsvrc_id = int(getattr(s, "ILSVRC2012_ID", 0))
            wnid = getattr(s, "WNID", None)
            if wnid is not None and 1 <= ilsvrc_id <= 1000:
                mapping[ilsvrc_id] = str(wnid)
        return mapping
    except Exception:
        pass
    try:
        import h5py
        mapping = {}
        with h5py.File(str(meta_mat), "r") as f:
            # Heuristic for v7.3 HDF5 .mat
            # synsets is usually a cell array; we traverse and read fields
            synsets = f["synsets"]
            # synsets is a cell -> array of object refs
            for ref in synsets[0]:
                obj = f[ref[0]]
                # Fields might be stored by index; we try common names
                def read_str(field):
                    arr = obj[field]
                    # strings are stored as bytes with nested refs
                    if arr.dtype.kind == "O":
                        arr = f[arr[0][0]]
                    return "".join(chr(x) for x in arr[:].flatten())
                def read_int(field):
                    return int(obj[field][:].squeeze())
                try:
                    ilsvrc_id = read_int("ILSVRC2012_ID")
                    wnid = read_str("WNID")
                    if 1 <= ilsvrc_id <= 1000 and wnid:
                        mapping[ilsvrc_id] = wnid
                except Exception:
                    continue
        return mapping
    except Exception:
        return {}

if map_file.exists():
    id_to_wnid = try_map_clsloc()

if not id_to_wnid:
    if meta_mat.exists():
        id_to_wnid = try_meta_mat()

if len(id_to_wnid) != 1000:
    # Proceed anyway but warn; many setups still succeed
    print(f"[warn] id_to_wnid has {len(id_to_wnid)} entries (expected 1000)", file=sys.stderr)

labels = [int(x.strip()) for x in gt_file.read_text().splitlines()]
if len(labels) != 50000:
    raise SystemExit(f"Expected 50k val labels, got {len(labels)}")

def val_name(i: int) -> str:
    return f"ILSVRC2012_val_{i:08d}.JPEG"

moved = 0
missing = 0
for i, cls in enumerate(labels, start=1):
    wnid = id_to_wnid.get(cls)
    if not wnid:
        raise SystemExit(f"No WNID for class id {cls}. Mapping incomplete.")
    src = val_dir / val_name(i)
    if not src.exists():
        # Might already be sorted; skip silently but count
        missing += 1
        continue
    dst_dir = val_dir / wnid
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_dir / src.name))
    moved += 1

print(f"Moved {moved} validation images into class folders.")
if missing:
    print(f"{missing} files were already moved (or missing).")
PY
else
  log "Validation images already appear sorted; skipping sorting."
fi

# --- 5) basic summary ---
log "Summary:"
train_classes=$(find "$ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
val_classes=$(find "$ROOT/val"   -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
train_images=$(find "$ROOT/train" -type f \( -iname '*.jpeg' -o -iname '*.jpg' \) | wc -l | tr -d ' ')
val_images=$(find "$ROOT/val"     -type f \( -iname '*.jpeg' -o -iname '*.jpg' \) | wc -l | tr -d ' ')

echo "  Train classes: $train_classes"
echo "  Val classes:   $val_classes"
echo "  Train images:  $train_images"
echo "  Val images:    $val_images"

echo "[✓] ImageNet prepared under: $ROOT"
echo "    Train path: $ROOT/train"
echo "    Val path:   $ROOT/val"
