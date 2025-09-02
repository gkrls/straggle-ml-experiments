set -euo pipefail

git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
git -C "$HOME/straggle-ml-experiments" pull --ff-only || true

source ~/straggle-ml-experiments/venv/bin/activate

python ~/straggle-ml-experiments/datasets/openwebtext-to-parquet.py \
  --arc_dir  ~/datasets/openwebtext/openwebtext \
  --out_dir  ~/datasets/openwebtext/parquet \
  --docs_per_shard 300000 \
  --val_frac 0.01



  # DIR="${1:-$HOME/datasets/imagenet}"
# python3 $HOME/straggle-ml-experiments/datasets/imagenet-1k-prepare.py $DIR

# DATA_ROOT="${1:-$HOME/datasets/openwebtext}"
# LOADER_SRC="${2:-$HOME/straggle-ml-experiments/datasets/openwebtext_zenodo.py}"

# [ -d "$DATA_ROOT" ] || { echo "DATA_ROOT '$DATA_ROOT' not found" >&2; exit 1; }
# [ -f "$DATA_ROOT/openwebtext.tar.xz" ] || { echo "Missing $DATA_ROOT/openwebtext.tar.xz" >&2; exit 1; }
# [ -f "$LOADER_SRC" ] || { echo "Missing loader at $LOADER_SRC" >&2; exit 1; }

# # 1) extract shards into DATA_ROOT/openwebtext
# rm -rf "$DATA_ROOT/openwebtext"
# tar -xvf "$DATA_ROOT/openwebtext.tar.xz" -C "$DATA_ROOT"  # creates openwebtext/*.xz

# # 2) install loader at DATA_ROOT/openwebtext.py (basename matches directory)
# cp "$LOADER_SRC" "$DATA_ROOT/openwebtext.py"

# # 3) make cache dir (optional; training script will also create it)
# mkdir -p "$DATA_ROOT/openwebtext/cache"

# echo "Ready:"
# echo "  loader: $DATA_ROOT/openwebtext.py"
# echo "  shards: $DATA_ROOT/openwebtext/*.xz"
# echo "  cache:  $DATA_ROOT/openwebtext/cache"

