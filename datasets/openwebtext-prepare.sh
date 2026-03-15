set -euo pipefail

git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
git -C "$HOME/straggle-ml-experiments" pull --ff-only || true

source ~/straggle-ml-experiments/venv/bin/activate

ARCHIVE_DIR="$HOME/datasets/openwebtext/openwebtext"
PARQUET_DIR="$HOME/datasets/openwebtext/parquet"
TOKENIZ_DIR="$HOME/datasets/openwebtext/tokenized"

FORCE=0
TOKEN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force|--clean) FORCE=1 ;;
    --token-only)    TOKEN_ONLY=1 ;;
  esac
done

# --- Parquet ---
if [ "$TOKEN_ONLY" -eq 1 ]; then
  echo "[openwebtext-to-parquet] SKIPPED (--token-only)"
elif [ "$FORCE" -eq 1 ]; then
  echo "[openwebtext-to-parquet] ..."
  rm -rf $PARQUET_DIR/train $PARQUET_DIR/val
  python ~/straggle-ml-experiments/datasets/openwebtext-to-parquet.py \
    --arc_dir $ARCHIVE_DIR --out_dir $PARQUET_DIR --docs_per_shard 300000 --val_frac 0.01
elif [ -d $PARQUET_DIR/train ] && [ -d $PARQUET_DIR/val ]; then
  echo "[openwebtext-to-parquet] SKIPPED: parquet/train and parquet/val already exist"
else
  echo "[openwebtext-to-parquet] ..."
  rm -rf $PARQUET_DIR/train $PARQUET_DIR/val
  python ~/straggle-ml-experiments/datasets/openwebtext-to-parquet.py \
    --arc_dir $ARCHIVE_DIR --out_dir $PARQUET_DIR --docs_per_shard 300000 --val_frac 0.01
fi

# --- Tokenize ---
if [ "$FORCE" -eq 1 ] || ! [ -f $TOKENIZ_DIR/train.bin ] || ! [ -f $TOKENIZ_DIR/val.bin ]; then
  echo "[openwebtext-tokenize] ..."
  rm -rf $TOKENIZ_DIR
  python ~/straggle-ml-experiments/datasets/openwebtext-tokenize.py \
    --data $PARQUET_DIR --out $TOKENIZ_DIR --num_proc 24
else
  echo "[openwebtext-tokenize] SKIPPED: tokenized/train.bin and tokenized/val.bin already exist"
fi

# set -euo pipefail

# git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
# git -C "$HOME/straggle-ml-experiments" pull --ff-only || true

# source ~/straggle-ml-experiments/venv/bin/activate

# ARCHIVE_DIR="$HOME/datasets/openwebtext/openwebtext"
# PARQUET_DIR="$HOME/datasets/openwebtext/parquet"
# TOKENIZ_DIR="$HOME/datasets/openwebtext/tokenized"

# if [ "${1:-}" = "--force" ] || [ "${1:-}" = "--clean" ]; then
#   echo "[prepare] Cleaning all generated data"
#   rm -rf $PARQUET_DIR/train $PARQUET_DIR/val $TOKENIZ_DIR
# fi

# if [ -d $PARQUET_DIR/train ] && [ -d $PARQUET_DIR/val ]; then
#   echo "[openwebtext-to-parquet] SKIPPED: parquet/train and parquet/val already exist"
# else
#   echo "[openwebtext-to-parquet] ..."
#   rm -rf $PARQUET_DIR/train $PARQUET_DIR/val
#   python ~/straggle-ml-experiments/datasets/openwebtext-to-parquet.py \
#     --arc_dir $ARCHIVE_DIR --out_dir $PARQUET_DIR --docs_per_shard 300000 --val_frac 0.01
# fi

# if [ -f $TOKENIZ_DIR/train.bin ] && [ -f $TOKENIZ_DIR/val.bin ]; then
#   echo "[openwebtext-tokenize] SKIPPED: tokenized/train.bin and tokenized/val.bin already exist"
# else
#   echo "[openwebtext-tokenize] ..."
#   rm -rf $TOKENIZ_DIR
#   python ~/straggle-ml-experiments/datasets/openwebtext-tokenize.py \
#     --data $PARQUET_DIR --out $TOKENIZ_DIR --num_proc 24
# fi


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

