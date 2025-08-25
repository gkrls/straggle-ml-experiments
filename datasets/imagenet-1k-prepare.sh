set -euo pipefail

git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
git -C "$HOME/straggle-ml-experiments" pull --ff-only || true

DIR="${1:-$HOME/datasets/imagenet}"
python3 $HOME/straggle-ml-experiments/datasets/imagenet-1k-prepare.py $DIR