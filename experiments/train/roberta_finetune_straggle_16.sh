#!/usr/bin/env bash
set -euo pipefail

BRANCH="wip-simple"

# export PKG_CONFIG_PATH=/opt/mellanox/dpdk/23.11/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig

# Sync the repos if needed
if [[ $# -ge 1 && "$1" == "sync" ]]; then
  shift

  [[ -d "$HOME/straggle-ml/.git" ]] || git clone https://github.com/gkrls/straggle-ml.git "$HOME/straggle-ml"

  git -C "$HOME/straggle-ml" fetch -q origin || true
  git -C "$HOME/straggle-ml" checkout "$BRANCH" 2>/dev/null || true
  git -C "$HOME/straggle-ml" reset --hard origin/"$BRANCH" 2>/dev/null || true #git -C "$DIR" reset --hard || true
  git -C "$HOME/straggle-ml" pull --ff-only origin "$BRANCH" 2>/dev/null || true
  git -C "$HOME/straggle-ml" clean -ffd || true

  if [ ! -d "$HOME/straggle-ml-experiments/.git" ]; then
    git clone https://github.com/gkrls/straggle-ml-experiments.git "$HOME/straggle-ml-experiments"
  else
    git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
    git -C "$HOME/straggle-ml-experiments" pull --ff-only || true
  fi

  mkdir -p $HOME/straggle-ml/build
  cd $HOME/straggle-ml/build
  cmake -DCMAKE_INSTALL_MESSAGE=LAZY \
        -DCMAKE_BUILD_TYPE=Release \
        -DDPA_TRACE=OFF \
        -DDPA_DEVELOP=OFF \
        -DDPA_SWITCH=OFF \
        -DDPA_AVX=ON \
        -DDPA_DPDK_RX_REUSE=ON \
        -DDPA_DPDK_WIN_HUGE=ON \
        -DDPA_DPDK_RE_FIRST=ΟFF \
        -DDPA_TORCH_PINNEDPOOL=ON \
        -DDPA_TORCH_WORKSTEALING=ON ..
  make -j4 install

  # Install the plugin
  source $HOME/straggle-ml-experiments/venv/bin/activate

  # Make sure env is ok
  python -m pip install --upgrade pip 
  python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"

  # Compile the plugin
  PYTHONWARNINGS="ignore::setuptools.errors.SetuptoolsDeprecationWarning,ignore::setuptools.errors.EasyInstallDeprecationWarning" \
  python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py -q develop
else
  source $HOME/straggle-ml-experiments/venv/bin/activate
fi


cd $HOME/straggle-ml-experiments

SCRIPT=${0##*/}
DPA_CONF=$HOME/straggle-ml-experiments/configs/edgecore.json
IFACE="${IFACE:-ens4f1}"                 # network interface to read IP from
WORLD_SIZE="${WORLD_SIZE:-6}"            # set by launcher or leave 1 for single-node
BACKEND="${BACKEND:-dpa_dpdk}"
MASTER_ADDR="${MASTER_ADDR:-"42.0.1.1"}"
MASTER_PORT="${MASTER_PORT:-"29500"}"

# Derive IP on IFACE, rank = last octet - 1
IP=$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | cut -d/ -f1 || true)
if [[ -z "${IP}" ]]; then
  echo "[$SCRIPT] ERROR: could not get IPv4 for IFACE=$IFACE" >&2
  exit 1
fi

RANK=$(( ${IP##*.} - 1 ))

MASTER_ADDR="${MASTER_ADDR:-$(awk -F. '{print $1"."$2"."$3".1"}' <<< "$IP")}"

echo "[$SCRIPT] iface=$IFACE ip=$IP rank=$RANK world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} backend=$BACKEND"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
NCCL_SOCKET_IFNAME=ens4f0 NCCL_IB_HCA=mlx5_0,mlx5_1 \

set -x

GDB='gdb -ex run --args'

set -x
# squad_v1
# sudo -E $(which python) experiments/train/roberta_finetune.py \
#   --rank "$RANK" \
#   --world_size "$WORLD_SIZE" \
#   --iface "$IFACE" \
#   --master_addr "$MASTER_ADDR" \
#   --master_port "$MASTER_PORT" \
#   --backend $BACKEND \
#   --dpa_conf $DPA_CONF \
#   --data ~/datasets/squad_v1 \
#   --squad_version v1 \
#   --n_best_size 100 \
#   --epochs 6 \
#   --batch_size 32 \
#   --learning_rate 5e-5 \
#   --warmup_ratio 0.1 \
#   --deterministic \
#   --workers 4 \
#   --prefetch_factor 4 \
#   --log_interval 20 \
#   --json experiments/train/roberta_finetune.json \
#   "$@"


# squad v2
sudo -E $(which python) experiments/train/roberta_finetune2.py \
  --rank "$RANK" \
  --world_size "$WORLD_SIZE" \
  --iface "$IFACE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend $BACKEND \
  --dpa_conf $DPA_CONF \
  --data ~/datasets/squad_v2 \
  --squad_version v2 \
  --n_best_size 100 \
  --epochs 8 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.2 \
  --deterministic \
  --workers 4 \
  --prefetch_factor 4 \
  --log_interval 100 \
  --val_every 150 \
  --val_every_subset 0.25 \
  --json experiments/train/roberta_finetune.json \
  --straggle_points 3 \
  --straggle_prob 16 \
  --straggle_ranks 1 \
  --straggle_amount 1.2 \
  --straggle_multiply 0.5 2 \
  "$@"