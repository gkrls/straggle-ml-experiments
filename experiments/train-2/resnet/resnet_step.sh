#!/usr/bin/env bash
set -euo pipefail

sudo cpupower frequency-set -g performance 2>/dev/null && \
  echo "[cpufreq] Set governor to performance" || \
  echo "[cpufreq] Could not set governor, continuing"

# Disable C-states, restore on exit
if cpupower idle-info 2>/dev/null | grep -q "C[1-9]"; then
  sudo cpupower idle-set -D 0 2>/dev/null
  trap 'sudo cpupower idle-set -E 0 2>/dev/null' EXIT
  echo "[cpupower] C-states disabled (will restore on exit)"
fi

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
        -DDPA_PROFILE=OFF \
        -DDPA_SWITCH=OFF \
        -DDPA_AVX=ON \
        -DDPA_FASTESTK=ON \
        -DDPA_FASTESTK_EXIT=OFF \
        -DDPA_FASTESTK_BULK=OFF \
        -DDPA_IMPLICIT_SYN=OFF \
        -DDPA_SYNCHRON_BULK=OFF \
        -DDPA_DPDK_RX_REUSE=ON \
        -DDPA_DPDK_WIN_HUGE=ON \
        -DDPA_DPDK_RE_FIRST=ΟFF \
        -DDPA_TORCH_PINNEDPOOL=ON \
        -DDPA_TORCH_PINNEDPOOL_PRETOUCH=ON \
        -DDPA_TORCH_WORKSTEALING=ON ..
  make -j4 install

  # Install the plugin
  source $HOME/straggle-ml-experiments/venv/bin/activate

  # Make sure env is ok
  python -m pip install --upgrade pip 
  python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"

  # # Compile the plugin
  # PYTHONWARNINGS="ignore::setuptools.errors.SetuptoolsDeprecationWarning,ignore::setuptools.errors.EasyInstallDeprecationWarning" \
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
# IP_LAST=${IP##*.}
# if [ "$IP_LAST" -eq 1 ]; then
#     RANK=1
# elif [ "$IP_LAST" -eq 2 ]; then
#     RANK=0
# else
#     RANK=$(( IP_LAST - 1 ))
# fi
# MASTER_ADDR="${MASTER_ADDR:-$(awk -F. '{print $1"."$2"."$3".1"}' <<< "$IP")}"

echo "[$SCRIPT] iface=$IFACE ip=$IP rank=$RANK world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} backend=$BACKEND"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
NCCL_SOCKET_IFNAME=ens4f0 NCCL_IB_HCA=mlx5_0,mlx5_1 \

set -x

GDB='gdb -ex run --args'

sudo -E DPA_LOG=INFO DPA_SCHEDULER=OFF $(which python) experiments/train-2/resnet/resnet_step.py \
  --rank "$RANK" \
  --world_size "$WORLD_SIZE" \
  --iface "$IFACE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend dpa_dpdk \
  --dpa_conf $DPA_CONF \
  --dpa_repin \
  --model resnet50 \
  --data ~/datasets/imagenet \
  --workers 8 \
  --epochs 92 \
  --batch_size 128 \
  --deterministic \
  --prefetch_factor 4 \
  --drop_last_val \
  --json experiments/train-2/resnet_nat_50steps.json \
  --log_every_steps 50 \
  --dpa_k 6
  # --straggle_points 3 \
  # --straggle_prob 20 \
  # --straggle_ranks 1 \
  # --straggle_amount 0.6 \
  # --straggle_multiply 0.5 2.0
  
# sudo -E $(which python) experiments/train/resnet.py \
#   --rank "$RANK" \
#   --world_size "$WORLD_SIZE" \
#   --iface "$IFACE" \
#   --master_addr "$MASTER_ADDR" \
#   --master_port "$MASTER_PORT" \
#   --dpa_conf $DPA_CONF \
#   --backend $BACKEND \
#   --data ~/datasets/imagenet \
#   --model resnet50 \
#   --batch_size 128 \
#   --workers 8 \
#   --deterministic \
#   --drop_last_val \
#   --prefetch_factor 4 \
#   --json experiments/train/resnet50.json \
#   "$@"