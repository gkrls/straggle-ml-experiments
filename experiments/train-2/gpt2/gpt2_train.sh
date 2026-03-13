#!/usr/bin/env bash
set -euo pipefail

# Try to set CPU governor to performance; ignore all failures
if command -v cpupower &>/dev/null; then
    sudo cpupower frequency-set -g performance 2>/dev/null && \
        echo "[cpufreq] Set governor to performance" || \
        echo "[cpufreq] Could not set governor (BIOS-locked or no sudo?), continuing"
elif [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    # fallback: write directly if cpupower isn't installed
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo tee "$gov" 2>/dev/null
    done && \
        echo "[cpufreq] Set governor to performance (tee fallback)" || \
        echo "[cpufreq] Could not set governor via tee, continuing"
else
    echo "[cpufreq] No cpufreq support on this node, continuing"
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
        -DDPA_FASTESTK_EXIT=OFF \
        -DDPA_FASTESTK_BULK=OFF \
        -DDPA_SYNCHRON_BULK=OFF \
        -DDPA_DPDK_RX_REUSE=ON \
        -DDPA_DPDK_WIN_HUGE=ON \
        -DDPA_DPDK_RE_FIRST=ΟFF \
        -DDPA_TORCH_PINNEDPOOL=ON \
        -DDPA_TORCH_WORKSTEALING=ON ..
  make -j4 install

  # Install the plugin
  source $HOME/straggle-ml-experiments/venv/bin/activate

  # Make sure env is ok
  python -m pip install --upgrade -q pip 
  python -m pip install --no-user -q -r "$HOME/straggle-ml-experiments/requirements.txt"

  # Compile the plugin
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

PY="gpt2-2.py"

# Consumes around ~15.3GB of memory with AMP
sudo -E DPA_LOG=INFO DPA_SCHEDULER=OFF $(which python) experiments/train-2/gpt2/"$PY" \
  --rank "$RANK" \
  --world_size "$WORLD_SIZE" \
  --iface "$IFACE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --dpa_conf $DPA_CONF \
  --dpa_repin \
  --backend $BACKEND \
  --workers 4 \
  --epochs 1 \
  --batch_size 12 \
  --micro_steps_per_epoch 6000 \
  --gradient_accumulation_steps 5 --learning_rate 0.0006 --min_lr 0.00006 --mini_val_every_opt_steps 300 --log_every_opt_steps 50 \
  --seq_len 1024 \
  --amp \
  --deterministic \
  --prefetch_factor 4 \
  --json experiments/train-2/gpt2_sa_straggle_aggressive.json \
  --data ~/datasets/openwebtext \
  --cache_dir ~/datasets/openwebtext/cache \
  --dpa_world_k 5 \
  --straggle_points 3 \
  --straggle_prob 15 \
  --straggle_ranks 1 \
  --straggle_amount 1.66 \
  --straggle_skip 5 \
  --straggle_multiply 0.5 2
  # --best_model \
  # --best_model_ignore 1 \

  # --straggle_skip 33 \
  # --straggle_skip_every 66 \

# for GA=5 use:
# --gradient_accumulation_steps 5 --learning_rate 0.0006 --min_lr 0.00006 --mini_val_every_opt_steps 300 --log_every_opt_steps 50

# for GA=1 use: (maybe the same LR as with GA=5 is also fine)
# --gradient_accumulation_steps 1 --learning_rate 0.0003 --min_lr 0.00003 --mini_val_every_opt_steps 1500 --log_every_opt_steps 250