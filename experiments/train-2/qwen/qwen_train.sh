#!/usr/bin/env bash
set -euo pipefail

# Try to set CPU governor to performance;
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

export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig

# Sync the repos if needed
if [[ $# -ge 1 && "$1" == "sync" ]]; then
  shift

  [[ -d "$HOME/straggle-ml/.git" ]] || git clone https://github.com/gkrls/straggle-ml.git "$HOME/straggle-ml"

  git -C "$HOME/straggle-ml" fetch -q origin || true
  git -C "$HOME/straggle-ml" checkout "$BRANCH" 2>/dev/null || true
  git -C "$HOME/straggle-ml" reset --hard origin/"$BRANCH" 2>/dev/null || true
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
        -DDPA_TORCH_WORKSTEALING=ON ..
  make -j4 install

  source $HOME/straggle-ml-experiments/venv/bin/activate

  python -m pip install --upgrade -q pip
  python -m pip install --no-user -q -r "$HOME/straggle-ml-experiments/requirements.txt"

  python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py -q develop
else
  source $HOME/straggle-ml-experiments/venv/bin/activate
fi


cd $HOME/straggle-ml-experiments

SCRIPT=${0##*/}
DPA_CONF=$HOME/straggle-ml-experiments/configs/edgecore.json
IFACE="${IFACE:-ens4f1}"
WORLD_SIZE="${WORLD_SIZE:-6}"
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

echo "[$SCRIPT] iface=$IFACE ip=$IP rank=$RANK world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} backend=$BACKEND"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
NCCL_SOCKET_IFNAME=ens4f0 NCCL_IB_HCA=mlx5_0,mlx5_1 \

set -x

# Qwen2.5-0.5B full fine-tuning on Alpaca
# Memory: ~8GB params/optim + ~6GB activations with AMP, batch=4, seq=512

QWEN_15=Qwen/Qwen1.5-0.5B #--learning_rate 0.00002
QWEN_25=Qwen/Qwen2.5-0.5B #--learning_rate 0.000002

# https://debuggercafe.com/fine-tuning-qwen-1-5-for-coding/

sudo -E DPA_LOG=INFO DPA_SCHEDULER=OFF $(which python) experiments/train-2/qwen/qwen2.py \
  --rank "$RANK" \
  --world_size "$WORLD_SIZE" \
  --iface "$IFACE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend $BACKEND \
  --dpa_conf $DPA_CONF \
  --dpa_repin \
  --workers 0 \
  --model_name $QWEN_15 \
  --dataset sahil2801/CodeAlpaca-20k \
  --data ~/datasets/qwen-codealpaca \
  --seq_len 512 \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 0.000005 \
  --gradient_accumulation_steps 1 \
  --sched cosine \
  --amp \
  --deterministic \
  --prefetch_factor 4 \
  --log_every_opt_steps 1 \
  --mini_val_every_opt_steps 50 \
  --mini_val_0 \
  --json experiments/train-2/qwen_codealpaca_su_aggressive.json \
  --dpa_k 6 \
  --save_model ~/straggle-ml-experiments/saved_models/qwen15
  # --straggle_points 3 \
  # --straggle_prob 15 \
  # --straggle_ranks 1 \
  # --straggle_amount 1.47 \
  # --straggle_multiply 0.5 2.0 \

  # --epochs 6 \
  # --batch_size 3 \
  # --learning_rate 0.00002 \
  # --gradient_accumulation_steps 1 \

  # --epochs 10 \
  # --batch_size 4 \
  # --learning_rate 0.0001 \
  # --gradient_accumulation_steps 21 \

  # --epochs 10 \
  # --batch_size 4 \
  # --learning_rate 0.000005 \
  # --gradient_accumulation_steps 1 \

  # --gradient_accumulation_steps 2 \
# DATASETS
  # --dataset tatsu-lab/alpaca \
  # --data ~/datasets/qwen-alpaca \

  # --dataset sahil2801/CodeAlpaca-20k \
  # --data ~/datasets/qwen-codealpaca \

  # --dataset meta-math/MetaMathQA-40K \
  # --data ~/datasets/qwen-metamath \

# Notes:
# --straggle_amount 1.1 is estimated micro-step time for Qwen 0.5B (batch=4, seq=512, AMP)
#   Calibrated from GPT-2 micro-step time of 1.66s:
#   Qwen tokens/micro = 2048, GPT-2 tokens/micro = 12288
#   FLOPs ratio ~ (494M * 2048) / (124M * 12288) ~ 0.66x
#   Estimated: 1.66 * 0.66 ~ 1.1s (may need adjustment after first run)
#
# For baseline (no stragglers), comment out straggle_* lines and set:
#   --json experiments/train-2/qwen_alpaca_baseline.json
#
# For gloo baseline:
#   BACKEND=gloo and remove --dpa_conf, --dpa_repin, --dpa_k lines