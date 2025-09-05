#!/usr/bin/env bash
set -euo pipefail

IFACE="${IFACE:-ens4f0}"                 # network interface to read IP from
WORLD_SIZE="${WORLD_SIZE:-1}"            # set by launcher or leave 1 for single-node
BACKEND="${BACKEND:-gloo}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Derive IP on IFACE, rank = last octet - 1
IP=$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | cut -d/ -f1 || true)
if [[ -z "${IP}" ]]; then
  echo "[run_vgg.sh] ERROR: could not get IPv4 for IFACE=$IFACE" >&2
  exit 1
fi
RANK=$(( ${IP##*.} - 1 ))

MASTER_ADDR="${MASTER_ADDR:-$(awk -F. '{print $1"."$2"."$3".1"}' <<< "$IP")}"

echo "[run_gpt2.sh] iface=$IFACE ip=$IP rank=$RANK world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} backend=$BACKEND"

# sync repo: clone if missing, otherwise reset/pull
if [ ! -d "$HOME/straggle-ml-experiments/.git" ]; then
  git clone https://github.com/gkrls/straggle-ml-experiments.git "$HOME/straggle-ml-experiments"
else
  git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
  git -C "$HOME/straggle-ml-experiments" pull --ff-only || true
fi

source $HOME/straggle-ml/venv/bin/activate
python -m pip install --upgrade pip 
python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
NCCL_SOCKET_IFNAME=ens4f0 NCCL_IB_HCA=mlx5_0,mlx5_1 \

set -x

# Standard settings for GPT2 on a 16GB GPU
# Consumes around ~15.3GB of memory
exec python -u $HOME/straggle-ml-experiments/models/gpt2_2.py \
  --rank "$RANK" \
  --world_size "$WORLD_SIZE" \
  --iface "$IFACE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend gloo \
  --workers 8 \
  --epochs 12 \
  --steps_per_epoch 6000 \
  --mini_val_every_steps 300 \
  --gradient_accumulation_steps 5 \
  --batch_size 12 \
  --seq_len 1024 \
  --amp \
  --prefetch_factor 4 \
  --straggle_points 3 \
  --straggle_prob 2 \
  --straggle_ranks 1 \
  --straggle_amount 1.69 \
  --straggle_multiply 0.5 2 \
  --straggle_verbose \
  --log_every_steps 50 \
  --json $HOME/straggle-ml-experiments/models/gpt2.json \
  --data ~/datasets/openwebtext \
  --cache_dir ~/datasets/openwebtext/cache \
  "$@"


  # --backend "$BACKEND" \