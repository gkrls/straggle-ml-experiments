#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && "$1" == "sync" ]]; then
  # Sync the repos
  if [ ! -d "$HOME/straggle-ml/.git" ]; then
    git clone https://github.com/gkrls/straggle-ml.git "$HOME/straggle-ml"
  else
    git -C "$HOME/straggle-ml" reset --hard >/dev/null 2>&1 || true
    git -C "$HOME/straggle-ml" pull --ff-only || true
  fi
  if [ ! -d "$HOME/straggle-ml-experiments/.git" ]; then
    git clone https://github.com/gkrls/straggle-ml-experiments.git "$HOME/straggle-ml-experiments"
  else
    git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
    git -C "$HOME/straggle-ml-experiments" pull --ff-only || true
  fi

  # Make sure we have up to date DPA
  cd $HOME/straggle-ml/build
  cmake -DCMAKE_BUILD_TYPE=Release -DDPA_DEVELOP=OFF -DDPA_AVX=ON -DDPA_DPDK_RX_REUSE=ON -DDPA_SWITCH=OFF ..
  make -j4 install

  # Install the plugin
  source $HOME/straggle-ml-experiments/venv/bin/activate
  export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig
  python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py develop
else
  source $HOME/straggle-ml-experiments/venv/bin/activate
fi


cd $HOME/straggle-ml-experiments

IFACE=ens4f1
# Derive IP on IFACE, rank = last octet - 1
IP=$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | cut -d/ -f1 || true)
if [[ -z "${IP}" ]]; then
  echo "ERROR: could not get IPv4 for IFACE=$IFACE" >&2
  exit 1
fi
RANK=$(( ${IP##*.} - 1 ))
WORLD=6
MASTER_ADDR=42.0.1.1
MASTER_PORT=29500

PROG=experiments/allreduce/allreduce-benchmark.py
CONF=experiments/allreduce/edgecore.json
VALGRIND=valgrind #--leak-check=full --show-leak-kinds=all --track-origins=yes"

sudo -E $(which python) $PROG --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  --dpa_conf $CONF --dpa_pipes 4 -b dpa_dpdk -d cuda -t int32 -s 100000000 -w 5 -i 20 \
  --gloo_socket_ifname=$IFACE --global_stats

# sudo -E $(which python) experiments/allreduce-benchmark.py --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --d_conf configs/config-edgecore.json -b nccl -d cuda -t float32 -s 1000 -i 5 -w 3 -v "$@"

# perf stat -e cache-misses,cache-references
# dpa: backend finished with pool[0:16] seqnums: 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1...


