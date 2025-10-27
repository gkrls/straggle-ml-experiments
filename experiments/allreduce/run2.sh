#!/usr/bin/env bash
set -euo pipefail

BRANCH="wip-simple"

# export PKG_CONFIG_PATH=/opt/mellanox/dpdk/23.11/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig

# Sync the repos if needed
if [[ $# -eq 1 && "$1" == "sync" ]]; then
  [[ -d "$HOME/straggle-ml/.git" ]] || git clone https://github.com/gkrls/straggle-ml.git "$HOME/straggle-ml"

  git -C "$HOME/straggle-ml" fetch -q origin || true
  git -C "$HOME/straggle-ml" checkout "$BRANCH" 2>/dev/null || true
  git -C "$HOME/straggle-ml" reset --hard origin/"$BRANCH" 2>/dev/null || true #git -C "$DIR" reset --hard || true
  git -C "$HOME/straggle-ml" pull --ff-only origin "$BRANCH" 2>/dev/null || true
  git -C "$HOME/straggle-ml" clean -ffd || true

  # git -C "$HOME/straggle-ml" checkout "$BRANCH" || true
  # git -C "$HOME/straggle-ml" reset --hard || true
  # git -C "$HOME/straggle-ml" pull --ff-only || true

  if [ ! -d "$HOME/straggle-ml-experiments/.git" ]; then
    git clone https://github.com/gkrls/straggle-ml-experiments.git "$HOME/straggle-ml-experiments"
  else
    git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
    git -C "$HOME/straggle-ml-experiments" pull --ff-only || true
  fi

  # Make sure we have up to date DPA
  # sudo rm -rf $HOME/straggle-ml/build
  mkdir -p $HOME/straggle-ml/build
  cd $HOME/straggle-ml/build
  cmake -DCMAKE_INSTALL_MESSAGE=LAZY \
        -DCMAKE_BUILD_TYPE=Debug \
        -DDPA_TRACE=ON \
        -DDPA_DEVELOP=OFF \
        -DDPA_SWITCH=OFF \
        -DDPA_AVX=ON \
        -DDPA_DPDK_RX_REUSE=ON \
        -DDPA_DPDK_WIN_HUGE=ON \
        -DDPA_DPDK_RE_FIRST=ΟFF \
        -DDPA_TORCH_PINNEDPOOL=ON \
        -DDPA_TORCH_PINNEDPOOL_PRETOUCH=OFF \
        -DDPA_TORCH_WORKSTEALING=ON ..
  make -j4 install

  # Install the plugin
  source $HOME/straggle-ml-experiments/venv/bin/activate
  python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py -q develop
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
WORLD=2
MASTER_ADDR=42.0.1.1
MASTER_PORT=29500

PROG=experiments/allreduce/allreduce-benchmark.py
# CONF=experiments/allreduce/edgecore.json
CONF=configs/edgecore.json
# CONF=experiments/allreduce/netberg.json
VALGRIND=valgrind #--leak-check=full --show-leak-kinds=all --track-origins=yes"
# PROF="nsys profile -o myprofile -t cuda,osrt --stats=true --force-overwrite=true"
NSYS="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true --sampling-period=200000 -d 30 -o $HOME/nsys_profile -f true"
PERF="perf stat -d --"

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8  # or .so.6 depending on your GCC version
# export ASAN_OPTIONS=symbolize=1,abort_on_error=1,print_stats=1,check_initialization_order=1

sudo -E $(which python) $PROG --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  --dpa_conf $CONF --dpa_pipes 4 -b dpa_dpdk -d cuda -t float32 -s 25000 -w 0 -i 1 --dpa_avg \
  --gloo_socket_ifname=$IFACE --global_stats --pattern 2 --straggle_k 1 --straggle_ms 10000 #--verify

# sudo -E $(which python) experiments/allreduce-benchmark.py --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --d_conf configs/config-edgecore.json -b nccl -d cuda -t float32 -s 1000 -i 5 -w 3 -v "$@"

# perf stat -e cache-misses,cache-references
# dpa: backend finished with pool[0:16] seqnums: 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1...


# "eal_extra_args": ["--log-level=pmd.net.mlx5:8"]