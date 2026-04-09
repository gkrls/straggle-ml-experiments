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
        -DCMAKE_BUILD_TYPE=Release \
        -DDPA_TRACE=OFF \
        -DDPA_DEVELOP=OFF \
        -DDPA_SWITCH=OFF \
        -DDPA_AVX=ON \
        -DDPA_PROFILE=OFF \
        -DDPA_IMPLICIT_SYN=OFF \
        -DDPA_FASTESTK_EXIT=OFF \
        -DDPA_FASTESTK_BULK=OFF \
        -DDPA_SYNCHRON_BULK=OFF \
        -DDPA_DPDK_RE_DISABLE=OFF \
        -DDPA_DPDK_RX_REUSE=ON \
        -DDPA_DPDK_WIN_HUGE=ON \
        -DDPA_DPDK_RE_FIRST=ΟFF \
        -DDPA_TORCH_PINNEDPOOL=ON \
        -DDPA_TORCH_PINNEDPOOL_PRETOUCH=OFF \
        -DDPA_TORCH_PIPELINE=ON \
        -DDPA_TORCH_WORKSTEAL=OFF ..
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
WORLD=6
MASTER_ADDR=42.0.1.1
MASTER_PORT=29500

PROG=experiments/perf/bench.py
CONF=configs/edgecore.json
CONF_NS=configs/edgecore-ns.json
VALGRIND=valgrind #--leak-check=full --show-leak-kinds=all --track-origins=yes"
NSYS="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true --sampling-period=200000 -d 30 -o $HOME/nsys_profile -f true"
PERF="perf stat -d --" # perf s44tat -e cache-misses,cache-references
GDB="gdb --args"

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8
# export ASAN_OPTIONS=symbolize=1,abort_on_error=1,print_stats=1,check_initialization_order=1

LOGFILE="allreduce_bench_$(date +%Y%m%d_%H%M%S).log"


# NCCL STUFF
export NCCL_SOCKET_IFNAME=$IFACE
export NCCL_IB_HCA=mlx5_1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# GDR  # export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_NET_GDR_LEVEL=SYS  # Tell NCCL that crossing the internal CPU fabric (SYS level) is okay
export NCCL_P2P_LEVEL=SYS
export NCCL_P2P_DISABLE=0      # Ensure Peer-to-Peer is not restricted
export NCCL_IB_GDR_FLUSH=1     # Recommended for Mellanox + GDR performance


# export NCCL_PROTO=LL           # LL, LL128, Simple
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=8


sudo modprobe nvidia_peermem 2>/dev/null || true

# DPA STUFF
export DPA_TORCH_PIPELINE_CHUNKS=4
W_TP_2=192
W_TP_4=96
W_TP_6=64
W_LA_6=128
W_LA_8=64



XXXXXXXS=25000  # 0.10MB
XXXXXXS=62500   # 0.25MB
XXXXXS=125000   # 0.50MB
XXXXS=250000    # 1MB
XXXS=625000     # 2.5MB
XXS=1250000     # 5.0MB
XS=2500000      # 10MB
S=6250000       # 25MB
M=12500000      # 50MB
L=25000000      # 100MB
XL=50000000     # 200MB
XXL=75000000    # 300ΜΒ
XXXL=100000000  # 400MB
XXXXL=125000000 # 500MB

export DPA_PREEMPTIVE=0
export DPA_SYN_DISABLE=0
export DPA_DPDK_MONITOR=0
export DPA_DPDK_MONITOR_INTERVAL_US=500

echo "[ALLREDUCE BENCHMARK]"
sudo -E DPA_LOG=Info DPA_SCHEDULER=OFF $(which python) $PROG \
  --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  -d cuda -t float32 -s $L -w 5 -i 20 --pattern 3 \
  -b dpa_dpdk \
  --dpa_conf $CONF --dpa_pipes 4 --dpa_window 96 --dpa_threads 6 \
  --dpa_k 6 #--dpa_timeout_us 60 --dpa_timeout_init_scaling 20 \
  # --gloo_socket_ifname $IFACE
  #  --straggle_rank 1 --straggle_ms 200 --straggle_num 10 --straggle_start 0 --straggle_mode op
  
  # --dpa_k 5 --dpa_preemptive --dpa_window 64 --dpa_threads 6 --dpa_timeout_us 100 --dpa_profile_skip 4 --dpa_timeout_init_scaling 5 --batch
 
#   # --gloo_socket_ifname $IFACE --gloo_num_threads 2
  # --nccl_socket_nthreads 6 --nccl_nsocks_perthread 2
  # --pattern 1 --nccl_ib_qps_per_connection 1

# ns run
# echo "[STRAGGLE UNAWARE BENCHMARK]"
# sudo -E $(which python) $PROG3 --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --dpa_conf $CONF_NS --dpa_pipes 4 -b dpa_dpdk -d cuda -t int32 -s 25000000 -w 0 -i 1 --pattern 1 --batch --verify 
  # --straggle_rank 1 --straggle_ms 2000 --straggle_num 10 --straggle_start 10 --straggle_mode op

# sudo -E $(which python) $PROG --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --dpa_conf $CONF --dpa_pipes 4 -b dpa_dpdk -d cuda -t float32 -s 25000000 -w 10 -i 100 --pattern 2 --avg --verify --batch 1 \
#   --straggle_k 5 --straggle_rank 1 --straggle_ms 2000 --straggle_num 10 --straggle_start 10

  # 
  # --gloo_socket_ifname $IFACE --gloo_num_threads 2
  # --nccl_ib_qps_per_connection 2
  

  #--verify
  # --gloo_socket_ifname=$IFACE --global_stats --straggle_rank 1 --straggle_ms 2000 --straggle_num 10 --straggle_start 10 --batch
  # --batch #--verify

# sudo -E $(which python) experiments/allreduce-benchmark.py --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --d_conf configs/config-edgecore.json -b nccl -d cuda -t float32 -s 1000 -i 5 -w 3 -v "$@"


# dpa: backend finished with pool[0:16] seqnums: 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1...


# "eal_extra_args": ["--log-level=pmd.net.mlx5:8"]