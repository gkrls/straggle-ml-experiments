# Make sure we have up to date DPA
cd $HOME/straggle-ml/build
make -j4 install

source $HOME/straggle-ml-experiments/venv/bin/activate

# Install the plugin
export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig
python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py develop

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

PROG=experiments/allreduce-bench/allreduce-benchmark.py
CONF=experiments/allreduce-bench/config.json

sudo -E $(which python) $PROG --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  --dpa_conf $CONF -b dpa_sock -d cuda -t float32 -s 10000 -w 3 -i 5 -v "$@"

# sudo -E $(which python) experiments/allreduce-benchmark.py --rank $RANK --world_size $WORLD --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#   --d_conf configs/config-edgecore.json -b nccl -d cuda -t float32 -s 1000 -i 5 -w 3 -v "$@"
