if [ ! -d "$HOME/straggle-ml-experiments/.git" ]; then
  git clone https://github.com/gkrls/straggle-ml-experiments.git "$HOME/straggle-ml-experiments"
else
  git -C "$HOME/straggle-ml-experiments" reset --hard >/dev/null 2>&1 || true
  git -C "$HOME/straggle-ml-experiments" pull --ff-only || true
fi

# cd $HOME/straggle-ml-experiments
# python3 -m venv venv
# source "$HOME/straggle-ml-experiments/venv/bin/activate"

cd $HOME/straggle-ml-experiments
if [ ! -d "venv" ]; then
  python -m venv venv
fi
source "$HOME/straggle-ml-experiments/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"

cd $HOME/straggle-ml
git reset --hard && git pull
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DDPA_DEVELOP=OFF -DDPA_AVX=ON -DDPA_AVX_512=ON -DDPA_SWITCH=OFF ..
make -j4 install

export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig

python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py develop