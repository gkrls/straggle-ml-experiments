cd $HOME/straggle-ml
git reset --hard && git pull
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DDPA_DEVELOP=OFF -DDPA_AVX=ON -DDPA_AVX_512=ON -DDPA_SWITCH=OFF ..
make -j4 install

cd $HOME/straggle-ml-experiments
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"

export PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig

python $HOME/straggle-ml/build/install/lib/dpa_plugin_pytorch/setup.py develop