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