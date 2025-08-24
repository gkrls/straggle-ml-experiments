cd $HOME/straggle-ml-experiments
python3 -m venv venv
source "$HOME/straggle-ml-experiments/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-user -r "$HOME/straggle-ml-experiments/requirements.txt"