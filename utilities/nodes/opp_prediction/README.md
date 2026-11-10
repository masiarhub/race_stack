# opp_prediction

Learns the opponent's policy as an ensemble of probabilistic networks. No ROS needed, it reads a `dataset.csv` of `datagen`.

```bash
conda create -n opp_prediction python=3.10
conda activate opp_prediction
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m pytest test                                       # from this folder
python -m opp_prediction.train <dataset.csv> <out_dir>      # writes ensemble.pt and report.json
```

`Ensemble.distribution(o)` returns the action mean and std in physical units, each `[members, N, 2]`.

`Predictor.predict(xi_0, ego_plan, track)` returns the particles of the opponent's trajectory. It needs `datagen` and
`f110_gym` on the path, for example:

```bash
export PYTHONPATH=$PWD:$PWD/../datagen:$PWD/../../../base_system/f110_simulator/f1tenth_gym/gym
```
