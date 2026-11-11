# opp_prediction

Learns the opponent's policy as an ensemble of probabilistic networks. No ROS needed, it reads a `dataset.csv` of `datagen`.

```bash
conda create -n opp_prediction python=3.10
conda activate opp_prediction
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m pytest test                                       # from this folder
python -m opp_prediction.train <dataset.csv> <out_dir>      # writes ensemble.pt, report.json and a TensorBoard log
tensorboard --logdir <out_dir>/tensorboard                  # one run per ensemble member, overlaid
```

`Ensemble.distribution(o)` returns the action mean and std in physical units, each `[members, N, 2]`.

`Predictor.predict(xi_0, ego_plan, track)` returns the particles of the opponent's trajectory. It needs `datagen` and
`f110_gym` on the path, for example:

```bash
export PYTHONPATH=$PWD:$PWD/../datagen:$PWD/../../../base_system/f110_simulator/f1tenth_gym/gym
```

`python -m opp_prediction.evaluate <run_dir> <model_dir>` predicts on the held-out episodes of a run and writes
`evaluation_<part>_rho<rho>.json` next to the model: accuracy, calibration and usefulness of the uncertainty, per step
of the horizon and separately along (s) and across (d) the track, against a constant-velocity baseline.

`python -m opp_prediction.action_plot <run_dir> <model_dir> [--part test] [--episodes ep_0006 ...] [--particles 20]`
plots one-step action particles (blue) against the recorded actions (green) per held-out episode, steering and speed
over time, into `<model_dir>/action_plots/`. Every step predicts from the recorded features of its dataset row.
