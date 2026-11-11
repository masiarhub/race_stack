# opp_prediction

Learns the opponent's policy as an ensemble of probabilistic networks. No ROS needed, it reads a `dataset.csv` of `datagen`.

```bash
conda create -n opp_prediction python=3.10
conda activate opp_prediction
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m pytest test                                       # from this folder
python -m opp_prediction.train <dataset.csv> <out_dir>      # writes ensemble.pt, report.json and a TensorBoard log
python -m opp_prediction.train <dataset.csv> <out_dir> --mirror   # also trains on the mirror image of every track
tensorboard --logdir <out_dir>/tensorboard                  # one run per ensemble member, overlaid
```

`Ensemble.distribution(o)` returns the action mean and std in physical units, each `[members, N, 2]`.

`report.json` holds the results on the held-out episodes as recorded (`test`) and on their mirror image
(`test_mirrored`), the same track driven the other way round in the lateral sense. `--mirror` adds the mirror image
of every train and val row, a second track orientation without new recordings.

`Predictor.predict(xi_0, ego_plan, track, past_inputs=...)` returns the particles of the opponent's trajectory. An
ensemble trained with `history: H` needs the H - 1 feature frames observed before the prediction as `past_inputs`.
`EgoPlan(x, y, speed, measured_speed)` drives the path of the ego's plan with its speed profile scaled to start at the
ego's measured speed: the plan's own speeds are those of the raceline, the ego drives slower whenever its controller
holds it back, e.g. while it trails the opponent. `evaluate` takes the measured speed from `/state_estimation/odom`.
The action noise of a particle is correlated between steps by `rho` (default 0.8). It needs `datagen` and `f110_gym`
on the path, for example:

```bash
export PYTHONPATH=$PWD:$PWD/../datagen:$PWD/../../../base_system/f110_simulator/f1tenth_gym/gym
```

A model remembers the dataset it was trained on (`dataset` in `report.json`). `evaluate` and `action_plot` read that
dataset of the run and its config (`<dataset>_columns.json`), for example a `dataset_h3.csv` that
`ros2 run datagen extract <run_dir> --spec <spec> --name dataset_h3.csv` derived with another history.

`python -m opp_prediction.evaluate <run_dir> <model_dir> [--rho 0.8]` predicts on the held-out episodes of a run and writes
`evaluation_<part>_rho<rho>.json` next to the model: accuracy, calibration and usefulness of the uncertainty, per step
of the horizon and separately along (s) and across (d) the track, against a constant-velocity baseline.

`python -m opp_prediction.action_plot <run_dir> <model_dir> [--part test] [--episodes ep_0006 ...] [--particles 20]`
plots one-step action particles (dark blue), their mean (red) and their ±1σ (light blue, with a vertical line per step)
against the recorded actions (green) per held-out episode, steering and speed over time, into
`<model_dir>/action_plots/`. Every step predicts from the recorded features of its dataset row.
