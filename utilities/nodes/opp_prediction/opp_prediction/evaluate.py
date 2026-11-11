"""Evaluates the predictor on held-out episodes: accuracy, calibration and usefulness of the uncertainty."""
import argparse
import json
import pathlib
from typing import NamedTuple

import numpy as np
import yaml

from datagen.rollout import RolloutModel
from datagen.track import Track
from opp_prediction.ensemble import Ensemble
from opp_prediction.metrics import summarize
from opp_prediction.predict import PREDICTION_DT_S, RHO, EgoPlan, Predictor, project
from opp_prediction.recording import Recording, load_recording
from opp_prediction.train import trained_on

REPO = pathlib.Path(__file__).resolve().parents[4]
HEADLINE_STEPS = (20, 30, 60)  # 1.0 s, 1.5 s, 3.0 s


class Errors(NamedTuple):
    particles: np.ndarray  # [samples, particles, horizon, 2] particle minus truth, along and across the track
    baseline: np.ndarray   # [samples, horizon, 2] constant velocity minus truth
    members: np.ndarray    # [particles]


def _at_or_before(times, t):
    return max(int(np.searchsorted(times, t, side='right')) - 1, 0)


def _frenet_error(track, s, d, s_true, d_true):
    return np.stack([track.wrap_ds(s - s_true), d - d_true], axis=-1)


def observed_frames(rows, input_columns, history: int) -> np.ndarray:
    """Dataset rows -> [rows, history - 1, features]: the frames each row holds before its current one."""
    features = len(input_columns) // history
    return rows[input_columns[:len(input_columns) - features]].to_numpy().reshape(len(rows), history - 1, features)


def rollout_errors(rec: Recording, predictor, track, horizon: int, start_rows, past_inputs=None) -> Errors:
    """
    Predicts from each start row of the dataset grid. Steps past the end of the episode stay NaN.
    past_inputs: [start rows, history - 1, features] frames observed before each start, for an ensemble with history.
    """
    particle_errors, baseline_errors, members = [], [], None
    for i, row in enumerate(start_rows):
        t0 = rec.grid_ns[row]
        xi_0 = rec.opp_states[_at_or_before(rec.opp_t, t0)]
        plan = _at_or_before(rec.plan_t, t0)
        previous_steer = float(rec.commands[_at_or_before(rec.cmd_t, t0), 0])
        ego_plan = EgoPlan(rec.plan_x[plan], rec.plan_y[plan], rec.plan_speed[plan],
                           measured_speed=rec.ego_speed[_at_or_before(rec.ego_t, t0)])
        result = predictor.predict(xi_0, ego_plan, track, horizon=horizon, previous_steer=previous_steer,
                                   past_inputs=None if past_inputs is None else past_inputs[i])
        members = result.members
        start = project(track, xi_0[None])
        particles = np.full((len(members), horizon, 2), np.nan)
        baseline = np.full((horizon, 2), np.nan)
        for h in range(1, horizon + 1):
            if row + h >= len(rec.grid_ns):
                break
            x, y = rec.opp_states[_at_or_before(rec.opp_t, rec.grid_ns[row + h])][:2]
            s_true, d_true = track.to_frenet(x, y)
            predicted = project(track, result.states[:, h])
            particles[:, h - 1] = _frenet_error(track, predicted['s'], predicted['d'], s_true, d_true)
            t = h * PREDICTION_DT_S
            baseline[h - 1] = _frenet_error(track, start['s'] + t * start['vs'], start['d'] + t * start['vd'],
                                            s_true, d_true)[0]
        particle_errors.append(particles)
        baseline_errors.append(baseline)
    return Errors(np.array(particle_errors), np.array(baseline_errors), members)


def evaluate(run_dir, model_dir, part: str = 'test', stride: int = 5, horizon: int = 60, rho: float = RHO,
             seed: int = 0) -> dict:
    run_dir, model_dir = pathlib.Path(run_dir), pathlib.Path(model_dir)
    dataset, cfg = trained_on(run_dir, model_dir)
    episodes = json.loads((model_dir / 'report.json').read_text())['episodes'][part]
    sim = yaml.safe_load((REPO / 'stack_master/config/SIM/sim_params.yaml').read_text())
    ensemble = Ensemble.load(model_dir / 'ensemble.pt')
    predictor = Predictor(ensemble, RolloutModel({k: float(v) for k, v in sim.items()}), cfg['features'], rho=rho,
                          seed=seed)
    tracks, results = {}, []
    for episode in episodes:
        rec = load_recording(run_dir / 'episodes' / episode, cfg, REPO / 'utilities/libraries/f110_msgs/msg')
        if rec.track not in tracks:
            waypoints = REPO / 'stack_master/maps' / rec.track / 'global_waypoints.json'
            tracks[rec.track] = Track.from_json(str(waypoints), cfg['reference'])
        warmup = json.loads((run_dir / 'episodes' / episode / 'episode.json').read_text())['rules']['warmup_s']
        starts = dataset[dataset['episode'] == episode].iloc[::stride]
        rows = np.round((starts['t'].to_numpy() - warmup) / cfg['dt']).astype(int)
        past = observed_frames(starts, ensemble.input_columns, cfg['history'])
        results.append(rollout_errors(rec, predictor, tracks[rec.track], horizon, rows, past))

    errors = np.concatenate([r.particles for r in results])
    baseline = np.concatenate([r.baseline for r in results])
    report = {'part': part, 'episodes': episodes, 'stride': stride, 'rho': rho,
              **summarize(errors, results[0].members, baseline, HEADLINE_STEPS, PREDICTION_DT_S)}
    (model_dir / f'evaluation_{part}_rho{rho:g}.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('model_dir')
    parser.add_argument('--part', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--stride', type=int, default=5, help='start a prediction at every n-th dataset row')
    parser.add_argument('--rho', type=float, default=RHO, help='correlation of the action noise between steps')
    args = parser.parse_args()
    report = evaluate(args.run_dir, args.model_dir, args.part, args.stride, rho=args.rho)
    print(json.dumps(report['steps'], indent=2))


if __name__ == '__main__':
    main()
