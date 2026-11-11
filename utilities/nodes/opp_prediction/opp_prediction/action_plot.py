"""
Plots the one-step action particles of the ensemble against the recorded actions of held-out episodes. Every step
predicts from the recorded features of its dataset row, so the plot shows the policy model without rollout drift.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from opp_prediction.ensemble import Ensemble
from opp_prediction.train import ACTIONS

PANELS = ('steer angle [rad]', 'speed command [m/s]')
GAP_FACTOR = 1.5  # a time step this many times the typical one means dataset rows were dropped


def _break_at_gaps(t, values):
    """Inserts NaN where rows are missing so that a line is not drawn across them."""
    if len(t) < 3:
        return t, values
    gaps = np.flatnonzero(np.diff(t) > GAP_FACTOR * np.median(np.diff(t))) + 1
    return np.insert(t.astype(float), gaps, np.nan), np.insert(values.astype(float), gaps, np.nan, axis=0)


def sample_actions(ensemble, inputs, particles: int, rng) -> np.ndarray:
    """inputs [N, n_inputs] -> actions [N, particles, 2]. Particle p samples mean + std * eps of member p % members,
    the same scheme as one step of the Predictor."""
    mu, sigma = ensemble.distribution(inputs)
    members = np.arange(particles) % len(ensemble.members)
    mean, std = mu[members].numpy().swapaxes(0, 1), sigma[members].numpy().swapaxes(0, 1)
    return mean + std * rng.standard_normal(mean.shape)


def plot_actions(t, truth, particles, title=None) -> Figure:
    """t [N], truth [N, 2], particles [N, P, 2] -> one panel per action over time: the recorded action as a green line
    on top of the particles as faint blue dots. The line breaks where dataset rows are missing."""
    fig = Figure(figsize=(12, 6), layout='constrained')
    axes = fig.subplots(2, 1, sharex=True)
    times = np.repeat(t, particles.shape[1])
    line_t, line = _break_at_gaps(np.asarray(t), np.asarray(truth))
    for a, (ax, label) in enumerate(zip(axes, PANELS)):
        ax.scatter(times, particles[..., a].ravel(), s=4, color='tab:blue', alpha=0.15, linewidths=0, label='particles')
        ax.plot(line_t, line[:, a], color='tab:green', linewidth=1.2, label='ground truth')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', markerscale=3)
    axes[-1].set_xlabel('time [s]')
    if title:
        fig.suptitle(title)
    return fig


def plot_episodes(run_dir, model_dir, part: str = 'test', episodes=None, particles: int = 20, seed: int = 0) -> list:
    """Writes <model_dir>/action_plots/actions_<episode>.png for the episodes of a part of the model's split, or for
    the named episodes. Returns the paths."""
    run_dir, model_dir = pathlib.Path(run_dir), pathlib.Path(model_dir)
    ensemble = Ensemble.load(model_dir / 'ensemble.pt')
    if episodes is None:
        episodes = json.loads((model_dir / 'report.json').read_text())['episodes'][part]
    dataset = pd.read_csv(run_dir / 'dataset.csv')
    out = model_dir / 'action_plots'
    out.mkdir(exist_ok=True)
    rng, paths = np.random.default_rng(seed), []
    for episode in episodes:
        rows = dataset[dataset['episode'] == episode].sort_values('t')
        inputs = torch.tensor(rows[ensemble.input_columns].to_numpy(), dtype=torch.float32)
        fig = plot_actions(rows['t'].to_numpy(), rows[list(ACTIONS.values())].to_numpy(),
                           sample_actions(ensemble, inputs, particles, rng), title=f'{episode}, {particles} particles')
        paths.append(out / f'actions_{episode}.png')
        fig.savefig(paths[-1], dpi=150)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('model_dir')
    parser.add_argument('--part', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--episodes', nargs='+', help='plot these episodes instead of a part')
    parser.add_argument('--particles', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    for path in plot_episodes(args.run_dir, args.model_dir, args.part, args.episodes, args.particles, args.seed):
        print(path)


if __name__ == '__main__':
    main()
