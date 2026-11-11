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
BAND_FACE, BAND_EDGE = '#c6dbef', '#08306b'  # lighter and darker than the particles
PARTICLE_INK = 5.0  # opacity summed over the particles of a step, so that more particles do not hide the band


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


def _band_label(band):
    return 'all particles' if band == 1.0 else f'central {band:.0%} of particles'


def plot_actions(t, truth, particles, title=None, band=0.9) -> Figure:
    """t [N], truth [N, 2], particles [N, P, 2] -> one panel per action over time: the recorded action as a green line
    on top of the particles as faint blue dots, on top of a band that encloses the central fraction band of the
    particles at each step (1.0 = min/max hull, None = no band). Line and band break where dataset rows are missing."""
    fig = Figure(figsize=(12, 6), layout='constrained')
    axes = fig.subplots(2, 1, sharex=True)
    t, particles = np.asarray(t), np.asarray(particles)
    times = np.repeat(t, particles.shape[1])
    line_t, line = _break_at_gaps(t, np.asarray(truth))
    if band is not None:
        _, lower = _break_at_gaps(t, np.quantile(particles, 0.5 - band / 2, axis=1))
        _, upper = _break_at_gaps(t, np.quantile(particles, 0.5 + band / 2, axis=1))
    for a, (ax, label) in enumerate(zip(axes, PANELS)):
        if band is not None:
            ax.fill_between(line_t, lower[:, a], upper[:, a], facecolor=BAND_FACE, edgecolor=BAND_EDGE, alpha=0.8,
                            linewidth=0.8, zorder=1, label=_band_label(band))
        ax.scatter(times, particles[..., a].ravel(), s=4, color='tab:blue', alpha=min(0.25, PARTICLE_INK / particles.shape[1]), linewidths=0, zorder=2,
                   label='particles')
        ax.plot(line_t, line[:, a], color='tab:green', linewidth=1.2, zorder=3, label='ground truth')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=3, markerscale=3)
    for handle in legend.legend_handles:
        if handle.get_label() == 'particles':
            handle.set_alpha(1.0)  # the faint particles would hardly show in the legend
    axes[-1].set_xlabel('time [s]')
    if title:
        fig.suptitle(title)
    return fig


def plot_episodes(run_dir, model_dir, part: str = 'test', episodes=None, particles: int = 20, seed: int = 0,
                  band=0.9) -> list:
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
                           sample_actions(ensemble, inputs, particles, rng), title=f'{episode}, {particles} particles', band=band)
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
    parser.add_argument('--band', type=float, default=0.9,
                        help='central fraction of the particles inside the band, 1 for their hull, 0 for no band')
    args = parser.parse_args()
    for path in plot_episodes(args.run_dir, args.model_dir, args.part, args.episodes, args.particles, args.seed,
                              args.band or None):
        print(path)


if __name__ == '__main__':
    main()
