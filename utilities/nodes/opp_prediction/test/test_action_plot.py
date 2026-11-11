import json

import matplotlib
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
import pandas as pd
import torch

from opp_prediction.action_plot import plot_actions, plot_episodes, sample_actions
from opp_prediction.ensemble import Ensemble

GREEN, BLUE = matplotlib.colors.to_rgba('tab:green'), matplotlib.colors.to_rgba('tab:blue')


class FixedEnsemble:
    """Member b predicts mean [b, 10 + b] with std [0.1, 0.2] for every input."""

    def __init__(self, members=3):
        self.members = [None] * members

    def distribution(self, inputs):
        n, m = len(inputs), len(self.members)
        mean = torch.stack([torch.tensor([b, 10.0 + b]).repeat(n, 1) for b in range(m)])
        return mean, torch.tensor([0.1, 0.2]).repeat(m, n, 1)


def test_one_action_per_sample_particle_and_action():
    assert sample_actions(FixedEnsemble(), torch.zeros(7, 4), particles=5, rng=np.random.default_rng(0)).shape == (7, 5, 2)


def test_particle_p_samples_from_member_p_mod_members():
    actions = sample_actions(FixedEnsemble(members=3), torch.zeros(2000, 4), particles=6, rng=np.random.default_rng(0))
    for p in range(6):
        np.testing.assert_allclose(actions[:, p].mean(0), [p % 3, 10 + p % 3], atol=0.02)


def test_samples_spread_with_the_predicted_std():
    actions = sample_actions(FixedEnsemble(members=1), torch.zeros(5000, 4), particles=2, rng=np.random.default_rng(0))
    np.testing.assert_allclose(actions.reshape(-1, 2).std(0), [0.1, 0.2], rtol=0.05)


def test_sampling_is_seeded():
    def draw(seed):
        return sample_actions(FixedEnsemble(), torch.zeros(3, 4), particles=4, rng=np.random.default_rng(seed))
    assert np.array_equal(draw(1), draw(1)) and not np.array_equal(draw(1), draw(2))


def plotted(**kwargs):
    t = np.arange(10) * 0.05
    truth = np.stack([np.sin(t), 3 + t], axis=1)
    particles = truth[:, None] + np.random.default_rng(0).normal(0, 0.1, (10, 40, 2))
    return plot_actions(t, truth, particles, title='ep_0001', **kwargs), t, truth, particles


def of_type(ax, kind):
    return [c for c in ax.collections if isinstance(c, kind)]


def luminance(color):
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def band_edges(band, t):
    """Lowest and highest y of the band outline at each time."""
    vertices = np.concatenate([path.vertices for path in band.get_paths()])
    return (np.array([vertices[np.isclose(vertices[:, 0], x), 1].min() for x in t]),
            np.array([vertices[np.isclose(vertices[:, 0], x), 1].max() for x in t]))


def test_one_panel_per_action_on_a_shared_time_axis():
    fig, *_ = plotted()
    steer, speed = fig.axes
    assert steer.get_shared_x_axes().joined(steer, speed)
    assert 'steer' in steer.get_ylabel() and 'speed' in speed.get_ylabel()


def test_truth_is_a_green_line_and_particles_are_blue_dots():
    fig, t, truth, particles = plotted()
    for a, ax in enumerate(fig.axes):
        line, = ax.get_lines()
        np.testing.assert_array_equal(line.get_xydata(), np.stack([t, truth[:, a]], axis=1))
        assert matplotlib.colors.to_rgba(line.get_color()) == GREEN
        dots, = of_type(ax, PathCollection)
        assert len(dots.get_offsets()) == particles[..., a].size
        np.testing.assert_allclose(dots.get_offsets()[:, 1], particles[..., a].ravel())
        assert tuple(dots.get_facecolor()[0][:3]) == BLUE[:3]


def run_with_model(tmp_path):
    """A run of three episodes and a model whose test part is ep_1 and ep_2."""
    rows = [{'episode': f'ep_{e}', 'track': 'test', 'opp_policy': 'a', 't': 2.0 + 0.05 * k, 'x_b': k, 'x_a': e,
             'y_steer_0': 0.1 * e, 'y_speed_0': 3.0} for e in range(3) for k in range(8)]
    pd.DataFrame(rows).to_csv(tmp_path / 'dataset.csv', index=False)
    model_dir = tmp_path / 'models' / 'm'
    model_dir.mkdir(parents=True)
    Ensemble(['x_a', 'x_b'], members=2, hidden=8, layers=1).save(model_dir / 'ensemble.pt')
    (model_dir / 'report.json').write_text(json.dumps({'episodes': {'test': ['ep_1', 'ep_2'], 'val': ['ep_0']}}))
    return tmp_path, model_dir


def test_writes_one_plot_per_episode_of_the_part(tmp_path):
    run_dir, model_dir = run_with_model(tmp_path)
    paths = plot_episodes(run_dir, model_dir, part='test', particles=3)
    assert [p.name for p in paths] == ['actions_ep_1.png', 'actions_ep_2.png']
    assert all(p.parent == model_dir / 'action_plots' and p.stat().st_size > 0 for p in paths)


def test_named_episodes_override_the_part(tmp_path):
    run_dir, model_dir = run_with_model(tmp_path)
    assert [p.name for p in plot_episodes(run_dir, model_dir, episodes=['ep_0'], particles=3)] == ['actions_ep_0.png']


def test_truth_line_breaks_where_dataset_rows_are_missing():
    t = np.array([0.0, 0.05, 0.1, 0.4, 0.45])  # rows between 0.1 and 0.4 were dropped
    truth = np.ones((5, 2))
    fig = plot_actions(t, truth, np.ones((5, 2, 2)))
    for ax in fig.axes:
        y = ax.get_lines()[0].get_ydata()
        assert np.isnan(y).sum() == 1 and np.isnan(y[3])


def test_band_encloses_the_central_fraction_of_the_particles_at_each_step():
    fig, t, _, particles = plotted(band=0.8)
    for a, ax in enumerate(fig.axes):
        band, = of_type(ax, PolyCollection)
        lower, upper = band_edges(band, t)
        np.testing.assert_allclose(lower, np.quantile(particles[..., a], 0.1, axis=1))
        np.testing.assert_allclose(upper, np.quantile(particles[..., a], 0.9, axis=1))


def test_full_band_is_the_hull_of_all_particles():
    fig, t, _, particles = plotted(band=1.0)
    lower, upper = band_edges(of_type(fig.axes[0], PolyCollection)[0], t)
    np.testing.assert_allclose(lower, particles[..., 0].min(1))
    np.testing.assert_allclose(upper, particles[..., 0].max(1))


def test_default_band_is_the_central_90_percent():
    fig, t, _, particles = plotted()
    _, upper = band_edges(of_type(fig.axes[0], PolyCollection)[0], t)
    np.testing.assert_allclose(upper, np.quantile(particles[..., 0], 0.95, axis=1))


def test_no_band_when_disabled():
    fig, *_ = plotted(band=None)
    assert all(not of_type(ax, PolyCollection) for ax in fig.axes)


def test_band_is_lighter_inside_and_darker_at_the_edge_than_the_particles():
    fig, *_ = plotted()
    ax = fig.axes[0]
    band, dots = of_type(ax, PolyCollection)[0], of_type(ax, PathCollection)[0]
    assert luminance(band.get_facecolor()[0]) > luminance(BLUE) > luminance(band.get_edgecolor()[0])
    assert band.get_zorder() < dots.get_zorder() < ax.get_lines()[0].get_zorder()


def test_band_breaks_where_dataset_rows_are_missing():
    t = np.array([0.0, 0.05, 0.1, 0.4, 0.45])
    fig = plot_actions(t, np.ones((5, 2)), np.random.default_rng(0).normal(size=(5, 4, 2)))
    assert len(of_type(fig.axes[0], PolyCollection)[0].get_paths()) == 2
