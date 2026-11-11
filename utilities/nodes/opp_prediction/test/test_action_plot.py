import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection, PolyCollection

from opp_prediction.action_plot import plot_actions, plot_episodes, sample_actions
from opp_prediction.ensemble import Ensemble

GREEN, RED, BLUE = (matplotlib.colors.to_rgba(c) for c in ('tab:green', 'tab:red', 'tab:blue'))


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


def plotted():
    t = np.arange(10) * 0.05
    truth = np.stack([np.sin(t), 3 + t], axis=1)
    particles = truth[:, None] + np.random.default_rng(0).normal(0, 0.1, (10, 40, 2))
    return plot_actions(t, truth, particles, title='ep_0001'), t, truth, particles


def artist(ax, gid):
    found = [a for a in ax.get_children() if a.get_gid() == gid]
    assert len(found) == 1, f'{len(found)} artists {gid}'
    return found[0]


def luminance(color):
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def one_sigma(particles, a):
    mean, std = particles[..., a].mean(1), particles[..., a].std(1, ddof=1)
    return mean - std, mean, mean + std


def test_one_panel_per_action_on_a_shared_time_axis():
    fig, *_ = plotted()
    steer, speed = fig.axes
    assert steer.get_shared_x_axes().joined(steer, speed)
    assert 'steer' in steer.get_ylabel() and 'speed' in speed.get_ylabel()


def test_truth_is_a_green_line():
    fig, t, truth, _ = plotted()
    for a, ax in enumerate(fig.axes):
        line = artist(ax, 'truth')
        np.testing.assert_array_equal(line.get_xydata(), np.stack([t, truth[:, a]], axis=1))
        assert matplotlib.colors.to_rgba(line.get_color()) == GREEN


def test_particles_are_dots_darker_than_tab_blue():
    fig, _, _, particles = plotted()
    for a, ax in enumerate(fig.axes):
        dots = artist(ax, 'particles')
        np.testing.assert_allclose(dots.get_offsets()[:, 1], particles[..., a].ravel())
        assert luminance(dots.get_facecolor()[0]) < luminance(BLUE)


def test_mean_is_a_red_line_through_the_mean_of_the_particles():
    fig, t, _, particles = plotted()
    for a, ax in enumerate(fig.axes):
        line = artist(ax, 'mean')
        np.testing.assert_allclose(line.get_xydata(), np.stack([t, one_sigma(particles, a)[1]], axis=1))
        assert matplotlib.colors.to_rgba(line.get_color()) == RED


def test_sigma_curves_go_through_the_mean_minus_and_plus_one_std():
    fig, _, _, particles = plotted()
    for a, ax in enumerate(fig.axes):
        lower, _, upper = one_sigma(particles, a)
        np.testing.assert_allclose(artist(ax, 'sigma_lower').get_ydata(), lower)
        np.testing.assert_allclose(artist(ax, 'sigma_upper').get_ydata(), upper)


def test_one_vertical_sigma_line_per_step():
    fig, t, _, particles = plotted()
    for a, ax in enumerate(fig.axes):
        bars = artist(ax, 'sigma_bars')
        assert isinstance(bars, LineCollection)
        lower, _, upper = one_sigma(particles, a)
        expected = np.stack([np.stack([t, lower], axis=1), np.stack([t, upper], axis=1)], axis=1)
        np.testing.assert_allclose(np.array(bars.get_segments()), expected)


def test_space_between_the_sigma_curves_is_light_blue():
    fig, t, _, particles = plotted()
    for a, ax in enumerate(fig.axes):
        band = artist(ax, 'sigma_band')
        assert isinstance(band, PolyCollection)
        vertices = np.concatenate([path.vertices for path in band.get_paths()])
        lower, _, upper = one_sigma(particles, a)
        np.testing.assert_allclose([vertices[np.isclose(vertices[:, 0], x), 1].min() for x in t], lower)
        np.testing.assert_allclose([vertices[np.isclose(vertices[:, 0], x), 1].max() for x in t], upper)
        assert luminance(band.get_facecolor()[0]) > luminance(BLUE)


def test_band_under_particles_under_mean_under_truth():
    fig, *_ = plotted()
    ax = fig.axes[0]
    order = [artist(ax, gid).get_zorder() for gid in ('sigma_band', 'particles', 'mean', 'truth')]
    assert order == sorted(order) and len(set(order)) == 4


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


def test_lines_and_band_break_where_dataset_rows_are_missing():
    t = np.array([0.0, 0.05, 0.1, 0.4, 0.45])  # rows between 0.1 and 0.4 were dropped
    fig = plot_actions(t, np.ones((5, 2)), np.random.default_rng(0).normal(size=(5, 4, 2)))
    for ax in fig.axes:
        for gid in ('truth', 'mean', 'sigma_lower', 'sigma_upper'):
            y = artist(ax, gid).get_ydata()
            assert np.isnan(y).sum() == 1 and np.isnan(y[3]), gid
        assert len(artist(ax, 'sigma_band').get_paths()) == 2
