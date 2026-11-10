import json

import numpy as np
import pandas as pd
import pytest
import torch

from opp_prediction.ensemble import Ensemble
from opp_prediction.train import TrainConfig, split_episodes, train

SMALL = dict(members=2, hidden=16, layers=1, max_epochs=3, patience=3, seed=0)


def episodes(n=20):
    return {f'ep_{i:02d}': ('a' if i % 2 else 'b') for i in range(n)}


def test_split_keeps_whole_episodes_apart_and_covers_all():
    s = split_episodes(episodes(), seed=1)
    train_set, val, test = set(s['train']), set(s['val']), set(s['test'])
    assert not (train_set & val or train_set & test or val & test)
    assert train_set | val | test == set(episodes())


def test_split_is_seeded():
    assert split_episodes(episodes(), seed=3) == split_episodes(episodes(), seed=3)
    assert split_episodes(episodes(), seed=3) != split_episodes(episodes(), seed=4)


def test_split_has_every_opponent_policy_in_every_part():
    pol = episodes()
    for part in split_episodes(pol, seed=2).values():
        assert {pol[e] for e in part} == {'a', 'b'}


@pytest.fixture
def noisy_line(tmp_path):
    """y_steer = x_a, y_speed = 2 + noise whose std is 0.05 for x_b < 0 and 0.4 for x_b > 0."""
    rng = np.random.default_rng(0)
    n = 6000
    a, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    df = pd.DataFrame({
        'episode': [f'ep_{i % 30:02d}' for i in range(n)], 'opp_policy': ['p'] * n,
        'x_a': a, 'x_b': b,
        'y_steer_0': a, 'y_speed_0': 2.0 + rng.normal(0, np.where(b < 0, 0.05, 0.4))})
    path = tmp_path / 'dataset.csv'
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def trained(noisy_line, tmp_path):
    out = tmp_path / 'out'
    train(noisy_line, out, TrainConfig(members=2, hidden=32, layers=2, max_epochs=60, patience=60, seed=0))
    return out


def test_training_recovers_the_mean_and_the_noise(trained):
    model = Ensemble.load(trained / 'ensemble.pt')
    o = torch.tensor([[0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
    mu, sigma = model.distribution(o)
    assert abs(mu[:, 0, 0].mean().item() - 0.5) < 0.1
    assert abs(sigma[:, 0, 1].mean().item() - 0.05) < 0.03
    assert abs(sigma[:, 1, 1].mean().item() - 0.4) < 0.1


def test_training_reports_held_out_results(trained):
    report = json.loads((trained / 'report.json').read_text())
    assert set(report['episodes']) == {'train', 'val', 'test'}
    assert report['test']['rmse']['speed'] < report['test']['rmse_of_train_mean']['speed'] + 0.05
    assert 0.0 < report['test']['coverage']['0.9']['speed'] <= 1.0


def scalar_tags(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(run_dir))
    events.Reload()
    return events


def test_tensorboard_writes_one_run_per_member_with_the_same_tags(noisy_line, tmp_path):
    train(noisy_line, tmp_path / 'out', TrainConfig(**SMALL))
    logs = tmp_path / 'out' / 'tensorboard'
    first, second = scalar_tags(logs / 'member_0'), scalar_tags(logs / 'member_1')
    assert first.Tags()['scalars'] == second.Tags()['scalars']
    for tag in ('loss/train', 'nll/val', 'rmse/steer', 'sigma_mean/speed', 'logvar_bound/min/steer',
                'logvar_bound/max/speed', 'bound_share/steer', 'coverage_90/speed'):
        assert tag in first.Tags()['scalars']
    assert len(first.Scalars('nll/val')) == 3
    assert 'sigma/steer' in first.Tags()['histograms']


def test_tensorboard_summarizes_the_ensemble(noisy_line, tmp_path):
    train(noisy_line, tmp_path / 'out', TrainConfig(**SMALL))
    tags = scalar_tags(tmp_path / 'out' / 'tensorboard' / 'ensemble').Tags()['scalars']
    for tag in ('epistemic_std/steer', 'aleatoric_std/speed', 'coverage_90/steer', 'nll'):
        assert tag in tags


def test_tensorboard_can_be_turned_off(noisy_line, tmp_path):
    train(noisy_line, tmp_path / 'out', TrainConfig(**SMALL, tensorboard=False))
    assert not (tmp_path / 'out' / 'tensorboard').exists()


def test_logging_does_not_change_the_training(noisy_line, tmp_path):
    train(noisy_line, tmp_path / 'with', TrainConfig(**SMALL, tensorboard=True))
    train(noisy_line, tmp_path / 'without', TrainConfig(**SMALL, tensorboard=False))
    logged = Ensemble.load(tmp_path / 'with' / 'ensemble.pt').state_dict()
    plain = Ensemble.load(tmp_path / 'without' / 'ensemble.pt').state_dict()
    assert all(torch.equal(logged[key], plain[key]) for key in plain)


def test_a_new_run_replaces_the_old_logs(noisy_line, tmp_path):
    train(noisy_line, tmp_path / 'out', TrainConfig(**SMALL))
    train(noisy_line, tmp_path / 'out', TrainConfig(**SMALL))
    assert len(list((tmp_path / 'out' / 'tensorboard' / 'member_0').glob('events.out.*'))) == 1
