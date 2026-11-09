import json

import numpy as np
import pandas as pd
import pytest
import torch

from opp_prediction.ensemble import Ensemble
from opp_prediction.train import TrainConfig, split_episodes, train


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
