import json

import numpy as np
import pandas as pd
import pytest
import torch

from opp_prediction.ensemble import Ensemble
from opp_prediction.train import TrainConfig, split_episodes, train, trained_on

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
    """y_steer = x_opp_d, y_speed = 2 + noise whose std is 0.05 for x_opp_vs < 0 and 0.4 for x_opp_vs > 0. Both laws
    also hold on the mirror image, where the offset and the steering change sign."""
    rng = np.random.default_rng(0)
    n = 6000
    a, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    df = pd.DataFrame({
        'episode': [f'ep_{i % 30:02d}' for i in range(n)], 'opp_policy': ['p'] * n,
        'x_opp_d': a, 'x_opp_vs': b,
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


def test_the_report_also_holds_the_mirror_image_of_the_test_episodes(trained):
    report = json.loads((trained / 'report.json').read_text())
    assert report['rows']['test_mirrored'] == report['rows']['test']
    assert report['test_mirrored']['rmse']['steer'] < 0.1  # the law of the fixture holds on the mirror image too


def test_mirroring_adds_the_mirror_image_of_every_train_and_val_row(noisy_line, tmp_path):
    plain = train(noisy_line, tmp_path / 'plain', TrainConfig(**SMALL, tensorboard=False))
    mirror = train(noisy_line, tmp_path / 'mirror', TrainConfig(**SMALL, tensorboard=False, mirror=True))
    assert mirror['rows'] == {**plain['rows'], 'train': 2 * plain['rows']['train'], 'val': 2 * plain['rows']['val']}
    assert mirror['config']['mirror'] is True


def test_the_report_names_the_dataset_of_the_model(trained):
    assert json.loads((trained / 'report.json').read_text())['dataset'] == 'dataset.csv'


def write_run(run_dir, model_dir, name, report_extra):
    run_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    pd.DataFrame({'episode': ['ep_00'], 'x_a': [1.0]}).to_csv(run_dir / name, index=False)
    (run_dir / name.replace('.csv', '_columns.json')).write_text(json.dumps({'config': {'history': 3}, 'columns': []}))
    (model_dir / 'report.json').write_text(json.dumps(report_extra))


def test_a_model_reads_back_the_dataset_it_was_trained_on_and_its_config(tmp_path):
    write_run(tmp_path / 'run', tmp_path / 'model', 'dataset_h3.csv', {'dataset': 'dataset_h3.csv'})
    dataset, config = trained_on(tmp_path / 'run', tmp_path / 'model')
    assert dataset['x_a'].tolist() == [1.0]
    assert config['history'] == 3


def test_a_model_without_a_named_dataset_was_trained_on_the_default_one(tmp_path):
    write_run(tmp_path / 'run', tmp_path / 'model', 'dataset.csv', {})
    assert trained_on(tmp_path / 'run', tmp_path / 'model')[1]['history'] == 3
