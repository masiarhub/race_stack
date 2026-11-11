"""Trains the ensemble on a dataset.csv of datagen and reports its action-level results on held-out episodes."""
import argparse
import copy
import dataclasses
import json
import math
import pathlib

import numpy as np
import pandas as pd
import torch

from datagen.extract import mirrored
from opp_prediction.ensemble import Ensemble
from opp_prediction.training_log import TrainingLog

ACTIONS = {'steer': 'y_steer_0', 'speed': 'y_speed_0'}
COVERAGE_LEVELS = (0.5, 0.9)
DEFAULT_DATASET = 'dataset.csv'


@dataclasses.dataclass
class TrainConfig:
    members: int = 5
    hidden: int = 256
    layers: int = 3
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    seed: int = 0
    tensorboard: bool = True
    mirror: bool = False  # add every train and val row a second time, on the mirror image of its track


def split_episodes(episode_policy: dict, seed: int, fractions=(0.7, 0.15, 0.15)) -> dict:
    """{episode: opponent policy} -> whole episodes per train / val / test, drawn per policy so that each part has all."""
    rng = np.random.default_rng(seed)
    parts = {'train': [], 'val': [], 'test': []}
    for policy in sorted(set(episode_policy.values())):
        eps = rng.permutation(sorted(e for e, p in episode_policy.items() if p == policy)).tolist()
        n_val, n_test = (max(1, round(len(eps) * f)) for f in fractions[1:])
        parts['val'] += eps[:n_val]
        parts['test'] += eps[n_val:n_val + n_test]
        parts['train'] += eps[n_val + n_test:]
    return {name: sorted(part) for name, part in parts.items()}


def _fit_member(model: Ensemble, member: int, data: dict, cfg: TrainConfig, log: TrainingLog):
    inputs, actions = data['train']
    val_inputs, val_actions = data['val']
    optimizer = torch.optim.Adam(model.members[member].parameters(), lr=cfg.lr)
    order = torch.Generator().manual_seed(cfg.seed + member)
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        for batch in torch.randperm(len(inputs), generator=order).split(cfg.batch_size):
            optimizer.zero_grad()
            loss = model.loss(member, inputs[batch], actions[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        with torch.no_grad():
            val = model.nll(member, val_inputs, val_actions).item()
        log.epoch(model, member, epoch, epoch_loss / len(inputs), val, val_inputs, val_actions)
        if val < best:
            best, best_state, best_epoch, stale = val, copy.deepcopy(model.members[member].state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.members[member].load_state_dict(best_state)
    log.member_done(member, best_epoch)


def _evaluate(model: Ensemble, inputs, actions, train_mean) -> dict:
    """Action-level results of the mixture of all members, per action where it can be separated."""
    mu, sigma = model.distribution(inputs)
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = torch.logsumexp(normal.log_prob(actions).sum(-1), dim=0) - math.log(len(mu))
    cdf = normal.cdf(actions).mean(0)

    def per_action(values):
        return {name: float(values[i]) for i, name in enumerate(ACTIONS)}

    return {
        'nll': float(-log_prob.mean()),
        'rmse': per_action(((mu.mean(0) - actions) ** 2).mean(0).sqrt()),
        'rmse_of_train_mean': per_action(((train_mean - actions) ** 2).mean(0).sqrt()),
        'aleatoric_std': per_action((sigma ** 2).mean((0, 1)).sqrt()),
        'epistemic_std': per_action(mu.var(0).mean(0).sqrt()),
        'coverage': {str(level): per_action(((cdf - 0.5).abs() <= level / 2).float().mean(0))
                     for level in COVERAGE_LEVELS},
    }


def trained_on(run_dir, model_dir) -> tuple:
    """The dataset of the run that the model was trained on, and the dataset config that derived it."""
    run_dir = pathlib.Path(run_dir)
    name = json.loads((pathlib.Path(model_dir) / 'report.json').read_text()).get('dataset', DEFAULT_DATASET)
    config = json.loads((run_dir / name).with_name(f'{pathlib.Path(name).stem}_columns.json').read_text())['config']
    return pd.read_csv(run_dir / name), config


def train(dataset_csv, out_dir, config: TrainConfig = TrainConfig()) -> dict:
    df = pd.read_csv(dataset_csv)
    input_columns = [c for c in df.columns if c.startswith('x_')]
    split = split_episodes(df.groupby('episode')['opp_policy'].first().to_dict(), config.seed)
    parts = {name: df[df['episode'].isin(episodes)] for name, episodes in split.items()}
    parts['test_mirrored'] = mirrored(parts['test'])  # before training: a column without a mirror rule fails here
    if config.mirror:
        for name in ('train', 'val'):
            parts[name] = pd.concat([parts[name], mirrored(parts[name])], ignore_index=True)
    data = {name: (torch.tensor(part[input_columns].to_numpy(), dtype=torch.float32),
                   torch.tensor(part[list(ACTIONS.values())].to_numpy(), dtype=torch.float32))
            for name, part in parts.items()}

    torch.manual_seed(config.seed)
    model = Ensemble(input_columns, config.members, config.hidden, config.layers)
    model.fit_normalization(*data['train'])
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = TrainingLog(out / 'tensorboard' if config.tensorboard else None, tuple(ACTIONS))
    for member in range(config.members):
        _fit_member(model, member, data, config, log)
    model.eval()

    train_mean = data['train'][1].mean(0)
    report = {'dataset': pathlib.Path(dataset_csv).name, 'config': dataclasses.asdict(config), 'episodes': split,
              'rows': {k: len(v[0]) for k, v in data.items()},
              **{name: _evaluate(model, *data[name], train_mean) for name in ('val', 'test', 'test_mirrored')}}
    log.ensemble(report['val'], dataclasses.asdict(config))
    log.close()
    model.save(out / 'ensemble.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_csv')
    parser.add_argument('out_dir')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-tensorboard', action='store_true', help='do not write the TensorBoard log')
    parser.add_argument('--mirror', action='store_true',
                        help='add every train and val row a second time, on the mirror image of its track')
    args = parser.parse_args()
    report = train(args.dataset_csv, args.out_dir,
                   TrainConfig(seed=args.seed, tensorboard=not args.no_tensorboard, mirror=args.mirror))
    print(json.dumps({k: report[k] for k in ('rows', 'val', 'test', 'test_mirrored')}, indent=2))


if __name__ == '__main__':
    main()
