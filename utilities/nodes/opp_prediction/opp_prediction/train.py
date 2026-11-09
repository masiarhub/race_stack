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

from opp_prediction.ensemble import Ensemble

ACTIONS = {'steer': 'y_steer_0', 'speed': 'y_speed_0'}
COVERAGE_LEVELS = (0.5, 0.9)


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


def _fit_member(model: Ensemble, member: int, data: dict, cfg: TrainConfig):
    inputs, actions = data['train']
    val_inputs, val_actions = data['val']
    optimizer = torch.optim.Adam(model.members[member].parameters(), lr=cfg.lr)
    order = torch.Generator().manual_seed(cfg.seed + member)
    best, best_state, stale = math.inf, None, 0
    for _ in range(cfg.max_epochs):
        for batch in torch.randperm(len(inputs), generator=order).split(cfg.batch_size):
            optimizer.zero_grad()
            model.loss(member, inputs[batch], actions[batch]).backward()
            optimizer.step()
        with torch.no_grad():
            val = model.nll(member, val_inputs, val_actions).item()
        if val < best:
            best, best_state, stale = val, copy.deepcopy(model.members[member].state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.members[member].load_state_dict(best_state)


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


def train(dataset_csv, out_dir, config: TrainConfig = TrainConfig()) -> dict:
    df = pd.read_csv(dataset_csv)
    input_columns = [c for c in df.columns if c.startswith('x_')]
    split = split_episodes(df.groupby('episode')['opp_policy'].first().to_dict(), config.seed)
    data = {}
    for name, episodes in split.items():
        part = df[df['episode'].isin(episodes)]
        data[name] = (torch.tensor(part[input_columns].to_numpy(), dtype=torch.float32),
                      torch.tensor(part[list(ACTIONS.values())].to_numpy(), dtype=torch.float32))

    torch.manual_seed(config.seed)
    model = Ensemble(input_columns, config.members, config.hidden, config.layers)
    model.fit_normalization(*data['train'])
    for member in range(config.members):
        _fit_member(model, member, data, config)
    model.eval()

    train_mean = data['train'][1].mean(0)
    report = {'config': dataclasses.asdict(config), 'episodes': split, 'rows': {k: len(v[0]) for k, v in data.items()},
              **{name: _evaluate(model, *data[name], train_mean) for name in ('val', 'test')}}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / 'ensemble.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_csv')
    parser.add_argument('out_dir')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    report = train(args.dataset_csv, args.out_dir, TrainConfig(seed=args.seed))
    print(json.dumps({k: report[k] for k in ('rows', 'val', 'test')}, indent=2))


if __name__ == '__main__':
    main()
