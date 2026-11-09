"""
Ensemble of probabilistic networks that models the opponent's policy pi(u_k | o_k).

Each member maps a policy input to a diagonal Gaussian over the action. The spread of one member is the aleatoric
uncertainty, the disagreement between members the epistemic uncertainty.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

LOGVAR_BOUND_WEIGHT = 0.01  # keeps the learned log-variance bounds tight (PETS, appendix A.1)


class _Member(nn.Module):
    """Normalized policy input -> mean and bounded log-variance of the normalized action."""

    def __init__(self, n_inputs: int, n_actions: int, hidden: int, layers: int):
        super().__init__()
        blocks, width = [], n_inputs
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.net = nn.Sequential(*blocks, nn.Linear(width, 2 * n_actions))
        self.max_logvar = nn.Parameter(torch.full((n_actions,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((n_actions,), -10.0))

    def forward(self, x):
        mean, raw = self.net(x).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw)
        return mean, self.min_logvar + F.softplus(logvar - self.min_logvar)


def _scale(std):
    return torch.where(std < 1e-6, torch.ones_like(std), std)


class Ensemble(nn.Module):
    def __init__(self, input_columns, members: int = 5, hidden: int = 256, layers: int = 3, n_actions: int = 2):
        super().__init__()
        self.input_columns = list(input_columns)
        self.architecture = {'members': members, 'hidden': hidden, 'layers': layers, 'n_actions': n_actions}
        n_inputs = len(self.input_columns)
        self.members = nn.ModuleList(_Member(n_inputs, n_actions, hidden, layers) for _ in range(members))
        self.register_buffer('input_mean', torch.zeros(n_inputs))
        self.register_buffer('input_std', torch.ones(n_inputs))
        self.register_buffer('action_mean', torch.zeros(n_actions))
        self.register_buffer('action_std', torch.ones(n_actions))

    def fit_normalization(self, inputs, actions):
        self.input_mean, self.input_std = inputs.mean(0), _scale(inputs.std(0))
        self.action_mean, self.action_std = actions.mean(0), _scale(actions.std(0))

    def _forward(self, member: int, inputs):
        return self.members[member]((inputs - self.input_mean) / self.input_std)

    def nll(self, member: int, inputs, actions):
        """Gaussian negative log-likelihood of one member, on normalized actions, averaged over the batch."""
        mean, logvar = self._forward(member, inputs)
        target = (actions - self.action_mean) / self.action_std
        return 0.5 * (((target - mean) ** 2 * torch.exp(-logvar) + logvar + math.log(2 * math.pi)).sum(-1)).mean()

    def loss(self, member: int, inputs, actions):
        """Training objective of one member: the likelihood plus the term that tightens the log-variance bounds."""
        m = self.members[member]
        return self.nll(member, inputs, actions) + LOGVAR_BOUND_WEIGHT * (m.max_logvar.sum() - m.min_logvar.sum())

    @torch.no_grad()
    def distribution(self, inputs):
        """inputs [N, n_inputs] -> mean and std of the action in physical units, each [members, N, n_actions]."""
        means, stds = [], []
        for b in range(len(self.members)):
            mean, logvar = self._forward(b, inputs)
            means.append(mean * self.action_std + self.action_mean)
            stds.append(torch.exp(0.5 * logvar) * self.action_std)
        return torch.stack(means), torch.stack(stds)

    def save(self, path):
        torch.save({'state': self.state_dict(), 'input_columns': self.input_columns, **self.architecture}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path)
        state = ckpt.pop('state')
        model = cls(ckpt.pop('input_columns'), **ckpt)
        model.load_state_dict(state)
        return model.eval()
