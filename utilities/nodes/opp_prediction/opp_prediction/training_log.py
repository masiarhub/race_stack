"""TensorBoard log of the ensemble training. Each member is one run with the same tags, so TensorBoard overlays them."""
import pathlib
import shutil

import torch

BOUND_MARGIN = 0.1      # a log-variance this close to a learned bound counts as sitting at the bound
COVERAGE_LEVEL = 0.9


class TrainingLog:
    """Logs on the validation data only. With directory None every method does nothing."""

    def __init__(self, directory, action_names, histogram_every: int = 10):
        self.directory = None if directory is None else pathlib.Path(directory)
        self.action_names, self.histogram_every = action_names, histogram_every
        self.writers = {}
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)  # event files of an older run would mix into the charts

    def _writer(self, run: str):
        from torch.utils.tensorboard import SummaryWriter
        if run not in self.writers:
            self.writers[run] = SummaryWriter(str(self.directory / run))
        return self.writers[run]

    @torch.no_grad()
    def epoch(self, model, member: int, epoch: int, train_loss: float, val_nll: float, inputs, actions):
        if self.directory is None:
            return
        writer = self._writer(f'member_{member}')
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('nll/val', val_nll, epoch)

        mu, sigma = (values[member] for values in model.distribution(inputs))
        network = model.members[member]
        logvar = 2.0 * torch.log(sigma / model.action_std)
        cdf = torch.distributions.Normal(mu, sigma).cdf(actions)
        for i, name in enumerate(self.action_names):
            at_bound = (logvar[:, i] - network.min_logvar[i] < BOUND_MARGIN) | \
                       (network.max_logvar[i] - logvar[:, i] < BOUND_MARGIN)
            writer.add_scalar(f'rmse/{name}', float(((mu[:, i] - actions[:, i]) ** 2).mean().sqrt()), epoch)
            writer.add_scalar(f'sigma_mean/{name}', float(sigma[:, i].mean()), epoch)
            writer.add_scalar(f'logvar_bound/min/{name}', float(network.min_logvar[i]), epoch)
            writer.add_scalar(f'logvar_bound/max/{name}', float(network.max_logvar[i]), epoch)
            writer.add_scalar(f'bound_share/{name}', float(at_bound.float().mean()), epoch)
            writer.add_scalar(f'coverage_90/{name}',
                              float(((cdf[:, i] - 0.5).abs() <= COVERAGE_LEVEL / 2).float().mean()), epoch)
            if epoch % self.histogram_every == 0:
                writer.add_histogram(f'mu/{name}', mu[:, i], epoch)
                writer.add_histogram(f'sigma/{name}', sigma[:, i], epoch)

    def member_done(self, member: int, best_epoch: int):
        if self.directory is not None:
            self._writer(f'member_{member}').add_scalar('early_stop/best_epoch', best_epoch, 0)

    def ensemble(self, results: dict, config: dict):
        """results: the validation results of the finished ensemble, as reported by the training."""
        if self.directory is None:
            return
        writer = self._writer('ensemble')
        writer.add_scalar('nll', results['nll'], 0)
        for name in self.action_names:
            writer.add_scalar(f'rmse/{name}', results['rmse'][name], 0)
            writer.add_scalar(f'aleatoric_std/{name}', results['aleatoric_std'][name], 0)
            writer.add_scalar(f'epistemic_std/{name}', results['epistemic_std'][name], 0)
            writer.add_scalar(f'coverage_90/{name}', results['coverage'][str(COVERAGE_LEVEL)][name], 0)
        writer.add_hparams(config, {'hparam/val_nll': results['nll']}, run_name='.')

    def close(self):
        for writer in self.writers.values():
            writer.close()
