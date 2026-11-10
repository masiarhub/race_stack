"""
Trajectory metrics, computed per step of the horizon and separately along (s) and across (d) the track.

Errors are particle minus truth, as [samples, particles, horizon, 2], NaN where the true position is not known.
"""
import warnings

import numpy as np

COVERAGE_LEVELS = (0.5, 0.9)
DIMENSIONS = ('s', 'd')
SPARSIFICATION_FRACTIONS = (0.0, 0.1, 0.25, 0.5)


def _nanmean(values, axis):
    """Mean ignoring NaN, and NaN without a warning where a step has no sample at all."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def accuracy(errors) -> dict:
    """Absolute error of the particle mean at each step (fde) and averaged up to each step (ade), [horizon, 2]."""
    absolute = np.abs(_nanmean(errors, axis=1))
    fde = _nanmean(absolute, axis=0)
    return {'fde': fde, 'ade': np.cumsum(fde, axis=0) / np.arange(1, len(fde) + 1)[:, None]}


def min_fde(errors) -> np.ndarray:
    """Distance of the closest particle at each step, averaged over the samples, [horizon]. Rewards wide spread."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(np.nanmin(np.hypot(errors[..., 0], errors[..., 1]), axis=1), axis=0)


def coverage_expected(particles: int) -> dict:
    """Share of the ranks 0..particles inside each central interval, the coverage of a calibrated predictor."""
    centres = (np.arange(particles + 1) + 0.5) / (particles + 1)
    return {str(level): float(np.mean(np.abs(centres - 0.5) <= level / 2)) for level in COVERAGE_LEVELS}


def calibration(errors) -> dict:
    """Where the truth falls among the sorted particles: histogram of the ranks and coverage of central intervals."""
    particles = errors.shape[1]
    valid = ~np.isnan(errors).any(axis=1)                  # [samples, horizon, 2]
    ranks = np.sum(errors < 0.0, axis=1)                   # number of particles below the truth
    horizon, dims = errors.shape[2:]
    histogram = np.zeros((horizon, dims, particles + 1), dtype=int)
    coverage = {str(level): np.zeros((horizon, dims)) for level in COVERAGE_LEVELS}
    centres = (ranks + 0.5) / (particles + 1)
    for h in range(horizon):
        for dim in range(dims):
            selected = ranks[:, h, dim][valid[:, h, dim]]
            histogram[h, dim] = np.bincount(selected, minlength=particles + 1)
            for level in COVERAGE_LEVELS:
                coverage[str(level)][h, dim] = np.mean(np.abs(centres[:, h, dim][valid[:, h, dim]] - 0.5) <= level / 2)
    return {'rank_histogram': histogram, 'coverage': coverage}


def spread(errors, members) -> dict:
    """Std of the particles per sample, [samples, horizon, 2]: within a member, between members, and combined."""
    means, variances = [], []
    for member in np.unique(members):
        group = errors[:, members == member]
        means.append(group.mean(axis=1))
        variances.append(group.var(axis=1))
    aleatoric = np.mean(variances, axis=0)
    epistemic = np.var(means, axis=0)
    return {'aleatoric': np.sqrt(aleatoric), 'epistemic': np.sqrt(epistemic), 'total': np.sqrt(aleatoric + epistemic)}


def sparsification(error, uncertainty, fractions=SPARSIFICATION_FRACTIONS) -> np.ndarray:
    """Mean error after removing the most uncertain samples, [horizon, 2, fractions]. error: [samples, horizon, 2]."""
    horizon, dims = error.shape[1:]
    remaining = np.zeros((horizon, dims, len(fractions)))
    for h in range(horizon):
        for dim in range(dims):
            valid = ~np.isnan(error[:, h, dim])
            ordered = error[:, h, dim][valid][np.argsort(uncertainty[:, h, dim][valid])]
            for f, fraction in enumerate(fractions):
                keep = max(1, int(round(len(ordered) * (1.0 - fraction))))
                remaining[h, dim, f] = ordered[:keep].mean()
    return remaining


def _per_dimension(values) -> dict:
    return {name: float(values[i]) for i, name in enumerate(DIMENSIONS)}


def summarize(errors, members, baseline, steps, dt: float) -> dict:
    """The primary metrics at the given steps (1-based), against a deterministic baseline [samples, horizon, 2]."""
    acc, calib, spreads = accuracy(errors), calibration(errors), spread(errors, members)
    closest = min_fde(errors)
    point_error = np.abs(_nanmean(errors, axis=1))
    usefulness = {source: sparsification(point_error, value) for source, value in spreads.items()}
    oracle = sparsification(point_error, point_error)
    baseline_absolute = np.abs(baseline)
    baseline_fde = _nanmean(baseline_absolute, axis=0)
    baseline_ade = np.cumsum(baseline_fde, axis=0) / np.arange(1, len(baseline_fde) + 1)[:, None]

    report = {'particles': int(errors.shape[1]), 'coverage_expected': coverage_expected(errors.shape[1]),
              'sparsification_fractions': list(SPARSIFICATION_FRACTIONS), 'steps': {}}
    for step in steps:
        h = step - 1
        valid = ~np.isnan(errors[:, 0, h, 0])
        report['steps'][str(step)] = {
            'time_s': step * dt,
            'samples': int(valid.sum()),
            'fde': _per_dimension(acc['fde'][h]),
            'ade': _per_dimension(acc['ade'][h]),
            'baseline_fde': _per_dimension(baseline_fde[h]),
            'baseline_ade': _per_dimension(baseline_ade[h]),
            'min_fde': float(closest[h]),
            'coverage': {level: _per_dimension(value[h]) for level, value in calib['coverage'].items()},
            'spread_mean': {source: _per_dimension(_nanmean(value[:, h], axis=0)) for source, value in spreads.items()},
            'remaining_error': {source: {name: value[h, i].tolist() for i, name in enumerate(DIMENSIONS)}
                                for source, value in {**usefulness, 'oracle': oracle}.items()},
        }
    report['fde_curve'] = {name: acc['fde'][:, i].tolist() for i, name in enumerate(DIMENSIONS)}
    report['baseline_fde_curve'] = {name: baseline_fde[:, i].tolist() for i, name in enumerate(DIMENSIONS)}
    return report
