"""
Distributions in spec / policy files. Any value can be replaced by one of:
    {uniform: [lo, hi]}       float in [lo, hi), add `step: 0.01` to quantize (e.g. for parameters with a step range)
    {int_uniform: [lo, hi]}   int in [lo, hi] (inclusive)
    {choice: [a, b, ...]}     one of the listed values
Everything else is taken as a constant.
"""
import numpy as np

DISTRIBUTIONS = ('uniform', 'int_uniform', 'choice')


def is_distribution(value) -> bool:
    return isinstance(value, dict) and set(value) - {'step'} <= set(DISTRIBUTIONS) \
        and len(set(value) & set(DISTRIBUTIONS)) == 1


def sample(value, rng: np.random.Generator):
    """Recursively replaces every distribution in value by one draw of rng."""
    if is_distribution(value):
        kind = next(k for k in value if k in DISTRIBUTIONS)
        arg = value[kind]
        if kind == 'uniform':
            x = float(rng.uniform(arg[0], arg[1]))
            return round(round(x / value['step']) * value['step'], 10) if 'step' in value else x
        if kind == 'int_uniform':
            return int(rng.integers(arg[0], arg[1], endpoint=True))
        return arg[int(rng.integers(len(arg)))]
    if isinstance(value, dict):
        # sorted keys: the draw order (and so the result) must not depend on how the file is written
        return {k: sample(value[k], rng) for k in sorted(value)}
    if isinstance(value, list):
        return [sample(v, rng) for v in value]
    return value


def episode_rng(seed: int, episode: int, stream: str) -> np.random.Generator:
    """Independent stream per (seed, episode, purpose), so adding a sampled field does not shift the others."""
    return np.random.default_rng([seed, episode, sum(ord(c) * 31**i for i, c in enumerate(stream)) % 2**32])
