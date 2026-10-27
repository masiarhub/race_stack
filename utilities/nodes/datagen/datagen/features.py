"""
Catalog of features (x) and labels (y). The representation is chosen by the dataset section of the spec:

    frame: frenet | cartesian     positions/velocities as (s, d, vs, vd) or map-frame (x, y, vx, vy)
    reference: centerline | raceline | shortest_path   Frenet reference line (also used for curvature)

Each feature returns a list of (column, values, unit, description). Add a feature by decorating a function
with @feature; it receives the Context and the options given in the spec ({curvature: {n: 10}}).
"""
import numpy as np

FEATURES = {}
LABELS = {}


class Context:
    """Per-episode arrays on the time grid. opp comes from the configured x source, opp_gt / ego from the sim."""

    def __init__(self, frame, ref, opp, opp_gt, ego, ego_cmd, opp_cmd):
        self.frame, self.ref = frame, ref
        self.opp, self.opp_gt, self.ego = opp, opp_gt, ego  # dicts with s, s_unwrapped, d, vs, vd, x, y, vx, vy
        self.ego_cmd, self.opp_cmd = ego_cmd, opp_cmd       # dicts with steer, speed


def feature(name):
    def register(fn):
        FEATURES[name] = fn
        return fn
    return register


def label(name):
    def register(fn):
        LABELS[name] = fn
        return fn
    return register


def _pos(ctx, agent, who):
    if ctx.frame == 'frenet':
        return [(f'{who}_s', agent['s'], 'm', f'{who} arc length on the reference line'),
                (f'{who}_d', agent['d'], 'm', f'{who} lateral offset, positive left')]
    return [(f'{who}_x', agent['x'], 'm', f'{who} x, map frame'),
            (f'{who}_y', agent['y'], 'm', f'{who} y, map frame')]


def _vel(ctx, agent, who):
    if ctx.frame == 'frenet':
        return [(f'{who}_vs', agent['vs'], 'm/s', f'{who} velocity along the reference line'),
                (f'{who}_vd', agent['vd'], 'm/s', f'{who} lateral velocity')]
    return [(f'{who}_vx', agent['vx'], 'm/s', f'{who} x velocity, map frame'),
            (f'{who}_vy', agent['vy'], 'm/s', f'{who} y velocity, map frame')]


def _frenet_only(ctx, name):
    if ctx.frame != 'frenet':
        raise ValueError(f"feature '{name}' needs frame: frenet")


@feature('rel_pos')
def rel_pos(ctx):
    if ctx.frame == 'frenet':
        return [('delta_s', ctx.ref.wrap_ds(ctx.opp['s'] - ctx.ego['s']), 'm', 's_opp - s_ego, wrapped to [-L/2, L/2)'),
                ('delta_d', ctx.opp['d'] - ctx.ego['d'], 'm', 'd_opp - d_ego')]
    return [('delta_x', ctx.opp['x'] - ctx.ego['x'], 'm', 'x_opp - x_ego, map frame'),
            ('delta_y', ctx.opp['y'] - ctx.ego['y'], 'm', 'y_opp - y_ego, map frame')]


@feature('rel_vel')
def rel_vel(ctx):
    keys = ('vs', 'vd') if ctx.frame == 'frenet' else ('vx', 'vy')
    return [(f'delta_{k}', ctx.opp[k] - ctx.ego[k], 'm/s', f'{k}_opp - {k}_ego') for k in keys]


@feature('opp_pos')
def opp_pos(ctx):
    return _pos(ctx, ctx.opp, 'opp')


@feature('opp_d')
def opp_d(ctx):
    _frenet_only(ctx, 'opp_d')
    return [('opp_d', ctx.opp['d'], 'm', 'opp lateral offset, positive left')]


@feature('opp_vel')
def opp_vel(ctx):
    return _vel(ctx, ctx.opp, 'opp')


@feature('ego_pos')
def ego_pos(ctx):
    return _pos(ctx, ctx.ego, 'ego')


@feature('ego_d')
def ego_d(ctx):
    _frenet_only(ctx, 'ego_d')
    return [('ego_d', ctx.ego['d'], 'm', 'ego lateral offset, positive left')]


@feature('ego_vel')
def ego_vel(ctx):
    return _vel(ctx, ctx.ego, 'ego')


@feature('ego_cmd')
def ego_cmd(ctx):
    return [('ego_steer_cmd', ctx.ego_cmd['steer'], 'rad', 'ego commanded steering angle'),
            ('ego_speed_cmd', ctx.ego_cmd['speed'], 'm/s', 'ego commanded speed')]


@feature('curvature')
def curvature(ctx, n=10, step_m=0.5):
    """Look-ahead curvature of the reference line ahead of the opponent (Zhu et al.)."""
    return [(f'kappa_{i}', ctx.ref.interp(ctx.ref.kappa, ctx.opp['s'] + i * step_m), '1/m',
             f'reference curvature {i * step_m:g} m ahead of the opponent') for i in range(1, n + 1)]


@feature('free_space')
def free_space(ctx):
    return [('opp_free_left', ctx.ref.interp(ctx.ref.d_left, ctx.opp['s']) - ctx.opp['d'], 'm',
             'distance from the opponent to the left track bound'),
            ('opp_free_right', ctx.ref.interp(ctx.ref.d_right, ctx.opp['s']) + ctx.opp['d'], 'm',
             'distance from the opponent to the right track bound')]


def _shift(a, i):
    """a[k + i] at row k, NaN past the end."""
    out = np.full_like(a, np.nan, dtype=float)
    out[:len(a) - i] = a[i:]
    return out


def _state(ctx):
    gt = ctx.opp_gt
    if ctx.frame == 'frenet':
        return [('s', gt['s_unwrapped'], 'm'), ('d', gt['d'], 'm'), ('vs', gt['vs'], 'm/s'), ('vd', gt['vd'], 'm/s')]
    return [('x', gt['x'], 'm'), ('y', gt['y'], 'm'), ('vx', gt['vx'], 'm/s'), ('vy', gt['vy'], 'm/s')]


@label('state_delta')
def state_delta(ctx, horizon=1):
    """z_{k+i} - z_k of the opponent ground truth, i = 1..horizon (s unwrapped, so no jump at the finish line)."""
    return [(f'd{name}_{i}', _shift(z, i) - z, unit, f'opp {name}(k+{i}) - {name}(k), ground truth')
            for i in range(1, horizon + 1) for name, z, unit in _state(ctx)]


@label('state')
def state(ctx, horizon=1):
    cols = []
    for i in range(1, horizon + 1):
        for name, z, unit in _state(ctx):
            z = ctx.ref.wrap_s(z) if name == 's' else z
            cols.append((f'{name}_{i}', _shift(z, i), unit, f'opp {name}(k+{i}), ground truth'))
    return cols


@label('action')
def action(ctx, horizon=1):
    """Opponent commands u_{k+i}, i = 0..horizon-1, as sent to the simulator (zero order hold)."""
    return [col for i in range(horizon) for col in (
        (f'steer_{i}', _shift(ctx.opp_cmd['steer'], i), 'rad', f'opp commanded steering angle at k+{i}'),
        (f'speed_{i}', _shift(ctx.opp_cmd['speed'], i), 'm/s', f'opp commanded speed at k+{i}'))]


def parse_entry(entry):
    """'name' or {name: {options}} -> (name, options)"""
    if isinstance(entry, str):
        return entry, {}
    (name, opts), = entry.items()
    return name, dict(opts or {})
