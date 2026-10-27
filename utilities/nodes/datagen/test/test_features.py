import os

import numpy as np
import pytest

from datagen.extract import agent_state, dataset_config
from datagen.features import FEATURES, LABELS, Context
from datagen.policy import AgentState, Obs, load_policy_class
from datagen.track import Track
from test_track import circle

HERE = os.path.dirname(__file__)


def make_ctx(frame='frenet'):
    tr = circle()
    t = np.arange(0, 5, 0.05)
    s_ego, s_opp = np.mod(1.0 * t, tr.length), np.mod(1.0 * t + 1.5, tr.length)
    ego = agent_state(tr, *tr.to_cartesian(s_ego, 0.0), *tr.cartesian_velocity(s_ego, 1.0, 0.0))
    opp = agent_state(tr, *tr.to_cartesian(s_opp, 0.2), *tr.cartesian_velocity(s_opp, 1.0, 0.0))
    cmd = {'steer': np.full(t.size, 0.1), 'speed': np.full(t.size, 1.0)}
    return Context(frame, tr, opp, opp, ego, cmd, cmd), tr


def test_rel_pos_wraps():
    ctx, tr = make_ctx()
    cols = {c[0]: c[1] for c in FEATURES['rel_pos'](ctx)}
    assert np.allclose(cols['delta_s'], 1.5, atol=0.02)  # also across the finish line
    assert np.allclose(cols['delta_d'], 0.2, atol=0.01)


def test_state_delta_has_no_jump_at_finish_line():
    ctx, _ = make_ctx()
    cols = {c[0]: c[1] for c in LABELS['state_delta'](ctx, horizon=2)}
    ds1 = cols['ds_1'][~np.isnan(cols['ds_1'])]
    assert np.allclose(ds1, 0.05, atol=0.005)
    assert np.isnan(cols['ds_2'][-2:]).all()


def test_cartesian_frame_columns():
    ctx, _ = make_ctx('cartesian')
    assert [c[0] for c in FEATURES['opp_vel'](ctx)] == ['opp_vx', 'opp_vy']
    with pytest.raises(ValueError):
        FEATURES['opp_d'](ctx)


def test_dataset_config_validation():
    assert dataset_config({'labels': {'horizon': 5}})['labels'] == {'type': 'state_delta', 'horizon': 5}
    with pytest.raises(ValueError):
        dataset_config({'features': ['nope']})


@pytest.mark.parametrize('entry', ['line_follower.py:LineFollower', 'blocker.py:Blocker'])
def test_example_policies_drive(entry):
    tr = circle(r=3.0)
    tracks = {'centerline': tr, 'raceline': tr, 'shortest_path': tr}
    params = {'line': 'centerline', 'd_offset_m': 0.0, 'speed_scale': 0.5, 'lookahead_m': 0.8,
              'block_range_m': 3.0, 'max_lat_rate_mps': 0.5, 'margin_m': 0.1}
    path, cls = entry.split(':')
    policy = load_policy_class(os.path.join(HERE, '..', 'policies', path) + ':' + cls)(params, tracks, np.random.default_rng(0))
    st = lambda s, d: AgentState(*tr.to_cartesian(s, d), float(tr.tangent_angle(s)) + np.pi / 2 * 0, 1.0, 0, 0, s, d, 1, 0)
    # on a counter clockwise circle the heading at s=0 (x=r, y=0) is +pi/2
    me = AgentState(3.0, 0.0, np.pi / 2, 0.5, 0, 0, 0.0, 0.0, 0.5, 0)
    steer, speed = policy.act(Obs(0.0, me, st(tr.length - 1.0, 0.3)))
    assert steer > 0  # turning left, into the circle
    assert np.isclose(speed, 0.5)
