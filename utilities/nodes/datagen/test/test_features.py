import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from datagen.extract import PHYSICS_DT_S, agent_state, columns_of, dataset_config, label_columns, with_slip
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


def test_action_label_clips_steering_to_the_limit():
    ctx, _ = make_ctx()
    ctx.opp_cmd = {'steer': np.array([0.1, 0.6, -0.9]), 'speed': np.array([1.0, 2.0, 3.0])}
    cols = {c[0]: c[1] for c in LABELS['action'](ctx, steer_limit_rad=0.4)}
    assert np.allclose(cols['steer_0'], [0.1, 0.4, -0.4])
    assert np.allclose(cols['speed_0'], [1.0, 2.0, 3.0])


def test_action_label_keeps_raw_commands_without_a_limit():
    ctx, _ = make_ctx()
    ctx.opp_cmd = {'steer': np.array([0.1, 0.6, -0.9]), 'speed': np.zeros(3)}
    cols = {c[0]: c[1] for c in LABELS['action'](ctx)}
    assert np.allclose(cols['steer_0'], [0.1, 0.6, -0.9])


def test_label_options_of_the_spec_reach_the_label():
    ctx, _ = make_ctx()
    ctx.opp_cmd = {'steer': np.array([0.6]), 'speed': np.zeros(1)}
    cfg = dataset_config({'labels': {'type': 'action', 'horizon': 1, 'steer_limit_rad': 0.4}})
    cols = {c[0]: c[1] for c in label_columns(cfg, ctx)}
    assert np.allclose(cols['y_steer_0'], [0.4])


def h2h_dataset():
    with open(os.path.join(HERE, '..', 'specs', 'h2h_v1.yaml')) as f:
        return yaml.safe_load(f)['dataset']


def test_h2h_policy_input_has_19_values():
    ctx, _ = make_ctx()
    assert len(columns_of(dataset_config(h2h_dataset())['features'], ctx, FEATURES, 'x')) == 19


def test_h2h_steering_limit_matches_the_simulator():
    with open(os.path.join(HERE, '..', '..', '..', '..', 'stack_master', 'config', 'SIM', 'sim_params.yaml')) as f:
        sim = yaml.safe_load(f)
    # the gym only stops steering once s_max is passed, so the angle can overshoot it by one physics step
    assert h2h_dataset()['labels']['steer_limit_rad'] == pytest.approx(sim['s_max'] + sim['sv_max'] * PHYSICS_DT_S)


def odom_moving_at(speed):
    t = np.array([0, 10, 20], dtype=np.int64)
    return {'t': t, 'vx_b': np.full(3, speed), 'vy_b': np.zeros(3), 'yaw': np.zeros(3)}


def test_with_slip_turns_the_speed_into_true_body_velocities():
    msgs = [(t, SimpleNamespace(data=[0, 0, 0, 2.0, 0, 0, 0.2])) for t in (0, 10, 20)]
    odom = with_slip(odom_moving_at(2.0), msgs)
    assert np.allclose(odom['vx_b'], 2.0 * np.cos(0.2))
    assert np.allclose(odom['vy_b'], 2.0 * np.sin(0.2))


def test_with_slip_leaves_recordings_without_rollout_state_unchanged():
    odom = odom_moving_at(2.0)
    assert with_slip(odom, []) is odom
