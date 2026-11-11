import numpy as np
import pytest
import torch

from datagen.extract import history_columns
from opp_prediction.predict import EgoPlan, Predictor
from test_track import circle

FEATURES = ['rel_pos', 'opp_d', 'opp_vel', 'ego_vel', {'curvature': {'n': 2, 'step_m': 0.5}}, 'free_space']
COLUMNS = ['x_delta_s', 'x_delta_d', 'x_opp_d', 'x_opp_vs', 'x_opp_vd', 'x_ego_vs', 'x_ego_vd', 'x_kappa_1',
           'x_kappa_2', 'x_opp_free_left', 'x_opp_free_right']
TRACK = circle()
VS = COLUMNS.index('x_opp_vs')


class FakeEnsemble:
    """Member b always predicts steer 0.0 and speed 1 + b, with the given std."""

    def __init__(self, members=5, std=0.0, columns=COLUMNS):
        self.members, self.std, self.input_columns = [None] * members, std, columns
        self.inputs = []

    def distribution(self, o):
        self.inputs.append(o)
        mu = torch.stack([torch.tensor([0.0, 1.0 + b]).repeat(len(o), 1) for b in range(len(self.members))])
        return mu, torch.full_like(mu, self.std)


class SpyRollout:
    """Records the actions and moves the car 1 cm per step along x, the speed becomes the commanded one."""

    def __init__(self):
        self.calls = []

    def predict_step(self, state, previous_steer, action):
        self.calls.append((previous_steer, tuple(action)))
        out = np.array(state, dtype=float)
        out[0] += 0.01
        out[3] = action[1]
        return out


XI0 = np.array([2.0, 0.0, 0.0, 1.0, np.pi / 2, 0.0, 0.0])  # on the circle, heading along it
PLAN = EgoPlan(x=np.array([2.0, 1.99, 1.96]), y=np.array([-1.5, -1.3, -1.1]), speed=np.array([1.0, 1.0, 1.0]),
               measured_speed=1.0)


def predictor(ensemble=None, rollout=None, **kw):
    return Predictor(ensemble or FakeEnsemble(), rollout or SpyRollout(), FEATURES, seed=0, **kw)


def test_particles_hold_the_initial_state_and_every_step():
    result = predictor().predict(XI0, PLAN, TRACK, horizon=3)
    assert result.states.shape == (20, 4, 7)
    assert np.array_equal(result.states[:, 0], np.tile(XI0, (20, 1)))


def test_each_particle_keeps_one_member_for_the_whole_horizon():
    result = predictor().predict(XI0, PLAN, TRACK, horizon=4)
    assert np.array_equal(np.bincount(result.members), [4] * 5)
    for p, member in enumerate(result.members):
        assert np.allclose(result.states[p, 1:, 3], 1.0 + member)


def test_a_rollout_step_feeds_the_previous_steering_into_the_delay():
    rollout = SpyRollout()
    predictor(FakeEnsemble(members=1, std=0.1), rollout, particles=1).predict(XI0, PLAN, TRACK, horizon=2,
                                                                              previous_steer=0.25)
    assert rollout.calls[0][0] == 0.25
    assert rollout.calls[1][0] == pytest.approx(rollout.calls[0][1][0])


def test_previous_steering_defaults_to_the_current_steering_angle():
    rollout = SpyRollout()
    xi = XI0.copy()
    xi[2] = 0.3
    predictor(FakeEnsemble(members=1), rollout, particles=1).predict(xi, PLAN, TRACK, horizon=1)
    assert rollout.calls[0][0] == pytest.approx(0.3)


def steers(rho, horizon=3):
    rollout = SpyRollout()
    predictor(FakeEnsemble(members=1, std=1.0), rollout, particles=1, rho=rho).predict(XI0, PLAN, TRACK, horizon)
    return [c[1][0] for c in rollout.calls]


def test_noise_is_fixed_over_the_horizon_with_rho_one():
    assert np.allclose(steers(1.0), steers(1.0)[0])


def test_noise_is_redrawn_every_step_with_rho_zero():
    assert len(set(np.round(steers(0.0), 6))) == 3


def test_correlated_noise_keeps_the_learned_spread():
    rollout = SpyRollout()
    predictor(FakeEnsemble(members=1, std=2.0), rollout, particles=4000, rho=0.9).predict(XI0, PLAN, TRACK, horizon=8)
    last_step = np.array([c[1][0] for c in rollout.calls[-4000:]])
    assert last_step.std() == pytest.approx(2.0, rel=0.1)


def test_same_seed_gives_the_same_particles():
    a = predictor(FakeEnsemble(std=0.3)).predict(XI0, PLAN, TRACK, horizon=3)
    b = predictor(FakeEnsemble(std=0.3)).predict(XI0, PLAN, TRACK, horizon=3)
    assert np.array_equal(a.states, b.states)


def test_policy_input_uses_the_slip_angle():
    ensemble = FakeEnsemble(members=1)
    xi = XI0.copy()
    xi[4], xi[6] = np.pi / 2 - 0.2, 0.2  # the car points inward, it still moves along the line
    predictor(ensemble, particles=1).predict(xi, PLAN, TRACK, horizon=1)
    assert ensemble.inputs[0][0, VS].item() == pytest.approx(1.0, abs=0.02)
    assert ensemble.inputs[0][0, VS + 1].item() == pytest.approx(0.0, abs=0.02)


def test_policy_input_columns_must_match_the_ensemble():
    with pytest.raises(ValueError):
        predictor(FakeEnsemble(columns=['x_other'])).predict(XI0, PLAN, TRACK, horizon=1)


def history_ensemble(history=3):
    return FakeEnsemble(members=1, columns=history_columns(COLUMNS, history))


PAST = np.arange(2 * len(COLUMNS), dtype=float).reshape(2, len(COLUMNS))  # observed frames x_{k-2}, x_{k-1}


def current_frames(horizon):
    """The frames a predictor without history feeds to the ensemble, one per step."""
    ensemble = FakeEnsemble(members=1)
    predictor(ensemble, particles=1).predict(XI0, PLAN, TRACK, horizon=horizon)
    return [o[0].numpy() for o in ensemble.inputs]


def test_the_first_input_stacks_the_observed_frames_before_the_current_one():
    ensemble = history_ensemble()
    predictor(ensemble, particles=2).predict(XI0, PLAN, TRACK, horizon=1, past_inputs=PAST)
    expected = np.concatenate([PAST[0], PAST[1], current_frames(1)[0]])
    assert np.allclose(ensemble.inputs[0].numpy(), np.tile(expected, (2, 1)))


def test_the_history_moves_on_with_the_frames_of_the_particle():
    ensemble = history_ensemble()
    predictor(ensemble, particles=1).predict(XI0, PLAN, TRACK, horizon=3, past_inputs=PAST)
    frames = current_frames(3)
    assert np.allclose(ensemble.inputs[1][0].numpy(), np.concatenate([PAST[1], frames[0], frames[1]]))
    assert np.allclose(ensemble.inputs[2][0].numpy(), np.concatenate([frames[0], frames[1], frames[2]]))


def test_an_ensemble_with_history_needs_the_observed_frames():
    with pytest.raises(ValueError):
        predictor(history_ensemble()).predict(XI0, PLAN, TRACK, horizon=1)
    with pytest.raises(ValueError):
        predictor(history_ensemble()).predict(XI0, PLAN, TRACK, horizon=1, past_inputs=PAST[:1])


def test_ego_plan_follows_the_waypoints_in_time():
    n = 21
    s = np.linspace(0.0, 2.0, n)
    plan = EgoPlan(*TRACK.to_cartesian(s, 0.0), speed=np.full(n, 2.0), measured_speed=2.0)
    ego = plan.states(TRACK, np.array([0.0, 0.5]))
    assert ego['s'] == pytest.approx([0.0, 1.0], abs=0.02)
    assert ego['d'] == pytest.approx([0.0, 0.0], abs=0.02)
    assert ego['vs'] == pytest.approx([2.0, 2.0], abs=0.1)


def test_ego_plan_continues_at_constant_frenet_velocity_past_its_end():
    n = 21
    s = np.linspace(0.0, 2.0, n)
    plan = EgoPlan(*TRACK.to_cartesian(s, 0.0), speed=np.full(n, 2.0), measured_speed=2.0)
    ego = plan.states(TRACK, np.array([1.0, 2.0]))  # the plan ends after 1 s
    assert ego['s'] == pytest.approx([2.0, 4.0], abs=0.1)
    assert ego['d'] == pytest.approx([0.0, 0.0], abs=0.02)


def test_ego_plan_drives_its_speed_profile_scaled_to_the_measured_speed():
    n = 21
    s = np.linspace(0.0, 2.0, n)
    plan = EgoPlan(*TRACK.to_cartesian(s, 0.0), speed=np.linspace(2.0, 4.0, n), measured_speed=1.0)
    ego = plan.states(TRACK, np.array([0.0, 10.0]))  # the second time is past the end of the plan
    assert ego['vs'] == pytest.approx([1.0, 2.0], abs=0.05)
