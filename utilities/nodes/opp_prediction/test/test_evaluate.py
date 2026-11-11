import numpy as np
import pytest

import pandas as pd

from opp_prediction.evaluate import observed_frames, rollout_errors
from opp_prediction.predict import Particles
from opp_prediction.recording import Recording
from test_track import circle

TRACK = circle()
DT_NS = 50_000_000
STEP_NS = 10_000_000


def motion(n_steps=400):
    """A car driving along the circle at 1 m/s, one state per physics step (0.01 s), with its recording."""
    t = np.arange(n_steps, dtype=np.int64) * STEP_NS
    s = 1.0 * t * 1e-9
    x, y = TRACK.to_cartesian(s, 0.0)
    yaw = TRACK.tangent_angle(s)
    states = np.stack([x, y, np.zeros(n_steps), np.ones(n_steps), yaw, np.zeros(n_steps), np.zeros(n_steps)], axis=1)
    return Recording(name='ep', track='t', grid_ns=np.arange(0, n_steps * STEP_NS, DT_NS, dtype=np.int64),
                     opp_t=t, opp_states=states, cmd_t=t, commands=np.tile([0.0, 1.0], (n_steps, 1)),
                     plan_t=t[:1], plan_x=x[:1:][None].repeat(3, 1), plan_y=y[:1:][None].repeat(3, 1),
                     plan_speed=np.ones((1, 3)))


class PerfectPredictor:
    """Returns the recorded future of the car for every particle."""

    def __init__(self, recording, offset=0.0):
        self.rec, self.offset = recording, offset
        self.past_inputs = []

    def predict(self, xi_0, ego_plan, track, horizon, previous_steer, past_inputs=None):
        self.past_inputs.append(past_inputs)
        start = int(np.argmin(np.hypot(*(self.rec.opp_states[:, :2] - np.asarray(xi_0)[:2]).T)))
        states = self.rec.opp_states[start::5][:horizon + 1].copy()
        states[:, 0] += self.offset
        return Particles(np.tile(states[None], (4, 1, 1)), np.array([0, 0, 1, 1]))


def test_a_perfect_predictor_has_no_error():
    rec = motion()
    errors = rollout_errors(rec, PerfectPredictor(rec), TRACK, horizon=10, start_rows=np.array([0, 5]))
    assert errors.particles.shape == (2, 4, 10, 2)
    assert np.allclose(errors.particles, 0.0, atol=0.01)


def test_constant_velocity_is_exact_for_constant_motion_along_the_line():
    rec = motion()
    errors = rollout_errors(rec, PerfectPredictor(rec), TRACK, horizon=10, start_rows=np.array([0]))
    assert errors.baseline.shape == (1, 10, 2)
    assert np.allclose(errors.baseline, 0.0, atol=0.02)


def test_prediction_errors_are_along_the_track_first():
    rec = motion()
    # on the circle at (2, 0) the +x offset points radially outward, so it is a lateral error
    errors = rollout_errors(rec, PerfectPredictor(rec, offset=0.1), TRACK, horizon=1, start_rows=np.array([0]))
    assert errors.particles[0, 0, 0, 1] == pytest.approx(-0.1, abs=0.02)


def test_steps_past_the_end_of_the_recording_have_no_error_value():
    rec = motion(n_steps=60)
    errors = rollout_errors(rec, PerfectPredictor(rec), TRACK, horizon=20, start_rows=np.array([0]))
    assert np.isnan(errors.particles[0, 0, -1, 0])
    assert not np.isnan(errors.particles[0, 0, 0, 0])
    assert np.isnan(errors.baseline[0, -1, 0])


def test_each_start_row_hands_its_observed_frames_to_the_predictor():
    rec = motion()
    predictor = PerfectPredictor(rec)
    past = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    rollout_errors(rec, predictor, TRACK, horizon=2, start_rows=np.array([0, 5]), past_inputs=past)
    assert [p.tolist() for p in predictor.past_inputs] == past.tolist()


def test_the_observed_frames_of_a_row_are_its_history_columns_oldest_first():
    rows = pd.DataFrame({'x_a_m2': [1.0], 'x_b_m2': [2.0], 'x_a_m1': [3.0], 'x_b_m1': [4.0], 'x_a': [5.0], 'x_b': [6.0]})
    frames = observed_frames(rows, list(rows.columns), history=3)
    assert frames.tolist() == [[[1.0, 2.0], [3.0, 4.0]]]
    assert observed_frames(rows[['x_a', 'x_b']], ['x_a', 'x_b'], history=1).shape == (1, 0, 2)
