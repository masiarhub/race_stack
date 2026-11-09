import os

import numpy as np
import yaml

from datagen.rollout import RolloutModel

HERE = os.path.dirname(__file__)
STEER, VEL = 2, 3


def model():
    with open(os.path.join(HERE, '..', '..', '..', '..', 'stack_master', 'config', 'SIM', 'sim_params.yaml')) as f:
        return RolloutModel({k: float(v) for k, v in yaml.safe_load(f).items()})


def driving(v=2.0):
    return np.array([0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0])  # x, y, steer, vel, yaw, yaw_rate, slip


def test_run_returns_the_full_state_after_each_physics_step():
    states = model().run(driving(), [0.0, 0.0], [(0.0, 2.0)] * 4)
    assert states.shape == (4, 7)
    assert states[-1, 0] > states[0, 0] > 0.0  # drives along x


def test_a_steering_command_takes_effect_after_two_physics_steps():
    states = model().run(driving(), [0.0, 0.0], [(0.3, 2.0)] * 4)
    assert np.allclose(states[:2, STEER], 0.0)
    assert states[2, STEER] > 0.0


def test_prediction_step_holds_the_action_and_delays_the_previous_steering():
    m = model()
    held = m.run(driving(), [0.1, 0.1], [(0.3, 2.5)] * 5)[-1]
    assert np.array_equal(m.predict_step(driving(), 0.1, (0.3, 2.5)), held)


def test_previous_steering_changes_the_prediction_step():
    m = model()
    assert not np.allclose(m.predict_step(driving(), 0.3, (0.0, 2.0)), m.predict_step(driving(), 0.0, (0.0, 2.0)))
