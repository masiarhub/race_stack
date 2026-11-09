"""
Rollout model: the simulator's vehicle dynamics, as an exact copy built from the gym's own functions.

State: [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]. Commands: (steer, speed) as sent to the simulator.
The steering delay of the gym is memory of the model and not part of the state.
"""
import numpy as np

from datagen.extract import PHYSICS_DT_S

STEER_DELAY_STEPS = 2
PHYSICS_STEPS_PER_PREDICTION = 5


class RolloutModel:
    """Steps one car with the gym's single-track model (RK4, pid, steering delay)."""

    def __init__(self, params: dict):
        from f110_gym.envs.dynamic_models import pid, vehicle_dynamics_st
        self.p, self.pid, self.f = params, pid, vehicle_dynamics_st
        self.args = tuple(params[k] for k in ('mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max',
                                               'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max'))

    def run(self, state, steer_buffer, commands) -> np.ndarray:
        """commands: one (steer, speed) per physics step. Returns the state after each physics step, (n, 7)."""
        x, buf, out = np.array(state, dtype=float), list(steer_buffer), []
        p = self.p
        for raw_steer, speed in commands:
            steer = buf[-1]
            buf = [raw_steer] + buf[:-1]
            accl, sv = self.pid(speed, steer, x[3], x[2], p['sv_max'], p['a_max'], p['v_max'], p['v_min'])
            u = np.array([sv, accl])
            k1 = self.f(x, u, *self.args)
            k2 = self.f(x + PHYSICS_DT_S * k1 / 2, u, *self.args)
            k3 = self.f(x + PHYSICS_DT_S * k2 / 2, u, *self.args)
            k4 = self.f(x + PHYSICS_DT_S * k3, u, *self.args)
            x = x + PHYSICS_DT_S / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(x.copy())
        return np.array(out)

    def predict_step(self, state, previous_steer: float, action, physics_steps: int = PHYSICS_STEPS_PER_PREDICTION):
        """State after one prediction step, with the action held. The delay still holds the previous steering command."""
        return self.run(state, [previous_steer] * STEER_DELAY_STEPS, [action] * physics_steps)[-1]
