"""
Trajectory prediction: particles of the opponent's motion, sampled from the ensemble and rolled through the rollout
model. Each particle keeps one ensemble member for the whole horizon (TS-infinity), its action noise is correlated
over time by rho.

Rollout state: [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle].
"""
from typing import NamedTuple

import numpy as np
import torch

from datagen.extract import PHYSICS_DT_S, agent_state, columns_of
from datagen.features import FEATURES, Context
from datagen.rollout import PHYSICS_STEPS_PER_PREDICTION

PREDICTION_DT_S = PHYSICS_STEPS_PER_PREDICTION * PHYSICS_DT_S
MIN_PLAN_SPEED = 0.05  # [m/s] keeps the time between waypoints finite for a standing plan


class Particles(NamedTuple):
    states: np.ndarray   # [particles, horizon + 1, 7], the first step is the initial rollout state
    members: np.ndarray  # [particles], the ensemble member of each particle


class EgoPlan:
    """The ego's intended motion at prediction time: waypoints in the map frame with their speed."""

    def __init__(self, x, y, speed):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.speed = np.maximum(np.asarray(speed, dtype=float), MIN_PLAN_SPEED)

    def states(self, track, times) -> dict:
        """Ego state at each time, read from the plan and continued at constant Frenet velocity past its end."""
        steps = np.hypot(np.diff(self.x), np.diff(self.y))
        t_plan = np.concatenate([[0.0], np.cumsum(steps / (0.5 * (self.speed[:-1] + self.speed[1:])))])
        heading_x, heading_y = np.gradient(self.x, edge_order=2), np.gradient(self.y, edge_order=2)
        norm = np.hypot(heading_x, heading_y)
        plan = agent_state(track, self.x, self.y, self.speed * heading_x / norm, self.speed * heading_y / norm)

        past_end = np.maximum(times - t_plan[-1], 0.0)
        s = np.interp(times, t_plan, plan['s_unwrapped']) + past_end * plan['vs'][-1]
        d = np.interp(times, t_plan, plan['d']) + past_end * plan['vd'][-1]
        vs, vd = np.interp(times, t_plan, plan['vs']), np.interp(times, t_plan, plan['vd'])
        x, y = track.to_cartesian(s, d)
        vx, vy = track.cartesian_velocity(s, vs, vd)
        return {'x': x, 'y': y, 'vx': vx, 'vy': vy, 's': track.wrap_s(s), 'd': d, 'vs': vs, 'vd': vd}


def _project(track, rollout_states) -> dict:
    """The rollout state in the Frenet frame of the track, with the velocity along yaw + slip."""
    xi = np.asarray(rollout_states)
    course = xi[:, 4] + xi[:, 6]
    return agent_state(track, xi[:, 0], xi[:, 1], xi[:, 3] * np.cos(course), xi[:, 3] * np.sin(course))


class Predictor:
    def __init__(self, ensemble, rollout_model, features, particles: int = 20, rho: float = 0.0, seed=None):
        self.ensemble, self.rollout_model, self.features = ensemble, rollout_model, features
        self.particles, self.rho = particles, rho
        self.rng = np.random.default_rng(seed)

    def _policy_input(self, rollout_states, ego: dict, track):
        """The same features as the training data, built from the rollout state of every particle."""
        opp = _project(track, rollout_states)
        columns = columns_of(self.features, Context('frenet', track, opp, opp, ego, None, None), FEATURES, 'x')
        names = [c[0] for c in columns]
        if names != self.ensemble.input_columns:
            raise ValueError(f'policy input {names} does not match the ensemble input {self.ensemble.input_columns}')
        return torch.tensor(np.stack([c[1] for c in columns], axis=1), dtype=torch.float32)

    def predict(self, xi_0, ego_plan: EgoPlan, track, horizon: int = 60, previous_steer: float = None) -> Particles:
        """previous_steer: last steering command before the prediction, by default the current steering angle."""
        n, n_members = self.particles, len(self.ensemble.members)
        members = np.arange(n) % n_members
        ego_states = ego_plan.states(track, np.arange(horizon) * PREDICTION_DT_S)
        states = np.empty((n, horizon + 1, len(xi_0)))
        states[:, 0] = xi_0
        previous = np.full(n, xi_0[2] if previous_steer is None else previous_steer, dtype=float)
        noise = self.rng.standard_normal((n, 2))

        for k in range(horizon):
            ego = {key: np.full(n, value[k]) for key, value in ego_states.items()}
            mu, sigma = self.ensemble.distribution(self._policy_input(states[:, k], ego, track))
            mean, std = mu[members, np.arange(n)].numpy(), sigma[members, np.arange(n)].numpy()
            actions = mean + std * noise
            states[:, k + 1] = [self.rollout_model.predict_step(states[p, k], previous[p], tuple(actions[p]))
                                for p in range(n)]
            previous = actions[:, 0]
            noise = self.rho * noise + np.sqrt(1.0 - self.rho ** 2) * self.rng.standard_normal((n, 2))
        return Particles(states, members)
