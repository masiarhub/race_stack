"""Example python policy: pure pursuit on a line with a constant lateral offset."""
import numpy as np

from datagen.policy import Obs, Policy

WHEELBASE_M = 0.33  # lf + lr of the sim car


class LineFollower(Policy):
    def __init__(self, params, tracks, rng):
        super().__init__(params, tracks, rng)
        self.line = tracks[params['line']]
        self.raceline = tracks['raceline']

    def target_d(self, obs: Obs, s: float) -> float:
        return self.params['d_offset_m']

    def act(self, obs: Obs):
        me = obs.opp
        s, _ = self.line.to_frenet(me.x, me.y)
        tx, ty = self.line.to_cartesian(s + self.params['lookahead_m'], self.target_d(obs, float(s)))
        alpha = np.arctan2(ty - me.y, tx - me.x) - me.yaw
        steer = np.arctan2(2 * WHEELBASE_M * np.sin(alpha), np.hypot(tx - me.x, ty - me.y))
        s_rl, _ = self.raceline.to_frenet(me.x, me.y)
        speed = self.params['speed_scale'] * self.raceline.interp(self.raceline.vx, s_rl)
        return float(steer), float(speed)
