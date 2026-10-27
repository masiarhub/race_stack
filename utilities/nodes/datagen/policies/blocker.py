"""Example python policy: blocks the ego by moving onto its lateral position when it is close behind."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from line_follower import LineFollower  # noqa: E402


class Blocker(LineFollower):
    def __init__(self, params, tracks, rng):
        super().__init__(params, tracks, rng)
        self.d_target = params['d_offset_m']
        self.t_prev = None

    def target_d(self, obs, s):
        p = self.params
        dt = 0.0 if self.t_prev is None else obs.t - self.t_prev
        self.t_prev = obs.t
        s_ego, d_ego = self.line.to_frenet(obs.ego.x, obs.ego.y)
        gap = self.line.wrap_ds(s - s_ego)  # > 0: ego behind
        goal = float(d_ego) if 0.0 < gap < p['block_range_m'] else p['d_offset_m']
        step = np.clip(goal - self.d_target, -p['max_lat_rate_mps'] * dt, p['max_lat_rate_mps'] * dt)
        self.d_target += step
        lo = -(self.line.interp(self.line.d_right, s) - p['margin_m'])
        hi = self.line.interp(self.line.d_left, s) - p['margin_m']
        self.d_target = float(np.clip(self.d_target, lo, hi))
        return self.d_target
