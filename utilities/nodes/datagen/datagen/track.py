"""
Closed reference lines (centerline / raceline) of a map and vectorized Frenet conversions.

Convention as in the race stack: s along the line, d positive to the left of the driving direction.
"""
import json

import numpy as np
from scipy.interpolate import CubicSpline

# names of the lines in global_waypoints.json
LINE_KEYS = {
    'centerline': 'centerline_waypoints',
    'raceline': 'global_traj_wpnts_iqp',
    'shortest_path': 'global_traj_wpnts_sp',
}


class Track:
    def __init__(self, wpnts: list):
        x = np.array([w['x_m'] for w in wpnts])
        y = np.array([w['y_m'] for w in wpnts])
        # the json closes the loop by repeating the first point, the wrap is handled explicitly here
        if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
            wpnts = wpnts[:-1]
            x, y = x[:-1], y[:-1]
        self.x, self.y = x, y
        self.kappa = np.array([w['kappa_radpm'] for w in wpnts])
        self.vx = np.array([w['vx_mps'] for w in wpnts])
        self.d_left = np.array([w['d_left'] for w in wpnts])
        self.d_right = np.array([w['d_right'] for w in wpnts])

        # periodic spline through the waypoints, parametrized by the chord length (~ arc length at 0.1 m spacing).
        # A plain polyline makes s jump by up to |d| * (heading change) at every vertex.
        self.seg_dx = np.roll(x, -1) - x
        self.seg_dy = np.roll(y, -1) - y
        self.seg_len = np.hypot(self.seg_dx, self.seg_dy)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(np.sum(self.seg_len))
        s_closed = np.append(self.s, self.length)
        self.spline_x = CubicSpline(s_closed, np.append(x, x[0]), bc_type='periodic')
        self.spline_y = CubicSpline(s_closed, np.append(y, y[0]), bc_type='periodic')

    @classmethod
    def from_json(cls, path: str, line: str = 'centerline') -> 'Track':
        with open(path) as f:
            data = json.load(f)
        return cls(data[LINE_KEYS[line]]['wpnts'])

    def wrap_s(self, s):
        return np.mod(s, self.length)

    def wrap_ds(self, ds):
        """Wraps a difference of s into [-L/2, L/2)."""
        return np.mod(np.asarray(ds) + self.length / 2, self.length) - self.length / 2

    def interp(self, field: np.ndarray, s):
        """Linear interpolation of a per-waypoint field at s (with wrap)."""
        s_ext = np.append(self.s, self.length)
        f_ext = np.append(field, field[0])
        return np.interp(self.wrap_s(s), s_ext, f_ext)

    def tangent_angle(self, s):
        s = self.wrap_s(s)
        return np.arctan2(self.spline_y(s, 1), self.spline_x(s, 1))

    def to_frenet(self, x, y, chunk: int = 4096, newton_iter: int = 3):
        """Closest point on the line: polyline projection as initial guess, refined with Newton on the spline."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        s = np.empty_like(x)
        for a in range(0, x.size, chunk):
            rx, ry = x[a:a + chunk, None] - self.x[None], y[a:a + chunk, None] - self.y[None]
            t = np.clip((rx * self.seg_dx + ry * self.seg_dy) / self.seg_len**2, 0.0, 1.0)
            i = np.argmin((rx - t * self.seg_dx)**2 + (ry - t * self.seg_dy)**2, axis=1)
            s[a:a + chunk] = self.s[i] + t[np.arange(i.size), i] * self.seg_len[i]
        for _ in range(newton_iter):
            ex, ey = self.spline_x(s) - x, self.spline_y(s) - y
            dx, dy = self.spline_x(s, 1), self.spline_y(s, 1)
            ddx, ddy = self.spline_x(s, 2), self.spline_y(s, 2)
            s = self.wrap_s(s - (ex * dx + ey * dy) / (dx**2 + dy**2 + ex * ddx + ey * ddy))
        dx, dy = self.spline_x(s, 1), self.spline_y(s, 1)
        rx, ry = x - self.spline_x(s), y - self.spline_y(s)
        d = (dx * ry - dy * rx) / np.hypot(dx, dy)  # positive on the left
        return s.reshape(shape), d.reshape(shape)

    def to_cartesian(self, s, d):
        s = self.wrap_s(np.asarray(s, dtype=float))
        psi = self.tangent_angle(s)
        return self.spline_x(s) - d * np.sin(psi), self.spline_y(s) + d * np.cos(psi)

    def frenet_velocity(self, s, vx_map, vy_map):
        """Projects map-frame velocities on the tangent / normal of the line at s."""
        psi = self.tangent_angle(s)
        vs = np.cos(psi) * vx_map + np.sin(psi) * vy_map
        vd = -np.sin(psi) * vx_map + np.cos(psi) * vy_map
        return vs, vd

    def cartesian_velocity(self, s, vs, vd):
        psi = self.tangent_angle(s)
        return np.cos(psi) * vs - np.sin(psi) * vd, np.sin(psi) * vs + np.cos(psi) * vd


def unwrap_s(s, length: float):
    """Turns a wrapped s time series into a continuous progress signal."""
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return s
    jumps = np.diff(s)
    jumps = np.mod(jumps + length / 2, length) - length / 2
    return np.concatenate([[s[0]], s[0] + np.cumsum(jumps)])


def body_to_map(yaw, vx_body, vy_body):
    return np.cos(yaw) * vx_body - np.sin(yaw) * vy_body, np.sin(yaw) * vx_body + np.cos(yaw) * vy_body
