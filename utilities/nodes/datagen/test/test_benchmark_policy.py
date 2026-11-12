"""BenchmarkPlanner: the f1tenth_benchmarks planners (third-party submodule) as datagen python policies."""
import os

import numpy as np
import pytest

from datagen.policy import AgentState, Obs, load_policy_class
from datagen.track import Track

ENTRYPOINT = os.path.join(os.path.dirname(__file__), '..', 'policies', 'benchmark.py') + ':BenchmarkPlanner'
BenchmarkPlanner = load_policy_class(ENTRYPOINT)
if not BenchmarkPlanner.available():
    pytest.skip('f1tenth_benchmarks submodule or its dependencies missing', allow_module_level=True)

R = 5.0  # [m] radius of the circular test track, counter clockwise
WHEELBASE_M = 0.33


def circle(n=400, vx=3.0):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return Track([{'x_m': R * np.cos(t), 'y_m': R * np.sin(t), 'kappa_radpm': 1 / R, 'vx_mps': vx,
                   'd_left': 1.0, 'd_right': 1.0} for t in a])


TRACKS = {'centerline': circle(), 'raceline': circle(), 'shortest_path': circle()}


def state(x=R, y=0.0, yaw=np.pi / 2, vx=3.0):
    return AgentState(x, y, yaw, vx, 0.0, 0.0, 0.0, 0.0, vx, 0.0)


def obs(scan=None, **kw):
    return Obs(t=0.0, opp=state(**kw), ego=state(x=-R, yaw=-np.pi / 2), scan=scan)


def make(params):
    return BenchmarkPlanner({'speed_scale': 1.0, 'overrides': {}, **params}, TRACKS, np.random.default_rng(0))


def test_pure_pursuit_on_the_line_steers_the_curvature():
    steer, speed = make({'planner': 'pure_pursuit', 'line': 'raceline'}).act(obs())
    assert steer == pytest.approx(np.arctan(WHEELBASE_M / R), abs=0.01)
    assert speed == pytest.approx(3.0)  # the line's speed profile, below upstream's speed limits


def test_speed_scale_scales_the_speed_only():
    steer1, speed1 = make({'planner': 'pure_pursuit', 'line': 'raceline'}).act(obs())
    steer2, speed2 = make({'planner': 'pure_pursuit', 'line': 'raceline', 'speed_scale': 0.5}).act(obs())
    assert steer2 == pytest.approx(steer1) and speed2 == pytest.approx(0.5 * speed1)


def test_scaled_pure_pursuit_leaves_upstreams_start_mode():
    # upstream drives straight at 4 m/s below 1 m/s; scaled to 0.25 it would never get faster than 1 m/s
    _, speed = make({'planner': 'pure_pursuit', 'line': 'raceline', 'speed_scale': 0.25}).act(obs(vx=0.5))
    assert speed > 1.0


def test_pure_pursuit_lookahead_follows_the_measured_speed():
    # the lookahead grows with speed: a scaled planner must see the true speed, or it cuts corners.
    # Off the line, since on a circle pure pursuit steers the same for any lookahead.
    full = make({'planner': 'pure_pursuit', 'line': 'raceline'}).act(obs(x=R - 0.3, vx=1.5))
    slow = make({'planner': 'pure_pursuit', 'line': 'raceline', 'speed_scale': 0.25}).act(obs(x=R - 0.3, vx=1.5))
    assert slow[0] == pytest.approx(full[0])


def test_follow_the_gap_steers_into_the_open_side():
    scan = np.full(1080, 1.0)
    scan[540:] = 8.0  # beams sweep right to left, the free space is on the left
    steer, speed = make({'planner': 'follow_the_gap'}).act(obs(scan=scan))
    assert steer > 0.05 and speed > 0


def test_needs_scan_only_for_follow_the_gap():
    assert make({'planner': 'follow_the_gap'}).needs_scan
    assert not make({'planner': 'pure_pursuit', 'line': 'raceline'}).needs_scan


def test_overrides_replace_upstream_params():
    p = make({'planner': 'follow_the_gap', 'overrides': {'corners_speed': 1.5, 'max_steer': 0.3}})
    assert p.planner.planner_params.corners_speed == 1.5 and p.planner.planner_params.max_steer == 0.3


def test_unknown_override_and_planner_are_errors():
    with pytest.raises(KeyError, match='corner_speed'):
        make({'planner': 'follow_the_gap', 'overrides': {'corner_speed': 1.5}})
    with pytest.raises(ValueError, match='mpcc'):
        make({'planner': 'mpcc'})


def test_leaves_the_working_directory_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make({'planner': 'follow_the_gap'})
    make({'planner': 'pure_pursuit', 'line': 'raceline'})
    assert os.getcwd() == str(tmp_path) and not os.listdir(tmp_path)
