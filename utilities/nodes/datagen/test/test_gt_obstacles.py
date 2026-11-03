import numpy as np
import pytest

from datagen.gt_obstacles import obstacle_from_state
from test_track import circle


def test_opponent_on_the_line_moving_along_it():
    tr = circle(r=2.0)
    x, y = tr.to_cartesian(2.0, 0.3)
    yaw = float(tr.tangent_angle(2.0))
    o = obstacle_from_state(tr, float(x), float(y), yaw, 1.5, 0.0, size_m=0.5)
    assert o['s_center'] == pytest.approx(2.0, abs=1e-2)
    assert o['d_center'] == pytest.approx(0.3, abs=1e-2)
    assert o['vs'] == pytest.approx(1.5, abs=1e-2)
    assert o['vd'] == pytest.approx(0.0, abs=1e-2)
    assert o['s_start'] == pytest.approx(1.75, abs=1e-2) and o['s_end'] == pytest.approx(2.25, abs=1e-2)
    assert o['d_right'] == pytest.approx(0.05, abs=1e-2) and o['d_left'] == pytest.approx(0.55, abs=1e-2)
    assert o['is_static'] is False and o['is_visible'] is True


def test_extent_wraps_at_the_finish_line():
    tr = circle(r=2.0)
    x, y = tr.to_cartesian(0.1, 0.0)
    o = obstacle_from_state(tr, float(x), float(y), float(tr.tangent_angle(0.1)), 1.0, 0.0, size_m=0.5)
    assert o['s_start'] == pytest.approx(tr.length - 0.15, abs=1e-2)
    assert 0.0 <= o['s_start'] < tr.length and 0.0 <= o['s_end'] < tr.length


def test_lateral_motion_shows_up_in_vd():
    tr = circle(r=2.0)
    x, y = tr.to_cartesian(1.0, 0.0)
    yaw = float(tr.tangent_angle(1.0)) + np.pi / 2  # heading straight to the left of the line
    o = obstacle_from_state(tr, float(x), float(y), yaw, 1.0, 0.0, size_m=0.5)
    assert o['vd'] == pytest.approx(1.0, abs=1e-2) and o['vs'] == pytest.approx(0.0, abs=1e-2)
