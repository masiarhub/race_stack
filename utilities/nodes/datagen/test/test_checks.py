import numpy as np
import pytest

from datagen.checks import step_over_velocity


def test_positions_and_velocities_that_agree_give_one():
    t = np.arange(0, 5, 0.05)
    s, sdot = 1.4 * t, np.full(t.size, 1.4)
    assert step_over_velocity(s, sdot, 0.05) == pytest.approx(1.0)


def test_slow_simulator_on_a_wall_clock_grid_gives_short_steps():
    # the physics advances only 0.9 s per wall second, the velocities stay in simulated m/s
    t_wall = np.arange(0, 5, 0.05)
    s, sdot = 1.4 * 0.9 * t_wall, np.full(t_wall.size, 1.4)
    assert step_over_velocity(s, sdot, 0.05) == pytest.approx(0.9)


def test_jumps_are_ignored():
    t = np.arange(0, 5, 0.05)
    s, sdot = 1.4 * t, np.full(t.size, 1.4)
    s[50:] += 3.0  # e.g. an s jump in an ambiguous Frenet corner
    assert step_over_velocity(s, sdot, 0.05) == pytest.approx(1.0)


def test_shift_longer_than_the_episode_gives_nan():
    from datagen.checks import _shifted
    a = np.array([1.0, 2.0])
    assert np.isnan(_shifted(a, 3)).all() and np.isnan(_shifted(a, -3)).all()
    assert _shifted(a, 1).tolist()[0] == 2.0 and np.isnan(_shifted(a, 1)[1])
