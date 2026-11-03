import numpy as np
import pytest

from datagen.extract import time_grid

T0 = 1_000_000_000_000  # [ns] start gate on the recording clock
MS = 1_000_000


def steps(n, wall_ms_per_step, first=T0):
    """Recording times of n physics steps after the start gate, one step every wall_ms_per_step."""
    return first + np.round(np.arange(1, n + 1) * wall_ms_per_step * MS).astype(np.int64)


def test_real_time_sim_matches_wall_clock():
    step_ns = steps(500, 10.0)  # 100 steps/s: simulated time runs as fast as the wall clock
    g = time_grid(T0, step_ns[-1], step_ns, dt=0.05, warmup_s=1.0, cut_end_s=0.0)
    assert np.allclose(np.diff(g.t_s), 0.05)
    assert g.t_s[0] == pytest.approx(1.0)
    assert np.allclose(np.diff(g.wall_ns) * 1e-9, 0.05, atol=1e-6)


def test_slow_sim_spaces_samples_in_simulated_time():
    step_ns = steps(500, 12.5)  # 80 steps/s: 1 s of wall time holds only 0.8 s of simulated time
    g = time_grid(T0, step_ns[-1], step_ns, dt=0.05, warmup_s=1.0, cut_end_s=0.0)
    assert np.allclose(np.diff(g.t_s), 0.05)
    assert np.allclose(np.diff(g.wall_ns) * 1e-9, 0.0625, atol=1e-6)  # 0.05 s simulated = 0.0625 s wall
    assert g.t_s[-1] <= 5.0 + 1e-9  # 500 steps of 10 ms


def test_cut_end_is_simulated_time():
    step_ns = steps(500, 12.5)
    g = time_grid(T0, step_ns[-1], step_ns, dt=0.05, warmup_s=0.0, cut_end_s=0.5)
    assert g.t_s[-1] < 4.5 + 1e-9
    assert g.t_s[-1] > 4.4


def test_ticks_before_the_start_gate_are_ignored():
    held = T0 - np.arange(1, 200)[::-1] * 10 * MS  # the bridge ticks during the hold, the physics does not step
    step_ns = np.concatenate([held, steps(300, 10.0)])
    g = time_grid(T0, step_ns[-1], step_ns, dt=0.05, warmup_s=0.0, cut_end_s=0.0)
    assert g.t_s[0] == 0.0 and g.wall_ns[0] == T0
    assert g.t_s[-1] <= 3.0 + 1e-9


def test_without_physics_steps_falls_back_to_wall_clock():
    g = time_grid(T0, T0 + 3 * 10**9, np.zeros(0, dtype=np.int64), dt=0.05, warmup_s=1.0, cut_end_s=0.5)
    assert np.allclose((g.wall_ns - T0) * 1e-9, g.t_s)
    assert g.t_s[0] == pytest.approx(1.0) and g.t_s[-1] < 2.5


def test_episode_shorter_than_warmup_gives_empty_grid():
    step_ns = steps(50, 10.0)
    g = time_grid(T0, step_ns[-1], step_ns, dt=0.05, warmup_s=2.0, cut_end_s=0.0)
    assert g.t_s.size == 0 and g.wall_ns.size == 0


def test_physics_steps_come_from_the_wall_topic_or_the_older_combined_topic():
    from datagen.extract import physics_steps_ns
    new = {'/sim/collision/walls': [(5, None), (7, None)], '/sim/collisions': []}
    old = {'/sim/collision/walls': [], '/sim/collisions': [(1, None), (2, None), (3, None)]}
    assert physics_steps_ns(new).tolist() == [5, 7]
    assert physics_steps_ns(old).tolist() == [1, 2, 3]
    assert physics_steps_ns({}).size == 0
