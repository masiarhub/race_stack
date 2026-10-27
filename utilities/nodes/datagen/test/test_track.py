import numpy as np

from datagen.track import Track, unwrap_s


def circle(r=2.0, n=200):
    a = np.linspace(0, 2 * np.pi, n + 1)  # closed: last point repeats the first, like the json
    return Track([{'x_m': r * np.cos(t), 'y_m': r * np.sin(t), 'kappa_radpm': 1 / r, 'vx_mps': 1.0,
                   'd_left': 0.5, 'd_right': 0.5} for t in a])


def test_frenet_roundtrip():
    tr = circle()
    s = np.linspace(0, tr.length * 0.999, 50)
    d = np.linspace(-0.4, 0.4, 50)
    x, y = tr.to_cartesian(s, d)
    s2, d2 = tr.to_frenet(x, y)
    assert np.allclose(tr.wrap_ds(s2 - s), 0, atol=2e-3)
    assert np.allclose(d2, d, atol=2e-3)


def test_left_is_positive():
    tr = circle(r=2.0)  # counter clockwise: the inside (smaller radius) is on the left
    _, d = tr.to_frenet(1.5, 0.0)
    assert d > 0


def test_wrap_and_unwrap():
    tr = circle()
    L = tr.length
    assert np.isclose(tr.wrap_ds(L - 0.1 - 0.1), -0.2)
    s = np.mod(np.linspace(0, 2.5 * L, 400), L)
    assert np.isclose(unwrap_s(s, L)[-1], 2.5 * L, atol=1e-6)


def test_frenet_velocity():
    tr = circle()
    vs, vd = tr.frenet_velocity(0.0, 0.0, 1.0)  # at (r, 0) the tangent points +y
    assert np.isclose(vs, 1.0, atol=0.05) and np.isclose(vd, 0.0, atol=0.05)
