import os

import pytest

HERE = os.path.dirname(__file__)
MAPS = os.path.realpath(os.path.join(HERE, '..', '..', '..', '..', 'stack_master', 'maps'))
os.environ.setdefault('DATAGEN_MAPS_DIR', MAPS)

from datagen.sampling import sample, episode_rng  # noqa: E402
from datagen.spec import load_spec, plan_episode  # noqa: E402

SPEC = os.path.join(HERE, '..', 'specs', 'h2h_v1.yaml')


def test_sampling_is_seeded():
    dist = {'b': {'uniform': [0, 1]}, 'a': {'choice': ['x', 'y', 'z']}, 'c': 3}
    assert sample(dist, episode_rng(1, 2, 'p')) == sample(dist, episode_rng(1, 2, 'p'))
    assert sample(dist, episode_rng(1, 2, 'p')) != sample(dist, episode_rng(1, 3, 'p'))


def test_plan_is_deterministic_and_frozen():
    spec = load_spec(SPEC)
    a, b = plan_episode(spec, 4), plan_episode(spec, 4)
    assert a == b
    pol = a['opp_policy']
    assert pol['name'] == 'opp_line_follower'  # round robin over the 3 policies
    assert 0.25 <= pol['params']['speed_scale'] <= 0.4
    assert len(pol['policy_hash']) == 12
    assert 2.0 <= a['spawn']['gap_m'] <= 6.0


def test_bad_spec(tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('tracks: [nope]\nopponents: {count: 3, policies: [missing_policy]}\nego: {policy: ego_h2h_spliner}\n')
    with pytest.raises(ValueError) as e:
        load_spec(str(bad))
    msg = str(e.value)
    assert 'nope' in msg and 'count' in msg and 'missing_policy' in msg


def test_uniform_step():
    for i in range(20):
        x = sample({'uniform': [0.25, 0.4], 'step': 0.01}, episode_rng(0, i, 'p'))
        assert abs(x * 100 - round(x * 100)) < 1e-9 and 0.25 <= x <= 0.4
