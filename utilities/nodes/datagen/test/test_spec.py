import os

import pytest
import yaml

HERE = os.path.dirname(__file__)
MAPS = os.path.realpath(os.path.join(HERE, '..', '..', '..', '..', 'stack_master', 'maps'))
os.environ.setdefault('DATAGEN_MAPS_DIR', MAPS)

from datagen.sampling import sample, episode_rng  # noqa: E402
from datagen.spec import RECORD_TOPICS, load_spec, plan_episode  # noqa: E402

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


def test_records_what_the_rollout_needs():
    for topic in ('/sim/state/ego', '/sim/state/opp', '/rollout_state', '/opp_racecar/rollout_state',
                  '/local_waypoints'):
        assert topic in RECORD_TOPICS


def _spec(tmp_path, monkeypatch, policies):
    monkeypatch.setenv('DATAGEN_POLICY_PATH', os.path.join(HERE, '..', 'policies'))
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump({'tracks': ['glc_ot_ez', 'test'], 'episodes': 6, 'ego': {'policy': 'ego_h2h_spliner'},
                                    'opponents': {'count': 1, 'policies': policies}}))
    return str(path)


def test_an_opponent_can_be_limited_to_some_tracks(tmp_path, monkeypatch):
    spec = load_spec(_spec(tmp_path, monkeypatch, ['opp_line_follower', {'policy': 'opp_blocker', 'tracks': ['test']}]))
    pairs = [(p['track'], p['opp_policy']['name']) for p in (plan_episode(spec, i) for i in range(4))]
    assert pairs == [('glc_ot_ez', 'opp_line_follower'), ('test', 'opp_line_follower'), ('test', 'opp_blocker'),
                     ('glc_ot_ez', 'opp_line_follower')]  # round robin over the allowed pairs only


def test_opponent_tracks_must_be_spec_tracks(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='hangar_1905_v0'):
        load_spec(_spec(tmp_path, monkeypatch, [{'policy': 'opp_blocker', 'tracks': ['hangar_1905_v0']}]))
    with pytest.raises(ValueError, match='policy'):
        load_spec(_spec(tmp_path, monkeypatch, [{'tracks': ['test']}]))


def _spec_with_rules(tmp_path, monkeypatch, rules):
    path = _spec(tmp_path, monkeypatch, ['opp_line_follower'])
    spec = yaml.safe_load(open(path))
    spec['episode'] = rules
    with open(path, 'w') as f:
        yaml.safe_dump(spec, f)
    return path


def test_the_opponent_starts_ahead_by_default():
    spec = load_spec(SPEC)
    assert all(plan_episode(spec, i)['spawn']['gap_m'] > 0 for i in range(6))


def test_the_opponent_can_start_behind(tmp_path, monkeypatch):
    spec = load_spec(_spec_with_rules(tmp_path, monkeypatch, {'opp_start': {'choice': ['ahead', 'behind']},
                                                               'start_gap_m': {'uniform': [2.0, 6.0]}}))
    gaps = [plan_episode(spec, i)['spawn']['gap_m'] for i in range(20)]
    assert any(g < 0 for g in gaps) and any(g > 0 for g in gaps)  # signed: negative is behind the ego
    assert all(2.0 <= abs(g) <= 6.0 for g in gaps)


def test_opp_start_must_be_ahead_or_behind(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='opp_start'):
        load_spec(_spec_with_rules(tmp_path, monkeypatch, {'opp_start': {'choice': ['ahead', 'beside']}}))


def test_an_opponent_can_see_the_ego_perfectly(tmp_path, monkeypatch):
    spec = load_spec(_spec(tmp_path, monkeypatch, ['opp_h2h_spliner']))
    assert plan_episode(spec, 0)['opp_policy']['ground_truth_obstacles']
