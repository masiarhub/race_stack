"""
Experiment spec: loading, validation and the deterministic per-episode plan.
"""
import copy
import itertools
import os

import yaml

from datagen.policy import find_policy_file, load_policy_def, resolve_policy
from datagen.sampling import episode_rng, sample
from datagen.track import Track

# the gym bridge supports one opponent, see f1tenth_gym_ros num_agent
MAX_OPPONENTS = 1
MAP_FILES = ('global_waypoints.json', 'speed_scaling.yaml', 'ot_sectors.yaml')

EPISODE_DEFAULTS = {
    'laps': 2,                      # normal end: the ego covered laps * track length
    'spawn_frac': {'uniform': [0.0, 1.0]},  # ego spawn on the raceline, fraction of the lap
    'start_gap_m': {'uniform': [2.0, 6.0]},  # opponent spawns this far ahead of the ego along the raceline
    'warmup_s': 2.0,                # both cars start at rest, these seconds are dropped at extraction
    'settle_s': 1.0,                # all readiness checks must hold this long before the start gate opens
    'startup_timeout_s': 90.0,
    'timeout_s': 120.0,
    'stall': {'speed_mps': 0.3, 'duration_s': 2.0},
    'off_track_margin_m': 0.3,      # beyond the track bounds by more than this ends the episode
    'decouple': {'gap_m': 15.0, 'duration_s': 5.0},  # null disables it
    'stale_input_s': 1.0,           # a required topic silent this long after t=0 invalidates the episode
    'max_retries': 2,               # only for startup failures, with the same seed
}

RECORD_TOPICS = [
    '/state_estimation/odom',       # ego ground truth (sim)
    '/opp_racecar/odom',            # opponent ground truth (sim)
    '/perception/obstacles',        # ego's tracker estimate of the opponent
    '/drive',                       # ego commands
    '/opp_drive',                   # opponent commands (action labels)
    '/car_state/frenet/odom',       # ego Frenet odom (raceline) of the stack
    '/state',                       # ego state machine (e.g. trailing / overtaking)
    '/opp_racecar/state',           # opponent state machine (launch policies of the race stack only)
    '/sim/collisions',
    '/datagen/events',
    '/parameter_events',
]


def maps_dir() -> str:
    if 'DATAGEN_MAPS_DIR' in os.environ:
        return os.environ['DATAGEN_MAPS_DIR']
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('stack_master'), 'maps')


def load_spec(path: str) -> dict:
    path = os.path.realpath(path)
    with open(path) as f:
        spec = yaml.safe_load(f)
    spec['_path'] = path
    spec.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    spec.setdefault('seed', 0)
    spec.setdefault('episodes', 1)
    if isinstance(spec.get('tracks'), str):
        spec['tracks'] = [spec['tracks']]
    spec['episode'] = {**copy.deepcopy(EPISODE_DEFAULTS), **(spec.get('episode') or {})}
    spec.setdefault('record', {})
    spec['record'].setdefault('extra_topics', [])
    spec.setdefault('dataset', {})
    validate(spec)
    return spec


def validate(spec: dict):
    errors = []
    if not spec.get('tracks'):
        errors.append("tracks: list at least one map of stack_master/maps")
    else:
        for track in spec['tracks']:
            missing = [f for f in MAP_FILES + (f'{track}.yaml',) if not os.path.isfile(os.path.join(maps_dir(), track, f))]
            if missing:
                errors.append(f"track {track}: missing {missing} in {os.path.join(maps_dir(), track)}")
    opp = spec.get('opponents') or {}
    if opp.get('count', 1) != MAX_OPPONENTS:
        errors.append(f"opponents.count: the gym bridge supports exactly {MAX_OPPONENTS} opponent for now")
    if not opp.get('policies'):
        errors.append("opponents.policies: list at least one policy")
    if not (spec.get('ego') or {}).get('policy'):
        errors.append("ego.policy: missing")
    if int(spec['episodes']) < 1:
        errors.append("episodes: must be >= 1")
    # resolve every policy once, so a broken definition fails before any simulation
    spec_dir = os.path.dirname(spec['_path'])
    for role, ref in [('ego', spec.get('ego', {}).get('policy'))] + [('opp', r) for r in opp.get('policies') or []]:
        if ref:
            try:
                pdef = load_policy_def(find_policy_file(ref, spec_dir))
                if role == 'ego' and pdef['kind'] != 'launch':
                    errors.append(f"ego policy {ref}: only launch policies can drive the ego for now")
            except (FileNotFoundError, ValueError) as e:
                errors.append(str(e))
    if errors:
        raise ValueError("Invalid spec " + spec['_path'] + ":\n  - " + "\n  - ".join(errors))


def plan_episode(spec: dict, idx: int) -> dict:
    """Everything one episode needs, derived only from the spec and (seed, idx)."""
    spec_dir = os.path.dirname(spec['_path'])
    combos = list(itertools.product(spec['tracks'], spec['opponents']['policies']))
    track, opp_ref = combos[idx % len(combos)]
    seed = int(spec['seed'])

    rules = sample(spec['episode'], episode_rng(seed, idx, 'episode'))
    raceline = Track.from_json(os.path.join(maps_dir(), track, 'global_waypoints.json'), 'raceline')
    ego_policy = resolve_policy(load_policy_def(find_policy_file(spec['ego']['policy'], spec_dir)),
                                episode_rng(seed, idx, 'ego'))
    opp_policy = resolve_policy(load_policy_def(find_policy_file(opp_ref, spec_dir)),
                                episode_rng(seed, idx, 'opponent'))
    return {
        'id': idx,
        'name': f"ep_{idx:04d}",
        'seed': seed,
        'policy_seed': int(episode_rng(seed, idx, 'policy_runtime').integers(2**31)),
        'track': track,
        'map_dir': os.path.join(maps_dir(), track),
        'track_length_m': raceline.length,
        'spawn': {
            'ego_s_m': rules['spawn_frac'] * raceline.length,
            'gap_m': rules['start_gap_m'],
        },
        'rules': rules,
        'ego_policy': ego_policy,
        'opp_policy': opp_policy,
        'record_topics': RECORD_TOPICS + list(spec['record']['extra_topics']),
    }
