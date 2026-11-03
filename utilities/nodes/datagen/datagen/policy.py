"""
Opponent (and ego) policies.

A policy is a YAML definition file. It is resolved once per episode (distributions sampled), written
into the episode folder and loaded before the start gate opens. After that it can not change:
python policies get read-only parameters, and the episode manager ends the episode as invalid if
any node of the opponent changes a parameter after t=0.

Two kinds:
  kind: launch   any ROS launch file (e.g. the race stack's time trials stack with sim_opp:=true).
                 `args` are launch arguments, `params` per fully qualified node name, e.g.
                 /opp_racecar/sector_tuner: {global_limit: 0.3}. Note that parameters a node
                 loads from its own <param from=...> file can not be overridden this way.
  kind: python   a class implementing datagen.policy.Policy, given as `entrypoint: file.py:Class`
                 (relative to the YAML) or `module.path:Class`. It is run by datagen's policy_runner.

Contract for an opponent policy: its nodes live in /opp_racecar and it publishes
AckermannDriveStamped on /opp_drive. An ego policy publishes on /drive.

`ground_truth_obstacles: true` (ego policies only): datagen's gt_obstacles node publishes the true opponent on
/perception/obstacles. The policy must then not run its own detection and tracking (e.g. perception: false).
"""
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from typing import Dict, Tuple

import numpy as np
import yaml

from datagen.sampling import sample

KINDS = ('launch', 'python')


def policy_search_path(spec_dir: str) -> list:
    path = [spec_dir, os.path.join(spec_dir, 'policies')]
    path += [p for p in os.environ.get('DATAGEN_POLICY_PATH', '').split(':') if p]
    try:
        from ament_index_python.packages import get_package_share_directory
        path.append(os.path.join(get_package_share_directory('datagen'), 'policies'))
    except Exception:
        path.append(os.path.join(os.path.dirname(__file__), '..', 'policies'))
    return path


def find_policy_file(ref: str, spec_dir: str) -> str:
    """ref is a path to a YAML file, or a policy name looked up as <name>.yaml in the search path."""
    if ref.endswith(('.yaml', '.yml')) or os.sep in ref:
        path = os.path.expanduser(ref if os.path.isabs(ref) else os.path.join(spec_dir, ref))
        if os.path.isfile(path):
            return os.path.realpath(path)
        raise FileNotFoundError(f"Policy file {path} not found")
    search = policy_search_path(spec_dir)
    for d in search:
        path = os.path.join(d, ref + '.yaml')
        if os.path.isfile(path):
            return os.path.realpath(path)
    raise FileNotFoundError(f"Policy '{ref}' not found in {search} (set DATAGEN_POLICY_PATH to add folders)")


def _sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_policy_def(path: str) -> dict:
    with open(path) as f:
        pdef = yaml.safe_load(f)
    pdef.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    pdef.setdefault('params', {})
    pdef.setdefault('ready_topics', [])
    pdef.setdefault('ground_truth_obstacles', False)
    if pdef.get('kind') not in KINDS:
        raise ValueError(f"{path}: kind must be one of {KINDS}")
    if pdef['kind'] == 'launch':
        if 'launch' not in pdef or not {'package', 'file'} <= set(pdef['launch']):
            raise ValueError(f"{path}: launch policies need launch: {{package: ..., file: ...}}")
        pdef.setdefault('args', {})
    else:
        if ':' not in pdef.get('entrypoint', ''):
            raise ValueError(f"{path}: python policies need entrypoint: file.py:Class or module:Class")
        pdef.setdefault('rate_hz', 50.0)
    pdef['source_file'] = path
    return pdef


def resolve_policy(pdef: dict, rng: np.random.Generator) -> dict:
    """Samples all distributions of a policy definition and freezes it into a plain dict."""
    resolved = {
        'name': pdef['name'],
        'kind': pdef['kind'],
        'source_file': pdef['source_file'],
        'ready_topics': list(pdef['ready_topics']),
        'params': sample(pdef['params'], rng),
    }
    if pdef['ground_truth_obstacles']:  # only when set, so the hashes of all other policies stay as they were
        resolved['ground_truth_obstacles'] = True
    sources = [pdef['source_file']]
    if pdef['kind'] == 'launch':
        resolved['launch'] = dict(pdef['launch'])
        resolved['args'] = {k: str(v).lower() if isinstance(v, bool) else str(v)
                            for k, v in sample(pdef['args'], rng).items()}
    else:
        target, cls = pdef['entrypoint'].rsplit(':', 1)
        if target.endswith('.py'):
            target = target if os.path.isabs(target) else os.path.join(os.path.dirname(pdef['source_file']), target)
            target = os.path.realpath(target)
            sources.append(target)
        resolved['entrypoint'] = f"{target}:{cls}"
        resolved['rate_hz'] = float(pdef['rate_hz'])
    # identifies the exact behaviour: definition + code files + sampled values
    resolved['source_sha256'] = {os.path.basename(p): _sha256(p) for p in sources}
    resolved['policy_hash'] = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:12]
    return resolved


def load_policy_class(entrypoint: str):
    target, cls = entrypoint.rsplit(':', 1)
    if target.endswith('.py'):
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(target))[0], target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, cls)


@dataclasses.dataclass
class AgentState:
    x: float         # [m] map frame
    y: float         # [m]
    yaw: float       # [rad]
    vx: float        # [m/s] body frame
    vy: float        # [m/s] body frame
    yaw_rate: float  # [rad/s]
    s: float         # [m] centerline Frenet
    d: float         # [m] positive to the left
    vs: float        # [m/s]
    vd: float        # [m/s]


@dataclasses.dataclass
class Obs:
    t: float          # [s] since the policy received its first observation
    opp: AgentState   # the car this policy drives
    ego: AgentState   # the other car (simulator ground truth)


class Policy:
    """
    Base class of python policies. One instance drives the opponent for a whole episode, with params fixed.

    tracks: {'centerline', 'raceline', 'shortest_path'} -> datagen.track.Track of the current map
    rng:    seeded per episode, use it for any randomness so episodes stay reproducible
    """

    def __init__(self, params: dict, tracks: Dict[str, 'Track'], rng: np.random.Generator):
        self.params = params
        self.tracks = tracks
        self.rng = rng

    def act(self, obs: Obs) -> Tuple[float, float]:
        """Returns (steering angle [rad], speed [m/s])."""
        raise NotImplementedError


def state_from_odom(msg, centerline) -> AgentState:
    """AgentState of an Odometry message, as a python policy observes it."""
    from datagen.track import body_to_map
    p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)))
    s, d = centerline.to_frenet(p.x, p.y)
    vs, vd = centerline.frenet_velocity(s, *body_to_map(yaw, tw.linear.x, tw.linear.y))
    return AgentState(p.x, p.y, yaw, tw.linear.x, tw.linear.y, tw.angular.z,
                      float(s), float(d), float(vs), float(vd))
