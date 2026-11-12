"""
Third-party opponent policies: the planners of f1tenth_benchmarks (B. Evans et al., "Unifying F1TENTH Autonomous
Racing: Survey, Methods and Benchmarks", 2024, Apache-2.0), run unchanged from the pinned submodule
utilities/third_party/f1tenth_benchmarks (F1TENTH_BENCHMARKS_DIR overrides the location).

params:
  planner:     pure_pursuit (upstream GlobalPurePursuit) | follow_the_gap (upstream FollowTheGap, LiDAR only)
  line:        pure_pursuit only: the datagen line (raceline | centerline | shortest_path) whose path and speed
               profile stand in for upstream's raceline CSV
  speed_scale: pure_pursuit: multiplies the line's speed profile, so the lookahead still follows the true speed
               and upstream's start mode (4 m/s below 1 m/s) stays as it is. follow_the_gap: multiplies the command
  overrides:   values replacing those of upstream's params/<Planner>.yaml, e.g. {corners_speed: 2.0}

Upstream reads params/ and writes Logs/ relative to the working directory. The planners are built in a temporary
folder, so neither the submodule nor the runner's working directory changes.
"""
import contextlib
import importlib
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml

from datagen.policy import Obs, Policy

UPSTREAM = Path(os.environ.get('F1TENTH_BENCHMARKS_DIR')
                or Path(__file__).resolve().parents[3] / 'third_party' / 'f1tenth_benchmarks')

PLANNERS = {  # name -> (upstream module, class; its parameters are params/<class>.yaml)
    'pure_pursuit': ('f1tenth_benchmarks.classic_racing.GlobalPurePursuit', 'GlobalPurePursuit'),
    'follow_the_gap': ('f1tenth_benchmarks.mapless_racing.FollowTheGap', 'FollowTheGap'),
}


def _import(module: str):
    if str(UPSTREAM) not in sys.path:
        sys.path.append(str(UPSTREAM))  # appended: installed packages (e.g. trajectory_planning_helpers) win
    return importlib.import_module(module)


@contextlib.contextmanager
def _upstream_workdir(params_file: str, overrides: dict):
    """A temporary working directory with upstream's params/, the overrides applied to params/<params_file>.yaml."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='f1tenth_benchmarks_') as tmp:
        shutil.copytree(UPSTREAM / 'params', os.path.join(tmp, 'params'))
        path = os.path.join(tmp, 'params', params_file + '.yaml')
        with open(path) as f:
            values = yaml.safe_load(f)
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise KeyError(f"overrides {unknown} are not in upstream params/{params_file}.yaml: {sorted(values)}")
        with open(path, 'w') as f:
            yaml.safe_dump({**values, **overrides}, f)
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(cwd)


def _racetrack(track, speed_scale: float):
    """Upstream's RaceTrack, built from a datagen Track (speed profile scaled) instead of a raceline CSV."""
    rt = _import('f1tenth_benchmarks.utils.track_utils').RaceTrack(None, load=False)
    rt.path = np.column_stack([track.x, track.y])
    rt.speeds = speed_scale * np.asarray(track.vx, dtype=float)
    with warnings.catch_warnings():  # the periodic spline closes the loop itself and warns about it
        warnings.simplefilter('ignore', RuntimeWarning)
        rt.init_path()
    return rt


class BenchmarkPlanner(Policy):
    def __init__(self, params, tracks, rng):
        super().__init__(params, tracks, rng)
        name = params['planner']
        if name not in PLANNERS:
            raise ValueError(f"planner {name!r} unknown, one of {sorted(PLANNERS)}")
        module, cls = PLANNERS[name]
        self.needs_scan = name == 'follow_the_gap'
        self.speed_scale = float(params.get('speed_scale', 1.0))
        self.command_scale = 1.0 if name == 'pure_pursuit' else self.speed_scale  # pure pursuit: in the profile
        with _upstream_workdir(cls, params.get('overrides') or {}):
            planner_cls = getattr(_import(module), cls)
            if name == 'pure_pursuit':
                self.planner = planner_cls('datagen', init_folder=False)
                self.planner.racetrack = _racetrack(tracks[params['line']], self.speed_scale)
            else:
                self.planner = planner_cls('datagen')

    @staticmethod
    def available() -> bool:
        """True if the submodule and its dependencies import."""
        try:
            for module, _ in PLANNERS.values():
                _import(module)
            return True
        except ImportError:
            return False

    def act(self, obs: Obs):
        if self.needs_scan and obs.scan is None:
            raise ValueError(f"{self.params['planner']} needs obs.scan")
        me = obs.opp
        steer, speed = self.planner.plan({'scan': obs.scan, 'pose': np.array([me.x, me.y, me.yaw]),
                                          'vehicle_speed': me.vx})  # upstream: [steering angle, speed]
        return float(steer), self.command_scale * float(speed)
