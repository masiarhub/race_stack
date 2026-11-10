"""What the evaluation needs from one recorded episode, read from its bag without ROS."""
import dataclasses
import json
import pathlib

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from datagen.extract import physics_steps_ns, time_grid

OPP_STATE, OPP_DRIVE, EGO_PLAN = '/opp_racecar/rollout_state', '/opp_drive', '/local_waypoints'
STEP_TOPICS = ('/sim/collision/walls', '/sim/collisions')
ACKERMANN_DRIVE = 'float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk'
ACKERMANN_STAMPED = 'std_msgs/Header header\nackermann_msgs/AckermannDrive drive'


@dataclasses.dataclass
class Recording:
    name: str
    track: str
    grid_ns: np.ndarray     # recording times of the samples of the dataset grid
    opp_t: np.ndarray
    opp_states: np.ndarray  # [n, 7] rollout state of the opponent, one per physics step
    cmd_t: np.ndarray
    commands: np.ndarray    # [n, 2] steering and speed commands of the opponent
    plan_t: np.ndarray
    plan_x: np.ndarray      # [n, waypoints] ego plan, map frame
    plan_y: np.ndarray
    plan_speed: np.ndarray


def _typestore(f110_msgs_dir):
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name in ('Wpnt', 'WpntArray'):
        types.update(get_types_from_msg((pathlib.Path(f110_msgs_dir) / f'{name}.msg').read_text(), f'f110_msgs/msg/{name}'))
    types.update(get_types_from_msg(ACKERMANN_DRIVE, 'ackermann_msgs/msg/AckermannDrive'))
    types.update(get_types_from_msg(ACKERMANN_STAMPED, 'ackermann_msgs/msg/AckermannDriveStamped'))
    store.register(types)
    return store


def _read(bag_dir, store, topics) -> dict:
    out = {topic: [] for topic in topics}
    with Reader(bag_dir) as reader:
        for connection, t_ns, raw in reader.messages([c for c in reader.connections if c.topic in out]):
            out[connection.topic].append((t_ns, store.deserialize_cdr(raw, connection.msgtype)))
    return out


def load_recording(episode_dir, dataset_cfg: dict, f110_msgs_dir) -> Recording:
    episode_dir = pathlib.Path(episode_dir)
    plan = json.loads((episode_dir / 'episode.json').read_text())
    result = json.loads((episode_dir / 'result.json').read_text())
    bag = _read(episode_dir / 'bag', _typestore(f110_msgs_dir), [OPP_STATE, OPP_DRIVE, EGO_PLAN, *STEP_TOPICS])
    cut_end_s = dataset_cfg['drop_before_collision_s'] if result['reason'] == 'collision' else 0.0
    grid = time_grid(result['t0_ns'], result['t_end_ns'], physics_steps_ns(bag), dataset_cfg['dt'],
                     plan['rules']['warmup_s'], cut_end_s)
    waypoints = bag[EGO_PLAN]
    return Recording(
        name=plan['name'], track=plan['track'], grid_ns=grid.wall_ns,
        opp_t=np.array([t for t, _ in bag[OPP_STATE]], dtype=np.int64),
        opp_states=np.array([m.data for _, m in bag[OPP_STATE]]),
        cmd_t=np.array([t for t, _ in bag[OPP_DRIVE]], dtype=np.int64),
        commands=np.array([[m.drive.steering_angle, m.drive.speed] for _, m in bag[OPP_DRIVE]]),
        plan_t=np.array([t for t, _ in waypoints], dtype=np.int64),
        plan_x=np.array([[w.x_m for w in m.wpnts] for _, m in waypoints]),
        plan_y=np.array([[w.y_m for w in m.wpnts] for _, m in waypoints]),
        plan_speed=np.array([[w.vx_mps for w in m.wpnts] for _, m in waypoints]))
