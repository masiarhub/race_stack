"""
Raw episode bags -> dataset table. Runs offline, so the representation can change without re-simulating.

Rules: x is built from what the ego can know (tracker estimate of the opponent, by default), y from the
simulator ground truth. Rows in the warm-up, past the end of the episode or with any missing value are dropped.
"""
import copy
import dataclasses
import json
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from datagen.features import FEATURES, LABELS, Context, parse_entry
from datagen.track import Track, body_to_map, unwrap_s

DATASET_DEFAULTS = {
    'dt': 0.05,                  # [s] simulated time between two samples
    'frame': 'frenet',           # frenet | cartesian
    'reference': 'centerline',   # Frenet reference: centerline | raceline | shortest_path
    'opp_source': 'tracker',     # x uses the ego's tracker estimate (tracker) or the sim ground truth (gt)
    'features': ['rel_pos', 'opp_d', 'opp_vel', 'ego_vel', {'curvature': {'n': 10, 'step_m': 0.5}}],
    'labels': {'type': 'state_delta', 'horizon': 1},
    'history': 1,                # stack x_{k-H+1} .. x_k
    'tracker': {'assoc_gate_m': 1.0, 'max_age_s': 0.2},
    'drop_before_collision_s': 0.5,
    # frame frenet: drop samples where kappa * d gets close to 1 (car near/past the centre of curvature on the
    # inside of a corner), there s is not unique and jumps along the corner
    'frenet_singular_kappa_d': 0.8,
}

# the race stack tracker works in the Frenet frame of /global_waypoints, i.e. the min curvature raceline
TRACKER_LINE = 'raceline'
# after the start gate the bridge advances the physics once per tick and publishes these once per tick;
# /sim/collisions is the combined collision topic of runs recorded before the car / wall split
PHYSICS_STEP_TOPICS = ('/sim/collision/walls', '/sim/collisions')
PHYSICS_DT_S = 0.01  # [s] physics time step of the gym


def dataset_config(spec_dataset: dict) -> dict:
    cfg = copy.deepcopy(DATASET_DEFAULTS)
    for k, v in (spec_dataset or {}).items():
        cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    unknown = [parse_entry(e)[0] for e in cfg['features'] if parse_entry(e)[0] not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features {unknown}, available: {sorted(FEATURES)}")
    if cfg['labels']['type'] not in LABELS:
        raise ValueError(f"Unknown label type {cfg['labels']['type']}, available: {sorted(LABELS)}")
    if cfg['frame'] not in ('frenet', 'cartesian') or cfg['opp_source'] not in ('tracker', 'gt'):
        raise ValueError("frame must be frenet|cartesian and opp_source tracker|gt")
    return cfg


# ------------------------------------------------------------------------------------------- bag reading
def read_bag(bag_dir: str, topics: list) -> dict:
    """topic -> list of (receive time [ns], msg). Receive times share one clock across all topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    with open(os.path.join(bag_dir, 'metadata.yaml')) as f:
        storage_id = yaml.safe_load(f)['rosbag2_bagfile_information']['storage_identifier']
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        out[topic].append((t_ns, deserialize_message(data, get_message(types[topic]))))
    return out


def odom_arrays(msgs) -> dict:
    t = np.array([m[0] for m in msgs], dtype=np.int64)
    p = [m[1].pose.pose for m in msgs]
    q = np.array([[x.orientation.x, x.orientation.y, x.orientation.z, x.orientation.w] for x in p])
    yaw = np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1]**2 + q[:, 2]**2))
    return {'t': t,
            'x': np.array([x.position.x for x in p]),
            'y': np.array([x.position.y for x in p]),
            'yaw': np.unwrap(yaw),
            'vx_b': np.array([m[1].twist.twist.linear.x for m in msgs]),
            'vy_b': np.array([m[1].twist.twist.linear.y for m in msgs]),
            'wz': np.array([m[1].twist.twist.angular.z for m in msgs])}


def drive_zoh(msgs, grid) -> dict:
    """Last command at or before each grid time (zero order hold), NaN before the first one."""
    t = np.array([m[0] for m in msgs], dtype=np.int64)
    idx = np.searchsorted(t, grid, side='right') - 1
    out = {}
    for key, get in (('steer', lambda d: d.steering_angle), ('speed', lambda d: d.speed)):
        vals = np.array([get(m[1].drive) for m in msgs]) if msgs else np.zeros(0)
        out[key] = np.where(idx >= 0, vals[np.clip(idx, 0, None)] if len(vals) else np.nan, np.nan)
    return out


def agent_state(ref: Track, x, y, vx_map, vy_map) -> dict:
    s, d = ref.to_frenet(x, y)
    vs, vd = ref.frenet_velocity(s, vx_map, vy_map)
    return {'x': x, 'y': y, 'vx': vx_map, 'vy': vy_map, 's': s, 'd': d, 'vs': vs, 'vd': vd,
            's_unwrapped': unwrap_s(s, ref.length)}


def gt_on_grid(odom: dict, grid, ref: Track) -> dict:
    x = np.interp(grid, odom['t'], odom['x'])
    y = np.interp(grid, odom['t'], odom['y'])
    yaw = np.interp(grid, odom['t'], odom['yaw'])
    vx, vy = body_to_map(yaw, np.interp(grid, odom['t'], odom['vx_b']), np.interp(grid, odom['t'], odom['vy_b']))
    return agent_state(ref, x, y, vx, vy)


def tracker_on_grid(obstacle_msgs, opp_odom: dict, grid, ref: Track, tracker_line: Track, cfg: dict) -> dict:
    """
    Picks the tracked obstacle that belongs to the opponent (closest to the ground truth, within the gate),
    converts it from the tracker's raceline Frenet frame to the requested frame and holds it on the grid.
    """
    t, xs, ys, vxs, vys = [], [], [], [], []
    gate = cfg['tracker']['assoc_gate_m']
    for t_ns, msg in obstacle_msgs:
        if not msg.obstacles:
            continue
        gx, gy = np.interp(t_ns, opp_odom['t'], opp_odom['x']), np.interp(t_ns, opp_odom['t'], opp_odom['y'])
        s_gt, d_gt = tracker_line.to_frenet(gx, gy)
        obs = np.array([[o.s_center, o.d_center, o.vs, o.vd] for o in msg.obstacles])
        err = np.hypot(tracker_line.wrap_ds(obs[:, 0] - s_gt), obs[:, 1] - d_gt)
        best = int(np.argmin(err))
        if err[best] > gate:
            continue
        s, d, vs, vd = obs[best]
        x, y = tracker_line.to_cartesian(s, d)
        vx, vy = tracker_line.cartesian_velocity(s, vs, vd)
        t.append(t_ns), xs.append(x), ys.append(y), vxs.append(vx), vys.append(vy)

    n = len(grid)
    if not t:
        nan = np.full(n, np.nan)
        return {k: nan for k in ('x', 'y', 'vx', 'vy', 's', 'd', 'vs', 'vd', 's_unwrapped')}
    t = np.array(t, dtype=np.int64)
    idx = np.searchsorted(t, grid, side='right') - 1
    fresh = (idx >= 0) & ((grid - t[np.clip(idx, 0, None)]) * 1e-9 <= cfg['tracker']['max_age_s'])
    pick = lambda a: np.where(fresh, np.asarray(a, dtype=float)[np.clip(idx, 0, None)], np.nan)
    x, y, vx, vy = pick(xs), pick(ys), pick(vxs), pick(vys)
    ok = ~np.isnan(x)
    state = {k: np.full(n, np.nan) for k in ('s', 'd', 'vs', 'vd', 's_unwrapped')}
    if ok.any():
        sub = agent_state(ref, x[ok], y[ok], vx[ok], vy[ok])
        for k in state:
            state[k][ok] = sub[k]
    return {'x': x, 'y': y, 'vx': vx, 'vy': vy, **state}


# ------------------------------------------------------------------------------------------- tables
def frenet_singular(ref: Track, agent: dict, limit: float) -> np.ndarray:
    """kappa * d > limit within +-0.5 m of the projection (d on the inside of the corner has the sign of kappa)."""
    worst = np.max([ref.interp(ref.kappa, agent['s'] + off) * agent['d'] for off in np.linspace(-0.5, 0.5, 5)], axis=0)
    return worst > limit


def columns_of(entries, ctx, registry, prefix):
    cols = []
    for entry in entries:
        name, opts = parse_entry(entry)
        for col, values, unit, desc in registry[name](ctx, **opts):
            cols.append((f'{prefix}_{col}', np.asarray(values, dtype=float), unit, desc))
    return cols


class Grid(NamedTuple):
    """Sample times of one episode, equally spaced in simulated time."""
    wall_ns: np.ndarray  # recording-clock times at which messages are looked up
    t_s: np.ndarray      # simulated time since the start gate [s]


@dataclasses.dataclass
class Episode:
    """One recorded episode, as everything downstream of the recording sees it."""
    name: str
    plan: dict    # episode.json
    result: dict  # result.json
    bag: dict     # topic -> [(recording time ns, msg)]
    grid: Grid


def time_grid(t0_ns: int, t_end_ns: int, step_ns, dt: float, warmup_s: float, cut_end_s: float) -> Grid:
    """
    One sample every dt of simulated time, from warmup_s after the start gate to cut_end_s before the end.
    The gym steps on a wall-clock timer and falls behind under load, so simulated time is counted in physics
    steps (step_ns: their recording times, PHYSICS_DT_S each). Without any step the wall clock is used instead.
    """
    steps = np.asarray(step_ns, dtype=np.int64)
    steps = steps[(steps > t0_ns) & (steps <= t_end_ns)]
    if steps.size:
        wall = np.concatenate([[t0_ns], steps]).astype(float)
        sim = np.arange(steps.size + 1) * PHYSICS_DT_S
    else:
        wall = np.array([t0_ns, t_end_ns], dtype=float)
        sim = (wall - t0_ns) * 1e-9
    n = max(0, int(np.ceil((sim[-1] - cut_end_s - warmup_s) / dt - 1e-9)))
    t_s = warmup_s + np.arange(n) * dt
    return Grid(np.round(np.interp(t_s, sim, wall)).astype(np.int64), t_s)


def physics_steps_ns(bag: dict) -> np.ndarray:
    """Recording times of the physics steps (ticks) of an episode."""
    for topic in PHYSICS_STEP_TOPICS:
        if bag.get(topic):
            return np.array([m[0] for m in bag[topic]], dtype=np.int64)
    return np.zeros(0, dtype=np.int64)


def load_episode(ep_dir: str, cfg: dict, topics: list) -> Episode:
    """Plan, result, the requested bag topics and the sample grid of one recorded episode."""
    with open(os.path.join(ep_dir, 'episode.json')) as f:
        plan = json.load(f)
    with open(os.path.join(ep_dir, 'result.json')) as f:
        result = json.load(f)
    bag = read_bag(os.path.join(ep_dir, 'bag'), list(dict.fromkeys([*topics, *PHYSICS_STEP_TOPICS])))
    cut_end_s = cfg['drop_before_collision_s'] if result['reason'] == 'collision' else 0.0
    grid = time_grid(result['t0_ns'], result['t_end_ns'], physics_steps_ns(bag),
                     cfg['dt'], plan['rules']['warmup_s'], cut_end_s)
    return Episode(plan['name'], plan, result, bag, grid)


def episode_table(ep_dir: str, cfg: dict):
    episode = load_episode(ep_dir, cfg, ['/state_estimation/odom', '/opp_racecar/odom', '/perception/obstacles',
                                         '/drive', '/opp_drive'])
    ep, bag, grid = episode.plan, episode.bag, episode.grid.wall_ns
    wpnts = os.path.join(ep['map_dir'], 'global_waypoints.json')
    ref = Track.from_json(wpnts, cfg['reference'])
    tracker_line = Track.from_json(wpnts, TRACKER_LINE)
    ego_odom, opp_odom = odom_arrays(bag['/state_estimation/odom']), odom_arrays(bag['/opp_racecar/odom'])
    if grid.size == 0:
        return None, [], {'rows': 0, 'note': 'episode shorter than the warm-up'}

    ego = gt_on_grid(ego_odom, grid, ref)
    opp_gt = gt_on_grid(opp_odom, grid, ref)
    opp = opp_gt if cfg['opp_source'] == 'gt' else \
        tracker_on_grid(bag['/perception/obstacles'], opp_odom, grid, ref, tracker_line, cfg)
    ctx = Context(cfg['frame'], ref, opp, opp_gt, ego,
                  drive_zoh(bag['/drive'], grid), drive_zoh(bag['/opp_drive'], grid))

    x_cols = columns_of(cfg['features'], ctx, FEATURES, 'x')
    if cfg['history'] > 1:
        hist = []
        for j in range(cfg['history'] - 1, 0, -1):
            for name, v, unit, desc in x_cols:
                shifted = np.full_like(v, np.nan)
                shifted[j:] = v[:-j]
                hist.append((f'{name}_m{j}', shifted, unit, f'{desc}, {j} step(s) earlier'))
        x_cols = hist + x_cols
    y_cols = columns_of([{cfg['labels']['type']: {'horizon': cfg['labels']['horizon']}}], ctx, LABELS, 'y')

    opp_pol = ep['opp_policy']
    meta = [('episode', ep['name'], '', 'episode id'),
            ('track', ep['track'], '', 'map name'),
            ('opp_policy', opp_pol['name'], '', 'opponent policy definition'),
            ('opp_policy_hash', opp_pol['policy_hash'], '', 'hash of the resolved (sampled) policy'),
            ('t', episode.grid.t_s, 's', 'simulated time since the start gate opened')]
    df = pd.DataFrame({name: values for name, values, _, _ in meta + x_cols + y_cols})
    data_cols = [c[0] for c in x_cols + y_cols]
    n_all = len(df)
    nan_x = df[[c[0] for c in x_cols]].isna().any(axis=1)
    singular = np.zeros(n_all, dtype=bool)
    if cfg['frame'] == 'frenet' and cfg['frenet_singular_kappa_d']:
        limit = cfg['frenet_singular_kappa_d']
        gt_sing = frenet_singular(ref, opp_gt, limit)
        span = cfg['labels']['horizon'] + (0 if cfg['labels']['type'] == 'action' else 1)
        for i in range(span):  # labels use the ground truth up to k + horizon
            singular[:n_all - i] |= gt_sing[i:]
        singular |= frenet_singular(ref, ego, limit)
        with np.errstate(invalid='ignore'):
            singular |= frenet_singular(ref, opp, limit)
    df = df[~df[data_cols].isna().any(axis=1) & ~singular].reset_index(drop=True)
    columns = [{'name': n, 'role': role, 'unit': u, 'description': d}
               for role, cols in (('meta', meta), ('x', x_cols), ('y', y_cols)) for n, _, u, d in cols]
    report = {'rows': len(df), 'grid_rows': n_all, 'rows_missing_x': int(nan_x.sum()),
              'rows_frenet_singular': int(singular.sum())}
    return df, columns, report


def extract_run(run_dir: str, dataset_override: dict = None, out_name: str = 'dataset.csv') -> str:
    with open(os.path.join(run_dir, 'spec.yaml')) as f:
        spec = yaml.safe_load(f)
    cfg = dataset_config(dataset_override if dataset_override is not None else spec.get('dataset'))
    with open(os.path.join(run_dir, 'manifest.json')) as f:
        manifest = json.load(f)

    tables, columns, report, episodes = [], None, {}, []
    for entry in manifest['episodes']:
        ep_dir = os.path.join(run_dir, 'episodes', entry['name'])
        if not entry.get('valid'):
            report[entry['name']] = {'rows': 0, 'note': f"invalid episode ({entry.get('reason')})"}
            continue
        df, cols, rep = episode_table(ep_dir, cfg)
        report[entry['name']] = rep
        if df is not None and len(df):
            tables.append(df)
            columns = cols
        with open(os.path.join(ep_dir, 'episode.json')) as f:
            ep = json.load(f)
        episodes.append({'episode': entry['name'], 'track': ep['track'], 'reason': entry.get('reason'),
                         'duration_s': entry.get('duration_s'), 'rows': rep['rows'],
                         'opp_policy': ep['opp_policy']['name'], 'opp_policy_hash': ep['opp_policy']['policy_hash'],
                         'start_gap_m': ep['spawn']['gap_m'], 'ego_spawn_s_m': ep['spawn']['ego_s_m'],
                         **{f'opp.{k}': json.dumps(v) if isinstance(v, (dict, list)) else v
                            for k, v in {**ep['opp_policy'].get('args', {}), **_flatten(ep['opp_policy']['params'])}.items()}})

    out = os.path.join(run_dir, out_name)
    base = os.path.splitext(out)[0]
    data = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    data.to_csv(out, index=False)
    pd.DataFrame(episodes).to_csv(base + '_episodes.csv', index=False)
    with open(base + '_columns.json', 'w') as f:
        json.dump({'config': cfg, 'columns': columns or []}, f, indent=2)
    with open(base + '_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    return out


def _flatten(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f'{prefix}{k}.'))
        else:
            out[prefix + k] = v
    return out
