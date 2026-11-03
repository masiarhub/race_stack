"""
Sanity checks of a recorded run, written next to the dataset as check_<name>.json.

perception: separates the pipeline from the ego's perception. Per grid step it asks whether the detector
(/perception/detection/raw_obstacles) and the tracker (/perception/obstacles) see the true opponent, and how far
the tracker estimate is from the truth. It then builds the dataset twice, x from the tracker and x from the
ground truth, and compares the x columns of the rows both versions keep.
"""
import json
import os

import numpy as np
import pandas as pd
import yaml

from datagen.extract import (PHYSICS_DT_S, PHYSICS_STEP_TOPIC, TRACKER_LINE, dataset_config, episode_table,
                             gt_on_grid, load_episode, odom_arrays)
from datagen.policy import Obs, _sha256, load_policy_class, state_from_odom
from datagen.track import LINE_KEYS, Track

DETECTION_TOPIC = '/perception/detection/raw_obstacles'
TRACKER_TOPIC = '/perception/obstacles'
NONE, FAR, MATCH = 0, 1, 2  # per grid step: no fresh obstacle, obstacles but none at the opponent, opponent seen


def match_on_grid(msgs, opp_odom: dict, grid, line: Track, gate_m: float, max_age_s: float) -> dict:
    """Latest message per grid step (if not older than max_age_s) and its obstacle closest to the true opponent."""
    n = len(grid)
    out = {'status': np.full(n, NONE), **{k: np.full(n, np.nan) for k in ('s', 'd', 'vs', 'vd', 'err', 'static')}}
    if not msgs:
        return out
    t = np.array([m[0] for m in msgs], dtype=np.int64)
    idx = np.searchsorted(t, grid, side='right') - 1
    for k, i in enumerate(idx):
        if i < 0 or (grid[k] - t[i]) * 1e-9 > max_age_s or not msgs[i][1].obstacles:
            continue
        gx, gy = np.interp(grid[k], opp_odom['t'], opp_odom['x']), np.interp(grid[k], opp_odom['t'], opp_odom['y'])
        s_gt, d_gt = line.to_frenet(np.array([gx]), np.array([gy]))
        obs = msgs[i][1].obstacles
        err = [np.hypot(line.wrap_ds(o.s_center - s_gt[0]), o.d_center - d_gt[0]) for o in obs]
        best = int(np.argmin(err))
        out['err'][k] = err[best]
        if err[best] > gate_m:
            out['status'][k] = FAR
            continue
        o = obs[best]
        out['status'][k] = MATCH
        out['s'][k], out['d'][k], out['vs'][k], out['vd'][k] = o.s_center, o.d_center, o.vs, o.vd
        out['static'][k] = float(o.is_static)
    return out


def _share(mask, of=None):
    of = np.ones_like(mask, dtype=bool) if of is None else of
    return round(float(mask[of].mean()), 3) if of.any() else None


def _stats(err):
    err = err[~np.isnan(err)]
    if not len(err):
        return None
    return {'bias': round(float(err.mean()), 3), 'rmse': round(float(np.sqrt((err**2).mean())), 3),
            'p95_abs': round(float(np.percentile(np.abs(err), 95)), 3)}


def perception_episode(ep_dir: str, cfg: dict) -> dict:
    episode = load_episode(ep_dir, cfg, ['/opp_racecar/odom', DETECTION_TOPIC, TRACKER_TOPIC])
    ep, bag, grid = episode.plan, episode.bag, episode.grid.wall_ns
    if grid.size == 0:
        return {'opp_policy': ep['opp_policy']['name'], 'grid_steps': 0}
    line = Track.from_json(os.path.join(ep['map_dir'], 'global_waypoints.json'), TRACKER_LINE)
    opp_odom = odom_arrays(bag['/opp_racecar/odom'])
    gt = gt_on_grid(opp_odom, grid, line)  # truth in the tracker's own frame (raceline Frenet)
    gate, age = cfg['tracker']['assoc_gate_m'], cfg['tracker']['max_age_s']
    det = match_on_grid(bag[DETECTION_TOPIC], opp_odom, grid, line, gate, age)
    trk = match_on_grid(bag[TRACKER_TOPIC], opp_odom, grid, line, gate, age)

    seen = trk['status'] == MATCH
    zero_v = seen & (trk['vs'] == 0) & (trk['vd'] == 0)
    return {
        'opp_policy': ep['opp_policy']['name'],
        'grid_steps': int(grid.size),
        'detection_recorded': bool(bag[DETECTION_TOPIC]),
        'detected': _share(det['status'] == MATCH),
        'tracked': _share(seen),
        'tracked_if_detected': _share(seen, det['status'] == MATCH),
        'tracker_obstacle_elsewhere': _share(trk['status'] == FAR),
        'tracked_static': _share(trk['static'] == 1, seen),
        'tracked_zero_velocity': _share(zero_v, seen),
        'tracker_error': {
            's_m': _stats(line.wrap_ds(trk['s'] - gt['s'])),
            'd_m': _stats(trk['d'] - gt['d']),
            'vs_mps': _stats(trk['vs'] - gt['vs']),
            'vd_mps': _stats(trk['vd'] - gt['vd']),
        },
    }


def check_perception(run_dir: str, dataset_override: dict = None) -> dict:
    with open(os.path.join(run_dir, 'spec.yaml')) as f:
        spec = yaml.safe_load(f)
    with open(os.path.join(run_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    base = dataset_config(dataset_override if dataset_override is not None else spec.get('dataset'))
    cfg = {src: {**base, 'opp_source': src} for src in ('tracker', 'gt')}

    episodes, tables = {}, {'tracker': [], 'gt': []}
    for entry in manifest['episodes']:
        if not entry.get('valid'):
            continue
        ep_dir = os.path.join(run_dir, 'episodes', entry['name'])
        rep = perception_episode(ep_dir, base)
        for src in tables:
            df, _, r = episode_table(ep_dir, cfg[src])
            rep[f'rows_x_{src}'] = r['rows']
            if df is not None and len(df):
                tables[src].append(df)
        episodes[entry['name']] = rep

    # same rows, x once from the tracker and once from the truth: the difference is the perception error in x
    x_error = {}
    if tables['tracker'] and tables['gt']:
        trk, gt = pd.concat(tables['tracker']), pd.concat(tables['gt'])
        both = trk.merge(gt, on=['episode', 't'], suffixes=('_trk', '_gt'))
        for col in [c for c in trk.columns if c.startswith('x_')]:
            x_error[col] = _stats((both[col + '_trk'] - both[col + '_gt']).to_numpy())
        x_error['_rows_compared'] = len(both)

    by_policy = {}
    for policy in sorted({e['opp_policy'] for e in episodes.values()}):
        eps = [e for e in episodes.values() if e['opp_policy'] == policy and e['grid_steps']]
        steps = sum(e['grid_steps'] for e in eps)
        weighted = lambda key: round(sum((e[key] or 0) * e['grid_steps'] for e in eps) / steps, 3) if steps else None
        by_policy[policy] = {'episodes': len(eps), 'grid_steps': steps,
                             'detected': weighted('detected'), 'tracked': weighted('tracked'),
                             'rows_x_tracker': sum(e['rows_x_tracker'] for e in eps),
                             'rows_x_gt': sum(e['rows_x_gt'] for e in eps)}

    report = {'config': base, 'by_policy': by_policy, 'x_error_tracker_minus_gt': x_error, 'episodes': episodes}
    with open(os.path.join(run_dir, 'check_perception.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def print_perception(report: dict):
    pct = lambda v: '   -' if v is None else f"{100 * v:3.0f}%"
    print(f"{'opponent policy':20s} {'eps':>3s} {'steps':>5s} {'detected':>8s} {'tracked':>7s} "
          f"{'rows x=tracker':>14s} {'rows x=gt':>9s}")
    for policy, p in report['by_policy'].items():
        print(f"{policy:20s} {p['episodes']:3d} {p['grid_steps']:5d} {pct(p['detected']):>8s} {pct(p['tracked']):>7s} "
              f"{p['rows_x_tracker']:14d} {p['rows_x_gt']:9d}")
    err = report['x_error_tracker_minus_gt']
    if err:
        print(f"\nx error, tracker - truth, on {err['_rows_compared']} rows both versions keep:")
        for col, s in err.items():
            if not col.startswith('_') and s:
                print(f"  {col:12s} bias {s['bias']:+7.3f}  rmse {s['rmse']:6.3f}  p95 |err| {s['p95_abs']:6.3f}")


# ------------------------------------------------------------------------------------------- policies
TT_LINES = {'mincurv': 'raceline', 'shortest_path': 'shortest_path', 'centerline': 'centerline'}  # time trials arg
EXACT = 1e-4  # replayed and recorded commands closer than this count as identical


def _stamp_ns(msg) -> int:
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def _latest(msgs, times, t):
    i = np.searchsorted(times, t, side='right') - 1
    return msgs[i][1] if i >= 0 else None


def replay_python_policy(ep: dict, bag: dict, tracks: dict) -> dict:
    """Recomputes every /opp_drive command from the latest odometry the policy could have had at that moment."""
    pol = ep['opp_policy']
    path = pol['entrypoint'].rsplit(':', 1)[0]
    same_code = _sha256(path) == pol['source_sha256'].get(os.path.basename(path))
    policy = load_policy_class(pol['entrypoint'])(pol['params'], tracks, np.random.default_rng(ep['policy_seed']))
    # the policy reads the ego from /opp_racecar/opp_odom, older runs only have /state_estimation/odom
    own, other = bag['/opp_racecar/odom'], bag['/opp_racecar/opp_odom'] or bag['/state_estimation/odom']
    t_own, t_other = np.array([m[0] for m in own]), np.array([m[0] for m in other])
    cl = tracks['centerline']
    t_first, err = None, []
    for _, drive in bag['/opp_drive']:
        t = _stamp_ns(drive)
        o, e = _latest(own, t_own, t), _latest(other, t_other, t)
        if o is None or e is None:
            continue
        t_first = t if t_first is None else t_first
        steer, speed = policy.act(Obs(t=(t - t_first) * 1e-9, opp=state_from_odom(o, cl), ego=state_from_odom(e, cl)))
        err.append((steer - drive.drive.steering_angle, speed - drive.drive.speed))
    err = np.abs(np.array(err)) if err else np.zeros((0, 2))
    return {'same_code': same_code, 'commands': len(err),
            'identical': _share((err < EXACT).all(axis=1)) if len(err) else None,
            'steer_err_p95_rad': round(float(np.percentile(err[:, 0], 95)), 4) if len(err) else None,
            'speed_err_p95_mps': round(float(np.percentile(err[:, 1], 95)), 4) if len(err) else None}


def _median(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return round(float(np.median(a)), 3) if len(a) else None


def policies_episode(ep_dir: str, cfg: dict) -> dict:
    episode = load_episode(ep_dir, cfg, ['/opp_racecar/odom', '/opp_racecar/opp_odom', '/state_estimation/odom',
                                         '/opp_drive'])
    ep, bag, grid = episode.plan, episode.bag, episode.grid.wall_ns
    pol = ep['opp_policy']
    out = {'opp_policy': pol['name'], 'kind': pol['kind']}
    wpnts = os.path.join(ep['map_dir'], 'global_waypoints.json')
    tracks = {line: Track.from_json(wpnts, line) for line in LINE_KEYS}
    if pol['kind'] == 'python':
        out['replay'] = replay_python_policy(ep, bag, tracks)
    if grid.size == 0:
        out['rows'] = 0
        return out

    # what the parameters prescribe, along the true opponent motion on the grid
    opp_odom = odom_arrays(bag['/opp_racecar/odom'])
    speed = np.hypot(np.interp(grid, opp_odom['t'], opp_odom['vx_b']), np.interp(grid, opp_odom['t'], opp_odom['vy_b']))
    if pol['kind'] == 'python':
        line, scale = tracks['raceline'], pol['params']['speed_scale']
    else:
        line, scale = tracks[TT_LINES[pol['args']['raceline']]], pol['params']['/opp_racecar/sector_tuner']['global_limit']
    on_line = gt_on_grid(opp_odom, grid, line)
    v_target = scale * line.interp(line.vx, on_line['s'])
    t_drive = np.array([m[0] for m in bag['/opp_drive']])
    idx = np.searchsorted(t_drive, grid, side='right') - 1
    cmd = np.where(idx >= 0, np.array([m[1].drive.speed for m in bag['/opp_drive']])[np.clip(idx, 0, None)], np.nan)
    raw = pd.DataFrame({'t': episode.grid.t_s, 'speed': speed, 'v_target': v_target,
                        'cmd': cmd, 'd_line': on_line['d']})

    # the dataset rows as the pipeline writes them, x from the ground truth so perception does not interfere
    df, _, _ = episode_table(ep_dir, {**cfg, 'opp_source': 'gt'})
    if df is None or not len(df):
        out['rows'] = 0
        return out
    df = df[df['episode'] == ep['name']].merge(raw, on='t')
    out['rows'] = len(df)
    out['speed_true_over_target'] = _median(df['speed'] / df['v_target'])
    out['speed_cmd_over_target'] = _median(df['cmd'] / df['v_target'])
    if 'x_opp_vs' in df:
        out['x_opp_vs_over_speed'] = _median(df['x_opp_vs'] / df['speed'])
    if pol['kind'] == 'python' and 'x_opp_d' in df and cfg['reference'] == pol['params']['line']:
        out['x_opp_d_minus_offset_m'] = _median(df['x_opp_d'] - pol['params']['d_offset_m'])
    if pol['kind'] != 'python':
        out['abs_d_to_own_line_m'] = _median(np.abs(df['d_line']))
    if 'block_range_m' in pol['params'] and {'x_delta_s', 'x_delta_d'} <= set(df):
        blocking = (df['x_delta_s'] > 0) & (df['x_delta_s'] < pol['params']['block_range_m'])
        out['abs_delta_d_blocking_m'] = _median(np.abs(df.loc[blocking, 'x_delta_d']))
        out['abs_delta_d_free_m'] = _median(np.abs(df.loc[~blocking, 'x_delta_d']))
        out['blocking_rows'] = int(blocking.sum())
    return out


def check_policies(run_dir: str, dataset_override: dict = None) -> dict:
    with open(os.path.join(run_dir, 'spec.yaml')) as f:
        spec = yaml.safe_load(f)
    with open(os.path.join(run_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    cfg = dataset_config(dataset_override if dataset_override is not None else spec.get('dataset'))
    episodes = {e['name']: policies_episode(os.path.join(run_dir, 'episodes', e['name']), cfg)
                for e in manifest['episodes'] if e.get('valid')}
    report = {'config': cfg, 'episodes': episodes}
    with open(os.path.join(run_dir, 'check_policies.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def print_policies(report: dict):
    fmt = lambda v, f='{:6.3f}': '     -' if v is None else f.format(v)
    print("Replay of the python policies (recorded /opp_drive vs recomputed from the recorded state):")
    print(f"  {'ep':8s} {'policy':18s} {'same code':>9s} {'commands':>8s} {'identical':>9s} {'steer p95':>9s} {'speed p95':>9s}")
    for name, e in report['episodes'].items():
        r = e.get('replay')
        if r:
            print(f"  {name:8s} {e['opp_policy']:18s} {str(r['same_code']):>9s} {r['commands']:8d} "
                  f"{fmt(r['identical'], '{:8.0%}'):>9s} {fmt(r['steer_err_p95_rad'], '{:9.4f}')} {fmt(r['speed_err_p95_mps'], '{:9.4f}')}")
    print("\nMotion vs parameters (medians over the dataset rows, x from the ground truth):")
    print(f"  {'ep':8s} {'policy':18s} {'rows':>4s} {'v/target':>8s} {'cmd/target':>10s} {'x_vs/v':>6s} "
          f"{'d-offset':>8s} {'|d| line':>8s} {'|dd| block/free':>15s}")
    for name, e in report['episodes'].items():
        if not e.get('rows'):
            print(f"  {name:8s} {e['opp_policy']:18s}    0")
            continue
        block = (f"{fmt(e['abs_delta_d_blocking_m'], '{:.2f}')}/{fmt(e['abs_delta_d_free_m'], '{:.2f}')}"
                 f" ({e['blocking_rows']})") if 'blocking_rows' in e else '-'
        print(f"  {name:8s} {e['opp_policy']:18s} {e['rows']:4d} {fmt(e['speed_true_over_target'], '{:8.2f}')} "
              f"{fmt(e['speed_cmd_over_target'], '{:10.2f}')} {fmt(e.get('x_opp_vs_over_speed'), '{:6.2f}')} "
              f"{fmt(e.get('x_opp_d_minus_offset_m'), '{:+8.3f}')} {fmt(e.get('abs_d_to_own_line_m'), '{:8.3f}')} {block:>15s}")


# ------------------------------------------------------------------------------------------- labels
SHIFTS = range(-3, 4)          # grid steps
HORIZONS_S = (0.25, 0.5, 1.0)  # open-loop replay horizons
REPLAY_EVERY = 5               # start a replay at every n-th dataset row


def _shifted(a, j):
    """a[k + j] at index k, NaN outside."""
    out = np.full(len(a), np.nan)
    if j >= 0:
        out[:len(a) - j] = a[j:]
    else:
        out[-j:] = a[:j]
    return out


def _sim_params() -> dict:
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('stack_master'), 'config', 'SIM', 'sim_params.yaml')
    with open(path) as f:
        return {k: float(v) for k, v in yaml.safe_load(f).items()}


class GymReplay:
    """Open-loop replay of drive commands with the gym's single-track model (RK4, pid, 2-step steering delay)."""

    def __init__(self, params: dict):
        from f110_gym.envs.dynamic_models import pid, vehicle_dynamics_st
        self.p, self.pid, self.f = params, pid, vehicle_dynamics_st
        self.args = tuple(params[k] for k in ('mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max',
                                               'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max'))

    def run(self, state, steer_buffer, commands):
        """state [x, y, steer, v, yaw, yaw_rate, slip]; commands: (steer, speed) per physics step. Returns xy per step."""
        x, buf, out = np.array(state, dtype=float), list(steer_buffer), []
        p = self.p
        for raw_steer, speed in commands:
            steer = buf[-1]
            buf = [raw_steer] + buf[:-1]
            accl, sv = self.pid(speed, steer, x[3], x[2], p['sv_max'], p['a_max'], p['v_max'], p['v_min'])
            u = np.array([sv, accl])
            k1 = self.f(x, u, *self.args)
            k2 = self.f(x + PHYSICS_DT_S * k1 / 2, u, *self.args)
            k3 = self.f(x + PHYSICS_DT_S * k2 / 2, u, *self.args)
            k4 = self.f(x + PHYSICS_DT_S * k3, u, *self.args)
            x = x + PHYSICS_DT_S / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(x[:2].copy())
        return np.array(out)


def labels_episode(ep_dir: str, cfg: dict, replay: GymReplay) -> dict:
    episode = load_episode(ep_dir, cfg, ['/opp_racecar/odom', '/opp_drive'])
    ep, bag, grid = episode.plan, episode.bag, episode.grid.wall_ns
    pol = ep['opp_policy']
    out = {'opp_policy': pol['name']}
    if grid.size < 2:
        out['rows'] = 0
        return out
    dt = cfg['dt']
    ref = Track.from_json(os.path.join(ep['map_dir'], 'global_waypoints.json'), cfg['reference'])
    opp_odom = odom_arrays(bag['/opp_racecar/odom'])

    # physics steps per second of wall-clock time
    t_step = np.array([m[0] for m in bag[PHYSICS_STEP_TOPIC]], dtype=np.int64)
    in_ep = (t_step >= grid[0]) & (t_step <= grid[-1])
    out['physics_steps_per_s'] = round(float(in_ep.sum() / ((grid[-1] - grid[0]) * 1e-9)), 1)

    # 1. state labels vs velocities: ds/dt = v_t / (1 - kappa d), dd/dt = v_n, trapezoid over [k, k+1]
    gt = gt_on_grid(opp_odom, grid, ref)
    kappa = ref.interp(ref.kappa, gt['s'])
    sdot, ddot = gt['vs'] / (1 - kappa * gt['d']), gt['vd']
    exp_ds = dt * (sdot + _shifted(sdot, 1)) / 2
    exp_dd = dt * (ddot + _shifted(ddot, 1)) / 2
    raw = pd.DataFrame({'t': episode.grid.t_s, 'k': np.arange(len(grid))})
    for j in SHIFTS:
        raw[f'exp_ds_{j}'], raw[f'exp_dd_{j}'] = _shifted(exp_ds, j), _shifted(exp_dd, j)
    df, _, _ = episode_table(ep_dir, {**cfg, 'opp_source': 'gt', 'labels': {'type': 'state_delta', 'horizon': 1}})
    if df is None or not len(df):
        out['rows'] = 0
        return out
    df = df.merge(raw, on='t')
    out['rows'] = len(df)
    out['y_ds_over_expected'] = _median(df['y_ds_1'] / df['exp_ds_0'])
    out['y_ds_error_m'] = _stats((df['y_ds_1'] - df['exp_ds_0']).to_numpy())
    out['y_dd_error_m'] = _stats((df['y_dd_1'] - df['exp_dd_0']).to_numpy())
    rmse = {j: float(np.sqrt(np.nanmean((df['y_ds_1'] - df[f'exp_ds_{j}'])**2 + (df['y_dd_1'] - df[f'exp_dd_{j}'])**2)))
            for j in SHIFTS}
    out['label_shift_rmse_m'] = {j: round(v, 4) for j, v in rmse.items()}
    out['best_shift'] = int(min(rmse, key=rmse.get))

    # 2. action labels replayed through the gym model, stepping exactly when the simulator stepped
    t_cmd = np.array([m[0] for m in bag['/opp_drive']], dtype=np.int64)
    cmds = np.array([[m[1].drive.steering_angle, m[1].drive.speed] for m in bag['/opp_drive']])
    zoh = lambda t: cmds[np.clip(np.searchsorted(t_cmd, t, side='right') - 1, 0, None)]
    pos = lambda t: np.array([np.interp(t, opp_odom['t'], opp_odom['x']), np.interp(t, opp_odom['t'], opp_odom['y'])])
    err = {h: {'replay': [], 'const_vel': []} for h in HORIZONS_S}
    n_steps = int(round(max(HORIZONS_S) / PHYSICS_DT_S))  # horizons are simulated time, i.e. physics steps
    for k in df['k'].to_numpy()[::REPLAY_EVERY]:
        t0 = grid[k]
        steps = t_step[t_step > t0][:n_steps]
        if len(steps) < n_steps or steps[-1] > grid[-1]:
            continue
        v = np.hypot(np.interp(t0, opp_odom['t'], opp_odom['vx_b']), np.interp(t0, opp_odom['t'], opp_odom['vy_b']))
        yaw = np.interp(t0, opp_odom['t'], opp_odom['yaw'])
        # the steering angle is not in the odometry: assume it has settled on the command applied two steps ago
        before = t_step[t_step <= t0][-3:]
        buf = [zoh(t)[0] for t in before[::-1][:2]] if len(before) >= 3 else [zoh(t0)[0]] * 2
        steer0 = zoh(before[0])[0] if len(before) >= 3 else zoh(t0)[0]
        wz = np.interp(t0, opp_odom['t'], opp_odom['wz'])
        xy = replay.run([*pos(t0), steer0, v, yaw, wz, 0.0], buf, [zoh(t) for t in steps])
        for h in HORIZONS_S:
            i = int(round(h / PHYSICS_DT_S)) - 1
            truth = pos(steps[i])
            err[h]['replay'].append(float(np.hypot(*(xy[i] - truth))))
            cv = pos(t0) + v * h * np.array([np.cos(yaw), np.sin(yaw)])
            err[h]['const_vel'].append(float(np.hypot(*(cv - truth))))
    out['replay_position_error_m'] = {
        str(h): {m: {'median': _median(e), 'p95': round(float(np.percentile(e, 95)), 3) if e else None, 'n': len(e)}
                 for m, e in err[h].items()} for h in HORIZONS_S}
    return out


def check_labels(run_dir: str, dataset_override: dict = None) -> dict:
    with open(os.path.join(run_dir, 'spec.yaml')) as f:
        spec = yaml.safe_load(f)
    with open(os.path.join(run_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    cfg = {**dataset_config(dataset_override if dataset_override is not None else spec.get('dataset')), 'frame': 'frenet'}
    replay = GymReplay(_sim_params())
    episodes = {e['name']: labels_episode(os.path.join(run_dir, 'episodes', e['name']), cfg, replay)
                for e in manifest['episodes'] if e.get('valid')}
    report = {'config': cfg, 'episodes': episodes}
    with open(os.path.join(run_dir, 'check_labels.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def print_labels(report: dict):
    fmt = lambda v, f='{:6.3f}': '     -' if v is None else f.format(v)
    print("State labels vs recorded velocities (x from the ground truth):")
    print(f"  {'ep':8s} {'policy':24s} {'rows':>4s} {'steps/s':>7s} {'y_ds/exp':>8s} {'ds rmse':>7s} {'dd rmse':>7s} "
          f"{'best shift':>10s}  rmse at shift -3..+3")
    for name, e in report['episodes'].items():
        if not e.get('rows'):
            print(f"  {name:8s} {e['opp_policy']:24s}    0")
            continue
        shifts = ' '.join(f"{v:.3f}" for v in e['label_shift_rmse_m'].values())
        print(f"  {name:8s} {e['opp_policy']:24s} {e['rows']:4d} {e['physics_steps_per_s']:7.1f} "
              f"{fmt(e['y_ds_over_expected'], '{:8.3f}')} {fmt(e['y_ds_error_m']['rmse'], '{:7.4f}')} "
              f"{fmt(e['y_dd_error_m']['rmse'], '{:7.4f}')} {e['best_shift']:+10d}  {shifts}")
    print("\nAction labels replayed through the gym model, position error [m] median / p95 (constant velocity for scale):")
    print(f"  {'ep':8s} {'policy':24s} " + ' '.join(f"{h:>9.2f} s replay {'':>4s} const vel" for h in HORIZONS_S))
    for name, e in report['episodes'].items():
        r = e.get('replay_position_error_m')
        if not r:
            continue
        cells = []
        for h in HORIZONS_S:
            a, b = r[str(h)]['replay'], r[str(h)]['const_vel']
            cells.append(f"{fmt(a['median'], '{:.3f}')}/{fmt(a['p95'], '{:.3f}')}  {fmt(b['median'], '{:.3f}')}/{fmt(b['p95'], '{:.3f}')}")
        print(f"  {name:8s} {e['opp_policy']:24s} " + '   '.join(cells))
