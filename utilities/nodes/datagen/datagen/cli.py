"""
Command line entry points.

    ros2 run datagen mine <spec.yaml> [--headless] [--episodes N] [--out DIR] [--dry-run] [--no-extract]
    ros2 run datagen extract <run_dir> [--spec other.yaml] [--name dataset.csv]
    ros2 run datagen check perception|policies|labels <run_dir> [--spec other.yaml]

Output of `mine`:
    <out>/<spec name>/<timestamp>/
        spec.yaml, manifest.json            what was asked and what happened
        episodes/ep_0000/                   episode.json (resolved plan), result.json, bag/, launch.log
        dataset.csv (+ _episodes.csv, _columns.json, _report.json)
"""
import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

from datagen.extract import dataset_config, extract_run
from datagen.spec import load_spec, plan_episode

SHUTDOWN_GRACE_S = 20.0  # [s] for the launch to stop before it is killed


def git_info(path: str) -> dict:
    def git(*args):
        return subprocess.run(['git', '-C', path, *args], capture_output=True, text=True).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
            'submodules': git('submodule', 'status')}


def sim_running() -> bool:
    return subprocess.run(['pgrep', '-f', 'gym_bridge'], capture_output=True).returncode == 0


def stop(proc: subprocess.Popen):
    """SIGINT to the whole launch process group (lets the bag close cleanly), SIGKILL if it hangs."""
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(SHUTDOWN_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_episode(ep_dir: str, rules: dict, visualize: bool) -> dict:
    for leftover in ('bag', 'result.json'):  # a retry starts from scratch
        path = os.path.join(ep_dir, leftover)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    cmd = ['ros2', 'launch', 'datagen', 'episode_launch.py',
           f"episode:={os.path.join(ep_dir, 'episode.json')}", f"rviz:={visualize}"]
    hard_timeout = rules['startup_timeout_s'] + rules['timeout_s'] + 60
    with open(os.path.join(ep_dir, 'launch.log'), 'a') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(hard_timeout)
        except subprocess.TimeoutExpired:
            pass
        finally:
            stop(proc)
    # every node of the episode must be gone before the next one starts
    deadline = time.time() + SHUTDOWN_GRACE_S
    while sim_running() and time.time() < deadline:
        time.sleep(0.5)
    result_file = os.path.join(ep_dir, 'result.json')
    if not os.path.exists(result_file):
        return {'valid': False, 'reason': 'aborted', 'detail': 'no result written, see launch.log', 'duration_s': 0.0}
    with open(result_file) as f:
        return json.load(f)


def mine(argv=None):
    parser = argparse.ArgumentParser(prog='ros2 run datagen mine', description='Generate a head-to-head dataset in simulation.')
    parser.add_argument('spec', help='experiment spec YAML')
    parser.add_argument('--headless', action='store_true', help='no RViz')
    parser.add_argument('--episodes', type=int, help='override the number of episodes of the spec')
    parser.add_argument('--out', default=os.environ.get('DATAGEN_OUT', os.path.join(os.getcwd(), 'data')),
                        help='output root (default $DATAGEN_OUT or ./data)')
    parser.add_argument('--dry-run', action='store_true', help='print the episode plan without simulating')
    parser.add_argument('--no-extract', action='store_true', help='only record the raw episodes')
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    if args.episodes:
        spec['episodes'] = args.episodes
    dataset_config(spec['dataset'])  # fail on a bad dataset section before simulating for an hour
    plans = [plan_episode(spec, i) for i in range(spec['episodes'])]

    if args.dry_run:
        for p in plans:
            print(f"{p['name']}  {p['track']:<16} gap {p['spawn']['gap_m']:4.1f} m  ego_s {p['spawn']['ego_s_m']:5.1f} m  "
                  f"opp {p['opp_policy']['name']} {json.dumps({**p['opp_policy'].get('args', {}), **p['opp_policy']['params']})}")
        return
    if sim_running():
        sys.exit("A simulator (gym_bridge) is already running, stop it first so episodes do not interfere.")

    run_dir = os.path.join(args.out, spec['name'], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.join(run_dir, 'episodes'))
    spec_copy = {k: v for k, v in spec.items() if not k.startswith('_')}
    with open(os.path.join(run_dir, 'spec.yaml'), 'w') as f:
        yaml.safe_dump(spec_copy, f, sort_keys=False)
    manifest = {'spec': spec['_path'], 'created': datetime.datetime.now().isoformat(timespec='seconds'),
                'git': git_info(os.path.dirname(spec['_path'])), 'headless': args.headless, 'episodes': []}

    def save_manifest():
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

    print(f"Run folder: {run_dir}")
    try:
        for plan in plans:
            ep_dir = os.path.join(run_dir, 'episodes', plan['name'])
            os.makedirs(ep_dir)
            with open(os.path.join(ep_dir, 'episode.json'), 'w') as f:
                json.dump(plan, f, indent=2)
            for attempt in range(plan['rules']['max_retries'] + 1):
                result = run_episode(ep_dir, plan['rules'], not args.headless)
                if result['reason'] not in ('startup_failed', 'aborted'):
                    break
                print(f"  {plan['name']}: {result['reason']} ({result.get('detail', '')}), retry {attempt + 1}")
            manifest['episodes'].append({'name': plan['name'], 'attempts': attempt + 1,
                                         **{k: result.get(k) for k in ('valid', 'reason', 'detail', 'duration_s', 'ego_laps')}})
            save_manifest()
            print(f"{plan['name']}  {plan['track']:<16} {plan['opp_policy']['name']:<20} "
                  f"{'ok' if result['valid'] else 'INVALID':<8} {result['reason']:<14} {result['duration_s']:5.1f}s")
    except KeyboardInterrupt:
        print("Interrupted, the episodes recorded so far are kept.")
    save_manifest()

    if not args.no_extract and any(e['valid'] for e in manifest['episodes']):
        out = extract_run(run_dir)
        print(f"Dataset: {out}")


def extract(argv=None):
    parser = argparse.ArgumentParser(prog='ros2 run datagen extract', description='(Re-)derive the dataset of a recorded run.')
    parser.add_argument('run_dir')
    parser.add_argument('--spec', help='take the dataset section from this spec instead of the recorded one')
    parser.add_argument('--name', default='dataset.csv', help='output file name inside the run folder')
    args = parser.parse_args(argv)
    override = None
    if args.spec:
        with open(args.spec) as f:
            override = yaml.safe_load(f).get('dataset', {})
    print(f"Dataset: {extract_run(args.run_dir, override, args.name)}")


def check(argv=None):
    from datagen import checks
    parser = argparse.ArgumentParser(prog='ros2 run datagen check', description='Sanity checks of a recorded run.')
    parser.add_argument('test', choices=['perception', 'policies', 'labels'])
    parser.add_argument('run_dir')
    parser.add_argument('--spec', help='take the dataset section from this spec instead of the recorded one')
    args = parser.parse_args(argv)
    override = None
    if args.spec:
        with open(args.spec) as f:
            override = yaml.safe_load(f).get('dataset', {})
    report = getattr(checks, f'check_{args.test}')(args.run_dir, override)
    getattr(checks, f'print_{args.test}')(report)
    print(f"\nReport: {os.path.join(args.run_dir, f'check_{args.test}.json')}")
