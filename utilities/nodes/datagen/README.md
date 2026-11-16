# datagen

Simulated head-to-head data generation for opponent prediction. One spec file in, one command, a dataset out.

```bash
ros2 run datagen mine utilities/nodes/datagen/specs/h2h_v1.yaml              # with RViz
ros2 run datagen mine utilities/nodes/datagen/specs/h2h_v1.yaml --headless   # no GUI
ros2 run datagen mine <spec> --dry-run                    # print the episode plan only
ros2 run datagen extract data/h2h_v1/<run>/ --spec other.yaml --name v2.csv   # new features, no re-simulation
ros2 run datagen check perception|policies|labels data/h2h_v1/<run>/          # sanity checks, see below
```

```
spec.yaml ─► mine: per episode  plan (seeded) ─► episode_launch.py ─► bag + result.json
                                                 (sim, ego policy, frozen opponent policy, episode_manager)
          ─► extract: bags ─► dataset.csv (+ _episodes.csv, _columns.json, _report.json)
```

Output: `<out>/<spec name>/<timestamp>/` with `spec.yaml`, `manifest.json` (git commit, per-episode outcome) and
`episodes/ep_XXXX/` (`episode.json` = fully resolved plan, `result.json`, `bag/`, `launch.log`).
`--out` defaults to `$DATAGEN_OUT` or `./data`.

## Spec

See [specs/h2h_v1.yaml](specs/h2h_v1.yaml). Episodes are assigned round robin over `tracks x opponents.policies`.
An opponent given as `{policy: name, tracks: [...]}` only runs on the tracks listed (e.g. a policy that can not lap
the others), and the round robin skips its other pairs.
Any value in `episode` or a policy file can be a distribution: `{uniform: [lo, hi], step: 0.01}`,
`{int_uniform: [lo, hi]}`, `{choice: [...]}`. Draws come from `(seed, episode index)` only.

The `dataset` section is applied offline (see `datagen/extract.py` for defaults and `datagen/features.py` for the
catalog): `frame: frenet|cartesian`, `reference: centerline|raceline|shortest_path`, `opp_source: tracker|gt`,
`features`, `labels: {type: state_delta|state|action, horizon: N}`, `history: H`.
x is built from the ego's tracker estimate (`opp_source: tracker`) or from the simulator (`gt`, used by
`h2h_v1`), y always from simulator ground truth.

Every feature and label declares how its columns change on the mirror image of the track (map y -> -y):
`@feature('opp_d', mirror={'opp_d': 'odd'})` with `even` (keeps), `odd` (changes sign) or the name of the column to
swap with. `mirrored(rows)` in `datagen/extract.py` applies these rules to dataset rows, e.g. for training on both
orientations of a track.

## Rules of one episode

Every episode is a fresh launch of the whole stack (no state carries over).

**Start**
1. Ego spawns on the raceline at `spawn_frac * L`, the opponent `start_gap_m` ahead, both at rest.
2. The simulator is frozen (start gate) until all are ready for `settle_s`: both odometries at the spawn,
   `/drive`, `/opp_drive` and the policies' `ready_topics` (e.g. `/perception/obstacles`) publishing.
   Not ready within `startup_timeout_s` -> `startup_failed`, retried with the same plan.
3. t = 0 when the gate opens. The first `warmup_s` are dropped at extraction.

All times of a dataset (`t`, `dt`, `warmup_s`, `drop_before_collision_s`) are simulated time. The gym advances the
physics on a 10 ms wall-clock timer and falls behind when the computer is loaded, so extraction counts simulated
time in physics steps (`/sim/collision/walls` is published once per step) instead of using the wall clock.

**End** (first one wins, stored in `result.json`)

| reason | condition | valid |
|---|---|---|
| `laps_done` | ego progress >= `laps` * L | yes |
| `timeout` | t > `timeout_s` | yes |
| `collision` | the cars overlap (`/sim/collision/cars`) or a car touches a wall (`/sim/collision/walls`); `detail` says which | yes, last `drop_before_collision_s` dropped |
| `stall` | a car slower than `stall.speed_mps` for `stall.duration_s` after warm-up | yes |
| `off_track` | \|d\| beyond the track bounds by `off_track_margin_m` | yes |
| `decoupled` | \|Δs\| > `decouple.gap_m` for `decouple.duration_s` (optional) | yes |
| `policy_changed` | a node in `/opp_racecar` changed a parameter after t = 0 | no |
| `stale_input` | a required topic silent for `stale_input_s` | no |
| `sim_too_slow` | fewer than `min_physics_rate_hz` physics steps per wall-clock second (checked at the end, original reason in `detail`) | no |
| `startup_failed` / `aborted` | see above / launch died | no |

Reproducibility: same spec + seed + commit give the same plan and rules. Trajectories are not bit-identical,
since the simulator steps on a wall-clock timer and nodes run asynchronously. The stacks run on the wall clock too,
so a slow simulator changes their behaviour relative to the physics; `sim_too_slow` keeps such episodes out.

## Policies

A policy is a YAML file, found by name in the spec folder, `$DATAGEN_POLICY_PATH` and `policies/`, or by path.
It is resolved (sampled) once per episode and frozen; its hash is stored per episode and per row.

- `kind: launch` any launch file, e.g. the race stack time trials stack ([opp_tt_raceline.yaml](policies/opp_tt_raceline.yaml)).
  `params` are keyed by fully qualified node name (parameters a node loads from its own param file can not be overridden).
- `kind: python` a class implementing `datagen.policy.Policy.act(obs) -> (steer, speed)`, given as
  `entrypoint: file.py:Class` ([line_follower.py](policies/line_follower.py), [blocker.py](policies/blocker.py)).
  Run by `policy_runner` with read-only parameters. A policy with `needs_scan = True` also gets its own LiDAR in
  `obs.scan` (from `/opp_scan`), and the runner waits for it before driving.

Third-party policies: [benchmark.py](policies/benchmark.py) runs the planners of
[f1tenth_benchmarks](https://github.com/BDEvan5/f1tenth_benchmarks) (Evans et al. 2024, Apache-2.0) unchanged from
the pinned submodule `utilities/third_party/f1tenth_benchmarks` (`git submodule update --init` after cloning).
[opp_bench_pp.yaml](policies/opp_bench_pp.yaml) is its pure pursuit on a datagen line (blind to the ego) and
[opp_bench_ftg.yaml](policies/opp_bench_ftg.yaml) its follow-the-gap, which steers from the LiDAR only and so
reacts to the ego in its scan. `overrides` replace values of upstream's `params/<Planner>.yaml`. Follow-the-gap
laps only glc_ot_ez and test; both policy files say why and how their ranges were checked.

Contract: opponent nodes live in `/opp_racecar` and publish `/opp_drive`; the ego publishes `/drive`.

Perfect perception for the ego: an ego policy with `ground_truth_obstacles: true` (and its own detection and
tracking off) gets the true opponent on `/perception/obstacles` from datagen's `gt_obstacles` node, in the
tracker's format and frame. [ego_h2h_spliner_gt.yaml](policies/ego_h2h_spliner_gt.yaml) is the race stack
head-to-head stack in this mode; [ego_h2h_spliner.yaml](policies/ego_h2h_spliner.yaml) keeps the real perception.

Ego perception: `ground_truth_obstacles: true` in an ego policy starts datagen's `gt_obstacles` node, which publishes
the true opponent on `/perception/obstacles` in the tracker's format, so the ego reacts as if perception were perfect.
[ego_h2h_spliner_gt.yaml](policies/ego_h2h_spliner_gt.yaml) uses it and turns the race stack's detection and tracking
off (`perception: 'False'`); [ego_h2h_spliner.yaml](policies/ego_h2h_spliner.yaml) keeps them, e.g. for `check perception`.

## Limits

- One opponent: the gym bridge supports `num_agent` <= 2.
- Frenet frame on tight corners: where κ·d approaches 1 (car near/past the centre of curvature) s is not unique.
  These samples are dropped (`frenet_singular_kappa_d`) and counted in `_report.json`.

## Checks

`ros2 run datagen check <test> <run_dir>` writes `check_<test>.json` into the run folder.

| test | question | method |
|---|---|---|
| `perception` | Is the pipeline or the ego's perception the problem? | detector and tracker vs ground truth per grid step; dataset built twice (x from tracker / from ground truth) |
| `policies` | Do the values match the opponent policies? | python policies replayed from the recorded state (scan policies only if the spec records `/opp_scan`); motion vs sampled parameters |
| `labels` | Do the labels match the raw recording? | state labels vs recorded velocities (also time-shifted, and on a wall-clock grid for comparison); action labels replayed through the gym model |

[specs/known_answer.yaml](specs/known_answer.yaml) records opponents with fixed, known parameters for `policies`.

## Data folder

`data/` (git-ignored, the default `--out`) has one folder per spec name and in it one timestamped run per `mine`
call. A run's `manifest.json` names the commit it was mined at.

| folder | spec | context |
|---|---|---|
| `h2h_v1/` | [h2h_v1](specs/h2h_v1.yaml) | First head-to-head dataset, glc_ot_ez only: the race stack head-to-head ego against the time-trials raceline, line follower and blocker opponents. Recorded with the perfect-perception ego and ground-truth x. The run also holds `dataset_h3.csv` (history 3, extracted later) and the predictors trained on it in `models/` (`baseline`, `history3`, `history3_mirror`). |
| `smoke/` | none | `diag_speed.py` and `diag_tracker.py`, diagnosis scripts from a smoke run of h2h_v1: what the ego's perception reported of the opponent, against the opponent's true state. |
| `known_answer/` | [known_answer](specs/known_answer.yaml) | Opponents with fixed, known parameters (line followers at a left and right offset, blockers) for `check policies`, recorded with the perfect-perception ego. |
| `h2h_pilot_test/` | [h2h_pilot_test](specs/h2h_pilot_test.yaml) | h2h_v1 on the hand-drawn `test` map, the first new track for the track-variety study. |
| `h2h_pilot_maps/` | [h2h_pilot_maps](specs/h2h_pilot_maps.yaml) | h2h_v1 on GLC_smile_small and hangar_1905_v0. `action_plots_history3[_mirror]` are the history-3 predictors' action plots on it. |
| `h2h_v2/`, `h2h_v2_mine.log` | [h2h_v2](specs/h2h_v2.yaml) | The h2h_v1 opponents on all four tracks, 240 episodes (122k rows). `models/variety` is the track-variety ensemble study (per-seed models, rollout and feedback results). The log is the console output of `mine`. |
| `bench_pilot/` | [bench_pilot](specs/bench_pilot.yaml) | Both f1tenth_benchmarks opponents (pure pursuit, follow-the-gap). On glc_ot_ez and test, the tracks follow-the-gap laps. |
| `opp_h2h_pilot/` | [opp_h2h_pilot](specs/opp_h2h_pilot.yaml) | The race stack head-to-head stack as the opponent (both cars with perfect perception), starting ahead of and behind the ego, with RViz. |

## Terms

| term | meaning |
|---|---|
| spec | experiment file: tracks, ego, opponent policies, episode rules, dataset representation |
| run | one `mine` call: a folder with all episodes of a spec |
| episode | one fresh launch from start gate to end reason; `episode.json` is its plan, `result.json` its outcome |
| policy | the driver of a car (launch file or python class), frozen for the whole episode |
| physics step | one 10 ms step of the gym; simulated time = number of steps x 10 ms |
| grid | the sample times of an episode, every `dt` of simulated time |
| row | one grid time of one episode in `dataset.csv`: meta columns, features x, labels y |

