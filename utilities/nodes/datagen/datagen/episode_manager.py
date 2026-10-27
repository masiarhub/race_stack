"""
Runs the rules of one episode: readiness gate -> start gate -> monitor end conditions -> result file.

Ends the node when the episode is over, the episode launch file then shuts everything down.
"""
import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterEvent
from std_msgs.msg import Empty, String, UInt8MultiArray

from datagen.track import Track

EGO_ODOM = '/state_estimation/odom'
OPP_ODOM = '/opp_racecar/odom'
REQUIRED = (EGO_ODOM, OPP_ODOM, '/drive', '/opp_drive')
SPAWN_TOLERANCE_M = 0.3
OPP_NAMESPACE = '/opp_racecar'


class EpisodeManager(Node):
    def __init__(self):
        super().__init__('episode_manager')
        self.declare_parameter('episode_file', '')
        self.declare_parameter('result_file', '')
        with open(self.get_parameter('episode_file').value) as f:
            self.ep = json.load(f)
        self.rules = self.ep['rules']
        wpnts = os.path.join(self.ep['map_dir'], 'global_waypoints.json')
        self.centerline = Track.from_json(wpnts, 'centerline')
        raceline = Track.from_json(wpnts, 'raceline')
        ego_s = self.ep['spawn']['ego_s_m']
        self.expected_spawn = {
            EGO_ODOM: np.array(raceline.to_cartesian(ego_s, 0.0)),
            OPP_ODOM: np.array(raceline.to_cartesian(ego_s + self.ep['spawn']['gap_m'], 0.0)),
        }

        self.phase = 'waiting'
        self.t_launch = self.now()
        self.t0 = None
        self.ready_since = None
        self.last_log = self.t_launch
        self.done = False

        self.last_rx = {}  # topic -> last receive time [s]
        self.odom = {}     # topic -> latest Odometry
        self.collided = [False, False]
        self.policy_change = None
        self.ego_progress = 0.0
        self.ego_s_prev = None
        self.low_speed_since = {EGO_ODOM: None, OPP_ODOM: None}
        self.decoupled_since = None

        for topic in (EGO_ODOM, OPP_ODOM):
            self.create_subscription(Odometry, topic, lambda m, t=topic: self.odom_cb(t, m), 10)
        for topic in ('/drive', '/opp_drive'):
            self.create_subscription(AckermannDriveStamped, topic, lambda m, t=topic: self.touch(t), 10)
        self.create_subscription(UInt8MultiArray, '/sim/collisions', self.collision_cb, 10)
        self.create_subscription(ParameterEvent, '/parameter_events', self.param_event_cb, 100)
        # policy specific readiness topics (e.g. the ego tracker), their type is only known once published
        self.extra_ready = sorted(set(self.ep['ego_policy']['ready_topics'] + self.ep['opp_policy']['ready_topics']))
        self.extra_subs = {}

        self.start_pub = self.create_publisher(Empty, '/sim/start', 10)
        self.event_pub = self.create_publisher(String, '/datagen/events', 10)
        self.create_timer(0.02, self.tick)
        self.get_logger().info(f"Episode {self.ep['name']}: {self.ep['track']}, opponent {self.ep['opp_policy']['name']}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def touch(self, topic):
        self.last_rx[topic] = self.now()

    def odom_cb(self, topic, msg):
        self.odom[topic] = msg
        self.touch(topic)

    def collision_cb(self, msg):
        if self.phase == 'running':
            # sticky, a one step contact between two ticks must not be missed
            self.collided = [a or bool(b) for a, b in zip(self.collided, msg.data)]

    def param_event_cb(self, msg):
        if self.phase == 'running' and msg.node.startswith(OPP_NAMESPACE + '/') \
                and (msg.changed_parameters or msg.deleted_parameters) and self.policy_change is None:
            names = [p.name for p in list(msg.changed_parameters) + list(msg.deleted_parameters)]
            self.policy_change = f"{msg.node}: {names}"

    def subscribe_extra_ready(self):
        types = dict(self.get_topic_names_and_types())
        for topic in self.extra_ready:
            if topic not in self.extra_subs and topic in types:
                self.extra_subs[topic] = self.create_subscription(
                    get_message(types[topic][0]), topic, lambda m, t=topic: self.touch(t), 10)

    def emit(self, event: dict):
        self.event_pub.publish(String(data=json.dumps(event)))

    # ---------------------------------------------------------------- phases
    def tick(self):
        if self.done:
            return
        if self.phase == 'waiting':
            self.wait_for_ready()
        elif self.phase == 'running':
            self.check_end()

    def readiness(self) -> dict:
        now = self.now()
        fresh = lambda t: t in self.last_rx and now - self.last_rx[t] < 0.5
        checks = {f"publishing {t}": fresh(t) for t in REQUIRED + tuple(self.extra_ready)}
        for topic, xy in self.expected_spawn.items():
            msg = self.odom.get(topic)
            at_spawn = msg is not None and np.hypot(msg.pose.pose.position.x - xy[0],
                                                     msg.pose.pose.position.y - xy[1]) < SPAWN_TOLERANCE_M
            checks[f"{topic} at spawn"] = at_spawn
        return checks

    def wait_for_ready(self):
        now = self.now()
        if now - self.t_launch > self.rules['startup_timeout_s']:
            missing = [k for k, ok in self.readiness().items() if not ok]
            self.finish('startup_failed', valid=False, detail=f"not ready: {missing}")
            return
        self.subscribe_extra_ready()
        checks = self.readiness()
        if all(checks.values()):
            if self.ready_since is None:
                self.ready_since = now
            elif now - self.ready_since >= self.rules['settle_s']:
                self.start()
        else:
            self.ready_since = None
            if now - self.last_log > 5.0:
                self.last_log = now
                self.get_logger().info(f"Waiting for: {[k for k, ok in checks.items() if not ok]}")

    def start(self):
        self.phase = 'running'
        self.t0 = self.now()
        self.t0_ns = self.get_clock().now().nanoseconds
        self.start_pub.publish(Empty())
        self.emit({'event': 'start', 'episode': self.ep['name'], 't_ns': self.t0_ns})
        self.get_logger().info(f"Start gate opened after {self.t0 - self.t_launch:.1f}s")

    def frenet(self, topic):
        p = self.odom[topic].pose.pose.position
        s, d = self.centerline.to_frenet(p.x, p.y)
        return float(s), float(d)

    def check_end(self):
        now = self.now()
        t = now - self.t0
        r = self.rules
        if t < 1.0:
            self.start_pub.publish(Empty())  # repeated briefly in case the first one was lost

        if self.policy_change:
            return self.finish('policy_changed', valid=False, detail=self.policy_change)
        for topic in REQUIRED:
            if now - self.last_rx.get(topic, 0.0) > r['stale_input_s']:
                return self.finish('stale_input', valid=False, detail=topic)
        if any(self.collided):
            who = [n for n, c in zip(('ego', 'opp'), self.collided) if c]
            return self.finish('collision', detail=f"{who}")

        (s_ego, d_ego), (s_opp, d_opp) = self.frenet(EGO_ODOM), self.frenet(OPP_ODOM)
        if self.ego_s_prev is not None:
            self.ego_progress += float(self.centerline.wrap_ds(s_ego - self.ego_s_prev))
        self.ego_s_prev = s_ego
        if self.ego_progress >= r['laps'] * self.centerline.length:
            return self.finish('laps_done')
        if t > r['timeout_s']:
            return self.finish('timeout')

        for name, topic, s, d in (('ego', EGO_ODOM, s_ego, d_ego), ('opp', OPP_ODOM, s_opp, d_opp)):
            if d > self.centerline.interp(self.centerline.d_left, s) + r['off_track_margin_m'] or \
                    -d > self.centerline.interp(self.centerline.d_right, s) + r['off_track_margin_m']:
                return self.finish('off_track', detail=name)
            if t > r['warmup_s']:
                v = self.odom[topic].twist.twist.linear
                if np.hypot(v.x, v.y) < r['stall']['speed_mps']:
                    self.low_speed_since[topic] = self.low_speed_since[topic] or now
                    if now - self.low_speed_since[topic] > r['stall']['duration_s']:
                        return self.finish('stall', detail=name)
                else:
                    self.low_speed_since[topic] = None

        if r.get('decouple') and t > r['warmup_s']:
            if abs(self.centerline.wrap_ds(s_opp - s_ego)) > r['decouple']['gap_m']:
                self.decoupled_since = self.decoupled_since or now
                if now - self.decoupled_since > r['decouple']['duration_s']:
                    return self.finish('decoupled')
            else:
                self.decoupled_since = None

    def finish(self, reason: str, valid: bool = True, detail: str = ''):
        self.done = True
        self.phase = 'done'
        t_end_ns = self.get_clock().now().nanoseconds
        result = {
            'episode': self.ep['name'],
            'valid': valid,
            'reason': reason,
            'detail': detail,
            't0_ns': self.t0_ns if self.t0 is not None else None,
            't_end_ns': t_end_ns,
            'duration_s': (t_end_ns - self.t0_ns) * 1e-9 if self.t0 is not None else 0.0,
            'startup_s': (self.t0 if self.t0 is not None else self.now()) - self.t_launch,
            'ego_progress_m': self.ego_progress,
            'ego_laps': self.ego_progress / self.centerline.length,
        }
        self.emit({'event': 'end', **result})
        with open(self.get_parameter('result_file').value, 'w') as f:
            json.dump(result, f, indent=2)
        log = self.get_logger().info if valid else self.get_logger().warn
        log(f"Episode ended: {reason} {detail} after {result['duration_s']:.1f}s")


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
