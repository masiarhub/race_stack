"""
Drives the simulated opponent with a python policy (see datagen.policy.Policy).

Runs in the /opp_racecar namespace: reads its own odometry and the ego odometry as seen by the opponent,
publishes /opp_drive. All parameters are read-only, so the policy is frozen for the whole episode.
"""
import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from datagen.policy import AgentState, Obs, load_policy_class, state_from_odom
from datagen.track import LINE_KEYS, Track


class PolicyRunner(Node):
    def __init__(self):
        super().__init__('policy_runner')
        ro = ParameterDescriptor(read_only=True)
        for name, default in (('entrypoint', ''), ('params_json', '{}'), ('seed', 0), ('map_dir', ''), ('rate_hz', 50.0)):
            self.declare_parameter(name, default, ro)
        wpnts = os.path.join(self.get_parameter('map_dir').value, 'global_waypoints.json')
        self.tracks = {line: Track.from_json(wpnts, line) for line in LINE_KEYS}
        entrypoint = self.get_parameter('entrypoint').value
        self.policy = load_policy_class(entrypoint)(
            json.loads(self.get_parameter('params_json').value), self.tracks,
            np.random.default_rng(self.get_parameter('seed').value))

        self.own = None
        self.other = None
        self.t_first = None
        self.create_subscription(Odometry, 'odom', lambda m: setattr(self, 'own', m), 10)
        self.create_subscription(Odometry, 'opp_odom', lambda m: setattr(self, 'other', m), 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.step)
        self.get_logger().info(f"Policy {entrypoint} loaded")

    def state(self, msg: Odometry) -> AgentState:
        return state_from_odom(msg, self.tracks['centerline'])

    def step(self):
        if self.own is None or self.other is None:
            return
        now = self.get_clock().now()
        self.t_first = self.t_first or now.nanoseconds
        obs = Obs(t=(now.nanoseconds - self.t_first) * 1e-9, opp=self.state(self.own), ego=self.state(self.other))
        steer, speed = self.policy.act(obs)
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'opp_racecar/base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
