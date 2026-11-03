"""
Ground-truth perception for the ego in simulation: publishes the opponent's true state on /perception/obstacles,
in the frame and format of the race stack tracker, in place of its detection and tracking. The ego then reacts to
the opponent as if perception were perfect, independent of how the detection module develops.
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from f110_msgs.msg import Obstacle, ObstacleArray
from nav_msgs.msg import Odometry

from datagen.extract import TRACKER_LINE
from datagen.track import Track, body_to_map

OPP_ODOM = '/opp_racecar/odom'


def obstacle_from_state(line: Track, x: float, y: float, yaw: float, vx_body: float, vy_body: float,
                        size_m: float) -> dict:
    """Obstacle fields of a car at (x, y, yaw) with body velocity (vx, vy), as the tracker reports a dynamic one."""
    s, d = line.to_frenet(x, y)
    s, d = float(s), float(d)
    vs, vd = line.frenet_velocity(s, *body_to_map(yaw, vx_body, vy_body))
    half = size_m / 2
    return {'id': 1, 'size': size_m, 's_center': s, 'd_center': d, 'vs': float(vs), 'vd': float(vd),
            's_start': (s - half) % line.length, 's_end': (s + half) % line.length,
            'd_right': d - half, 'd_left': d + half,
            'is_static': False, 'is_visible': True, 'is_actually_a_gap': False}


class GtObstacles(Node):
    def __init__(self):
        super().__init__('gt_obstacles')
        ro = ParameterDescriptor(read_only=True)
        # size_m: extent along and across the line, like the tracker's single size; 0.58 m is the sim car length
        for name, default in (('map_dir', ''), ('size_m', 0.58), ('rate_hz', 40.0)):
            self.declare_parameter(name, default, ro)
        wpnts = os.path.join(self.get_parameter('map_dir').value, 'global_waypoints.json')
        self.line = Track.from_json(wpnts, TRACKER_LINE)
        self.size = self.get_parameter('size_m').value
        self.opp = None
        self.create_subscription(Odometry, OPP_ODOM, lambda m: setattr(self, 'opp', m), 10)
        self.pub = self.create_publisher(ObstacleArray, '/perception/obstacles', 5)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.step)
        self.get_logger().info("Publishing the true opponent on /perception/obstacles")

    def step(self):
        if self.opp is None:
            return
        p, q, tw = self.opp.pose.pose.position, self.opp.pose.pose.orientation, self.opp.twist.twist
        yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)))
        msg = ObstacleArray()
        msg.header.stamp = self.opp.header.stamp
        msg.header.frame_id = 'map'
        msg.obstacles = [Obstacle(**obstacle_from_state(self.line, p.x, p.y, yaw, tw.linear.x, tw.linear.y, self.size))]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GtObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
