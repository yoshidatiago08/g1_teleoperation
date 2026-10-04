#!/usr/bin/env python3
"""Position-only, RViz visualization IK for the G1 right arm.

This node publishes visualization joint states; it does not command robot hardware.
The kinematic chain and joint limits are loaded from the selected URDF.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.optimize import minimize


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(joint):
    origin = joint.find('origin')
    if origin is None:
        return np.eye(4)
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    t = np.eye(4)
    t[:3, :3] = _rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


class G1ArmIK(Node):
    def __init__(self):
        super().__init__('g1_arm_ik_node')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('base_frame', 'torso_link')
        self.declare_parameter('target_topic', '/wrist_pose')
        self.declare_parameter('joint_state_topic', '/g1_visualization/joint_states')
        path = self.get_parameter('urdf_path').value
        if not path:
            raise RuntimeError('urdf_path must point to the selected G1 URDF')
        self.base_frame = self.get_parameter('base_frame').value
        self.all_joint_names, self.all_joint_positions = [], []
        self.chain = self._load_chain(path)
        self.chain_names = [j['name'] for j in self.chain]
        self.q = np.zeros(len(self.chain))
        self.bounds = [(j['lower'], j['upper']) for j in self.chain]
        self.publisher = self.create_publisher(
            JointState, self.get_parameter('joint_state_topic').value, 10)
        self.create_subscription(PoseStamped,
                                 self.get_parameter('target_topic').value,
                                 self.target_callback, 10)
        self.create_timer(1.0 / 30.0, self.publish_state)
        self.get_logger().info(f'URDF chain: {self.chain_names}; visualization only')

    def _load_chain(self, path):
        root = ET.parse(path).getroot()
        joints = root.findall('joint')
        child_map = {j.find('child').get('link'): j for j in joints if j.find('child') is not None}
        # Walk back from the right wrist and keep the movable arm joints.
        cur, reverse = 'right_wrist_yaw_link', []
        while cur != self.base_frame:
            joint = child_map.get(cur)
            if joint is None:
                raise RuntimeError(f'No URDF chain from {self.base_frame} to right_wrist_yaw_link (stopped at {cur})')
            reverse.append(joint)
            cur = joint.find('parent').get('link')
        chain = []
        for joint in reversed(reverse):
            typ = joint.get('type')
            if typ not in ('revolute', 'continuous', 'prismatic'):
                continue
            limit = joint.find('limit')
            if typ == 'continuous':
                lo, hi = -math.pi, math.pi
            else:
                if limit is None:
                    raise RuntimeError(f'Missing limits for {joint.get("name")}')
                lo, hi = float(limit.get('lower')), float(limit.get('upper'))
            axis = joint.find('axis')
            direction = np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.array([1., 0., 0.])
            chain.append({'name': joint.get('name'), 'type': typ, 'origin': _origin(joint),
                          'axis': direction, 'lower': lo, 'upper': hi})
        if len(chain) != 7:
            raise RuntimeError(f'Expected 7 movable right-arm joints, found {len(chain)}: {[j["name"] for j in chain]}')
        # Include every movable joint in the URDF so fixed-base/waist links are well-defined in RViz.
        seen = set()
        for joint in joints:
            typ = joint.get('type')
            if typ not in ('revolute', 'continuous', 'prismatic'):
                continue
            name = joint.get('name')
            if name in seen:
                continue
            seen.add(name)
            self.all_joint_names.append(name)
            self.all_joint_positions.append(0.0)
        return chain

    def _forward(self, q):
        t = np.eye(4)
        for joint, value in zip(self.chain, q):
            t = t @ joint['origin']
            axis = joint['axis']
            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
                n = axis / np.linalg.norm(axis)
                k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
                rot = np.eye(3) + math.sin(value) * k + (1 - math.cos(value)) * (k @ k)
                r = np.eye(4); r[:3, :3] = rot; t = t @ r
            else:
                trans = np.eye(4); trans[:3, 3] = axis * value; t = t @ trans
        return t[:3, 3]

    def target_callback(self, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if msg.header.frame_id != self.base_frame or not np.all(np.isfinite(p)):
            self.get_logger().warning(f'Ignoring invalid target; expected finite pose in {self.base_frame}')
            return
        result = minimize(lambda q: np.sum((self._forward(q) - p) ** 2) + 1e-5 * np.sum((q - self.q) ** 2),
                          self.q, method='L-BFGS-B', bounds=self.bounds,
                          options={'maxiter': 80, 'ftol': 1e-7})
        if np.all(np.isfinite(result.x)):
            self.q = np.clip(result.x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            for name, value in zip(self.chain_names, self.q):
                self.all_joint_positions[self.all_joint_names.index(name)] = float(value)

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.all_joint_names
        msg.position = self.all_joint_positions
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = G1ArmIK()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
