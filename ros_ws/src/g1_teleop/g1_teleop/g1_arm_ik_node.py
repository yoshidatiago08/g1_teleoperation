#!/usr/bin/env python3
"""RViz visualization IK for one G1 arm.

The wrist position is the main target. An optional elbow target only chooses the arm's
posture: with the shoulder and wrist fixed, the elbow can still swing on a circle, and the
node picks the point on that circle where the robot's forearm points the same way as the
operator's. Only the *direction* elbow -> wrist is used, so it needs no scale and no
shoulder calibration.

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

from g1_teleop.urdf_info import joint_origin as _origin


class G1ArmIK(Node):
    def __init__(self):
        super().__init__('g1_arm_ik_node')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('base_frame', 'torso_link')
        self.declare_parameter('target_topic', '/wrist_pose')
        self.declare_parameter('joint_state_topic', '/g1_visualization/joint_states')
        self.declare_parameter('side', 'right')
        self.declare_parameter('elbow_topic', '/elbow_pose')
        # Weight of the forearm-direction term against the wrist position error (in m^2).
        # 0 disables it and gives wrist-only IK.
        self.declare_parameter('elbow_weight', 0.01)
        # Ignore an elbow target older than this (s); the solver falls back to wrist-only.
        self.declare_parameter('elbow_timeout', 0.3)
        self.side = self.get_parameter('side').value
        self.tip_link = f'{self.side}_wrist_yaw_link'
        self.elbow_weight = float(self.get_parameter('elbow_weight').value)
        self.elbow_timeout = float(self.get_parameter('elbow_timeout').value)
        self.elbow_target = None  # (xyz, receive time in s)
        path = self.get_parameter('urdf_path').value
        if not path:
            raise RuntimeError('urdf_path must point to the selected G1 URDF')
        self.base_frame = self.get_parameter('base_frame').value
        self.all_joint_names, self.all_joint_positions = [], []
        self.chain = self._load_chain(path)
        self.chain_names = [j['name'] for j in self.chain]
        self.q = np.zeros(len(self.chain))
        self.bounds = [(j['lower'], j['upper']) for j in self.chain]
        names = self.chain_names
        elbow_joint, wrist_roll_joint = f'{self.side}_elbow_joint', f'{self.side}_wrist_roll_joint'
        self.elbow_idx = names.index(elbow_joint) if elbow_joint in names else None
        self.wrist_roll_idx = names.index(wrist_roll_joint) if wrist_roll_joint in names else None
        if self.elbow_weight > 0.0 and (self.elbow_idx is None or self.wrist_roll_idx is None):
            self.get_logger().warning(
                f'{elbow_joint}/{wrist_roll_joint} not in the chain: elbow term disabled')
            self.elbow_weight = 0.0
        self.publisher = self.create_publisher(
            JointState, self.get_parameter('joint_state_topic').value, 10)
        self.create_subscription(PoseStamped,
                                 self.get_parameter('target_topic').value,
                                 self.target_callback, 10)
        self.create_subscription(PoseStamped, self.get_parameter('elbow_topic').value,
                                 self.elbow_callback, 10)
        self.create_timer(1.0 / 30.0, self.publish_state)
        self.get_logger().info(
            f'URDF chain: {self.chain_names}; elbow weight {self.elbow_weight}; visualization only')

    def _load_chain(self, path):
        root = ET.parse(path).getroot()
        joints = root.findall('joint')
        child_map = {j.find('child').get('link'): j for j in joints if j.find('child') is not None}
        # Walk back from the wrist and keep the movable arm joints.
        cur, reverse = self.tip_link, []
        while cur != self.base_frame:
            joint = child_map.get(cur)
            if joint is None:
                raise RuntimeError(f'No URDF chain from {self.base_frame} to {self.tip_link} (stopped at {cur})')
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
            raise RuntimeError(f'Expected 7 movable {self.side}-arm joints, found {len(chain)}: {[j["name"] for j in chain]}')
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
        """Wrist (tip link) position."""
        return self._points(q)[0]

    def _points(self, q):
        """Positions of the wrist, the elbow joint and the wrist-roll joint, in the base frame."""
        t = np.eye(4)
        elbow = wrist_roll = None
        for i, (joint, value) in enumerate(zip(self.chain, q)):
            t = t @ joint['origin']
            if i == self.elbow_idx:
                elbow = t[:3, 3].copy()
            if i == self.wrist_roll_idx:
                wrist_roll = t[:3, 3].copy()
            axis = joint['axis']
            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
                n = axis / np.linalg.norm(axis)
                k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
                rot = np.eye(3) + math.sin(value) * k + (1 - math.cos(value)) * (k @ k)
                r = np.eye(4); r[:3, :3] = rot; t = t @ r
            else:
                trans = np.eye(4); trans[:3, 3] = axis * value; t = t @ trans
        return t[:3, 3].copy(), elbow, wrist_roll

    def target_callback(self, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if msg.header.frame_id != self.base_frame or not np.all(np.isfinite(p)):
            self.get_logger().warning(f'Ignoring invalid target; expected finite pose in {self.base_frame}')
            return
        forearm_dir = self._operator_forearm_direction(p)
        prev_q = self.q

        def cost(q):
            tip, elbow, wrist_roll = self._points(q)
            c = np.sum((tip - p) ** 2) + 1e-5 * np.sum((q - prev_q) ** 2)
            if forearm_dir is not None:
                forearm = wrist_roll - elbow
                c += self.elbow_weight * np.sum((forearm / (np.linalg.norm(forearm) + 1e-9)
                                                 - forearm_dir) ** 2)
            return c

        result = minimize(cost, prev_q, method='L-BFGS-B', bounds=self.bounds,
                          options={'maxiter': 80, 'ftol': 1e-7})
        if np.all(np.isfinite(result.x)):
            self.q = np.clip(result.x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            for name, value in zip(self.chain_names, self.q):
                self.all_joint_positions[self.all_joint_names.index(name)] = float(value)

    def elbow_callback(self, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if msg.header.frame_id == self.base_frame and np.all(np.isfinite(p)):
            self.elbow_target = (p, self.get_clock().now().nanoseconds * 1e-9)

    def _operator_forearm_direction(self, wrist):
        """Unit vector elbow -> wrist of the operator, or None if there is no usable elbow."""
        if self.elbow_weight <= 0.0 or self.elbow_target is None:
            return None
        elbow, received = self.elbow_target
        if self.get_clock().now().nanoseconds * 1e-9 - received > self.elbow_timeout:
            return None  # stale: the elbow was lost, fall back to wrist-only
        d = wrist - elbow
        norm = np.linalg.norm(d)
        return d / norm if norm > 0.02 else None  # <2 cm apart: direction is just noise

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
