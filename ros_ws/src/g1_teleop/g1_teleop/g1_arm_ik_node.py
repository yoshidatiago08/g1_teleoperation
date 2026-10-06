#!/usr/bin/env python3
"""RViz visualization IK for the G1 arms (one independent problem per arm).

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

from g1_teleop.hand_pose import hand_rotation, quaternion_to_matrix
from g1_teleop.urdf_info import joint_origin as _origin

# Knuckle positions of the Inspire hand in the wrist_yaw_link frame (from its URDF; both hands have
# the same positions, they only differ in which side the thumb and palm are). Landmark indices as
# in MediaPipe Hands: 0 wrist, 5 index, 9 middle, 17 pinky knuckle.
_HAND = np.zeros((21, 3))
_HAND[0], _HAND[5], _HAND[9], _HAND[17] = ((0.042, 0, 0), (0.178, 0, 0.032),
                                          (0.179, 0, 0.013), (0.177, 0, -0.025))


# Inspire hand (DFQ URDF): the joint that each of the six actuator values drives, per side. The
# other finger joints follow these through <mimic> tags in the URDF. An actuator value of 0 is
# open and 1 is closed; the joint goes from 0 to its upper limit.
HAND_JOINT_PREFIX = {'right': 'R_', 'left': 'L_'}
HAND_ACTUATOR_JOINT = {
    'pinky': 'pinky_proximal_joint', 'ring': 'ring_proximal_joint',
    'middle': 'middle_proximal_joint', 'index': 'index_proximal_joint',
    'thumb_bend': 'thumb_proximal_pitch_joint', 'thumb_rotation': 'thumb_proximal_yaw_joint'}


def hand_in_link(side):
    """Rotation from the operator's hand frame to the wrist link frame.

    It is the hand frame (x fingers, z palm normal) of the robot's own hand, built with the same
    function that builds the operator's, so the two definitions cannot disagree.
    """
    return hand_rotation(_HAND, side)


class ArmIK:
    """Kinematic chain, current solution and elbow target of one arm."""

    def __init__(self, side, chain, elbow_weight):
        self.side = side
        self.hand_in_link = hand_in_link(side)
        self.orientation_weight = 0.0
        self.last_solve_time = None
        self.chain = chain
        self.names = [j['name'] for j in chain]
        self.q = np.zeros(len(chain))
        self.bounds = [(j['lower'], j['upper']) for j in chain]
        self.elbow_target = None  # (xyz, receive time in s)
        self.elbow_weight = elbow_weight
        elbow_joint, wrist_roll_joint = f'{side}_elbow_joint', f'{side}_wrist_roll_joint'
        self.elbow_idx = self.names.index(elbow_joint) if elbow_joint in self.names else None
        self.wrist_roll_idx = (self.names.index(wrist_roll_joint)
                               if wrist_roll_joint in self.names else None)

    def points(self, q):
        """Wrist position, elbow joint, wrist-roll joint and wrist rotation, in the base frame."""
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
        return t[:3, 3].copy(), elbow, wrist_roll, t[:3, :3].copy()


class G1ArmIK(Node):
    def __init__(self):
        super().__init__('g1_arm_ik_node')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('base_frame', 'torso_link')
        self.declare_parameter('joint_state_topic', '/g1_visualization/joint_states')
        # One IK problem per arm. Targets arrive on /<side>/wrist_pose and /<side>/elbow_pose.
        self.declare_parameter('sides', ['right', 'left'])
        # Weight of the forearm-direction term against the wrist position error (in m^2).
        # 0 disables it and gives wrist-only IK.
        self.declare_parameter('elbow_weight', 0.01)
        # Ignore an elbow target older than this (s); the solver falls back to wrist-only.
        self.declare_parameter('elbow_timeout', 0.3)
        self.elbow_weight = float(self.get_parameter('elbow_weight').value)
        self.elbow_timeout = float(self.get_parameter('elbow_timeout').value)
        # Weight of the wrist-orientation error (squared Frobenius norm of R - R_target, about
        # angle^2 for small errors) against the wrist position error (in m^2). 0 disables it.
        self.declare_parameter('orientation_weight', 0.01)
        self.orientation_weight = float(self.get_parameter('orientation_weight').value)
        # Joint speed limits (rad/s). A bad reading must never be able to throw an arm across the
        # workspace in one step: the joints move toward the solution at most this fast. 0 = no limit.
        self.declare_parameter('max_joint_speed', 8.0)
        self.declare_parameter('max_waist_speed', 3.0)
        self.max_joint_speed = float(self.get_parameter('max_joint_speed').value)
        self.max_waist_speed = float(self.get_parameter('max_waist_speed').value)
        path = self.get_parameter('urdf_path').value
        if not path:
            raise RuntimeError('urdf_path must point to the selected G1 URDF')
        self.base_frame = self.get_parameter('base_frame').value
        self.all_joint_names, self.all_joint_positions = [], []
        self.arms = {}
        for side in self.get_parameter('sides').value:
            arm = ArmIK(side, self._load_chain(path, f'{side}_wrist_yaw_link', side),
                        self.elbow_weight)
            if self.elbow_weight > 0.0 and (arm.elbow_idx is None or arm.wrist_roll_idx is None):
                self.get_logger().warning(f'{side}: elbow/wrist-roll joint not in the chain: '
                                          'elbow term disabled')
                arm.elbow_weight = 0.0
            arm.orientation_weight = self.orientation_weight
            self.arms[side] = arm
            self.create_subscription(PoseStamped, f'/{side}/wrist_pose',
                                     lambda msg, a=arm: self.target_callback(a, msg), 10)
            self.create_subscription(PoseStamped, f'/{side}/elbow_pose',
                                     lambda msg, a=arm: self.elbow_callback(a, msg), 10)
            self.get_logger().info(f'{side} arm chain: {arm.names}; elbow weight '
                                   f'{arm.elbow_weight}, orientation weight {arm.orientation_weight}; '
                                   'visualization only')
        self._load_all_joints(path)
        self._load_hands(path)
        self._load_waist(path)
        self.publisher = self.create_publisher(
            JointState, self.get_parameter('joint_state_topic').value, 10)
        self.last_publish_time = None
        self.create_timer(1.0 / 30.0, self.publish_state)

    def _load_all_joints(self, path):
        # Every movable joint in the URDF is published (zeros for the ones no arm drives), so
        # fixed-base/waist links are well-defined in RViz.
        for joint in ET.parse(path).getroot().findall('joint'):
            name = joint.get('name')
            if joint.get('type') in ('revolute', 'continuous', 'prismatic') \
                    and name not in self.all_joint_names:
                self.all_joint_names.append(name)
                self.all_joint_positions.append(0.0)

    def _load_waist(self, path):
        """Waist joints driven by /waist_state (the operator's torso relative to the pelvis)."""
        joints = {j.get('name'): j for j in ET.parse(path).getroot().findall('joint')}
        self.waist_limits = {}
        for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
            if name in joints:
                limit = joints[name].find('limit')
                self.waist_limits[name] = (float(limit.get('lower')), float(limit.get('upper')))
        self.waist_goal = {name: 0.0 for name in self.waist_limits}
        self.waist_goal_time = None
        if self.waist_limits:
            self.create_subscription(JointState, '/waist_state', self.waist_callback, 10)

    def waist_callback(self, msg):
        for name, value in zip(msg.name, msg.position):
            if name in self.waist_limits and np.isfinite(value):
                lo, hi = self.waist_limits[name]
                self.waist_goal[name] = float(np.clip(value, lo, hi))
        self.waist_goal_time = self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _step_toward(current, goal, max_step):
        """Move `current` toward `goal` by at most `max_step` (per element)."""
        return current + np.clip(goal - current, -max_step, max_step)

    def _load_hands(self, path):
        """Finger joints driven by /<side>/hand_state, and the mimic joints that follow them."""
        joints = {j.get('name'): j for j in ET.parse(path).getroot().findall('joint')}
        self.mimic = {}  # joint -> (master joint, multiplier, offset)
        for name, joint in joints.items():
            mimic = joint.find('mimic')
            if mimic is not None:
                self.mimic[name] = (mimic.get('joint'), float(mimic.get('multiplier', 1.0)),
                                    float(mimic.get('offset', 0.0)))
        self.hand_joints = {}  # side -> {actuator name: (joint name, upper limit)}
        for side in self.arms:
            table = {}
            for actuator, joint_name in HAND_ACTUATOR_JOINT.items():
                name = HAND_JOINT_PREFIX[side] + joint_name
                if name in joints:
                    table[f'{side}_{actuator}'] = (name, float(joints[name].find('limit').get('upper')))
            if table:
                self.hand_joints[side] = table
                self.create_subscription(JointState, f'/{side}/hand_state',
                                         lambda msg, s=side: self.hand_callback(s, msg), 10)
                self.get_logger().info(f'{side} hand: driving {[v[0] for v in table.values()]}')
            else:
                self.get_logger().info(f'{side} hand: no Inspire hand in this URDF, fingers ignored')

    def hand_callback(self, side, msg):
        """Actuator values (0 open ... 1 closed) -> finger joint angles, including mimic joints."""
        table = self.hand_joints[side]
        for name, value in zip(msg.name, msg.position):
            if name in table and np.isfinite(value):
                joint, upper = table[name]
                self.all_joint_positions[self.all_joint_names.index(joint)] = \
                    float(np.clip(value, 0.0, 1.0)) * upper
        for joint, (master, multiplier, offset) in self.mimic.items():
            if joint in self.all_joint_names and master in self.all_joint_names:
                self.all_joint_positions[self.all_joint_names.index(joint)] = (
                    multiplier * self.all_joint_positions[self.all_joint_names.index(master)] + offset)

    def _load_chain(self, path, tip_link, side):
        root = ET.parse(path).getroot()
        joints = root.findall('joint')
        child_map = {j.find('child').get('link'): j for j in joints if j.find('child') is not None}
        # Walk back from the wrist and keep the movable arm joints.
        cur, reverse = tip_link, []
        while cur != self.base_frame:
            joint = child_map.get(cur)
            if joint is None:
                raise RuntimeError(f'No URDF chain from {self.base_frame} to {tip_link} (stopped at {cur})')
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
            raise RuntimeError(f'Expected 7 movable {side}-arm joints, found {len(chain)}: {[j["name"] for j in chain]}')
        return chain

    def target_callback(self, arm, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if msg.header.frame_id != self.base_frame or not np.all(np.isfinite(p)):
            self.get_logger().warning(f'Ignoring invalid target; expected finite pose in {self.base_frame}')
            return
        forearm_dir = self._operator_forearm_direction(arm, p)
        r_target = self._orientation_target(arm, msg)
        prev_q = arm.q

        def cost(q):
            tip, elbow, wrist_roll, r_tip = arm.points(q)
            c = np.sum((tip - p) ** 2) + 1e-5 * np.sum((q - prev_q) ** 2)
            if r_target is not None:
                c += arm.orientation_weight * np.sum((r_tip - r_target) ** 2)
            if forearm_dir is not None:
                forearm = wrist_roll - elbow
                c += arm.elbow_weight * np.sum((forearm / (np.linalg.norm(forearm) + 1e-9)
                                                - forearm_dir) ** 2)
            return c

        result = minimize(cost, prev_q, method='L-BFGS-B', bounds=arm.bounds,
                          options={'maxiter': 80, 'ftol': 1e-7})
        if np.all(np.isfinite(result.x)):
            q_new = np.clip(result.x, [b[0] for b in arm.bounds], [b[1] for b in arm.bounds])
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.max_joint_speed > 0.0 and arm.last_solve_time is not None:
                dt = float(np.clip(now - arm.last_solve_time, 1e-3, 0.2))
                q_new = self._step_toward(prev_q, q_new, self.max_joint_speed * dt)
            arm.last_solve_time = now
            arm.q = q_new
            for name, value in zip(arm.names, arm.q):
                self.all_joint_positions[self.all_joint_names.index(name)] = float(value)

    def _orientation_target(self, arm, msg):
        """Wrist-link rotation wanted by the operator's hand, or None (no hand reading).

        The pose message carries the operator's hand frame; an all-zero quaternion means that
        the perception has no hand orientation.
        """
        if arm.orientation_weight <= 0.0:
            return None
        q = msg.pose.orientation
        r_hand = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return None if r_hand is None else r_hand @ arm.hand_in_link.T

    def elbow_callback(self, arm, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if msg.header.frame_id == self.base_frame and np.all(np.isfinite(p)):
            arm.elbow_target = (p, self.get_clock().now().nanoseconds * 1e-9)

    def _operator_forearm_direction(self, arm, wrist):
        """Unit vector elbow -> wrist of the operator, or None if there is no usable elbow."""
        if arm.elbow_weight <= 0.0 or arm.elbow_target is None:
            return None
        elbow, received = arm.elbow_target
        if self.get_clock().now().nanoseconds * 1e-9 - received > self.elbow_timeout:
            return None  # stale: the elbow was lost, fall back to wrist-only
        d = wrist - elbow
        norm = np.linalg.norm(d)
        return d / norm if norm > 0.02 else None  # <2 cm apart: direction is just noise

    def publish_state(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.last_publish_time is None else float(np.clip(now - self.last_publish_time, 0.0, 0.2))
        self.last_publish_time = now
        # The waist eases toward the operator's torso rotation, or back to zero if it is no longer
        # being measured.
        measured = self.waist_goal_time is not None and now - self.waist_goal_time < 1.0
        for name, (lo, hi) in self.waist_limits.items():
            index = self.all_joint_names.index(name)
            goal = self.waist_goal[name] if measured else 0.0
            step = self.max_waist_speed * dt if self.max_waist_speed > 0.0 else abs(goal)
            self.all_joint_positions[index] = float(
                self._step_toward(self.all_joint_positions[index], goal, step))
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
