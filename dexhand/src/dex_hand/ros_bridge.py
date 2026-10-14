"""Notebook side, optional: show the VM's state in RViz. Needs ROS 2 (rclpy, robot_state_publisher, rviz2).

Starts robot_state_publisher and RViz as child processes and publishes what the VM sends:
    /joint_states, TF world -> <P>_hand_base_link (the simulated hand), hand_target (the scaled goal),
    operator_hand (your hand at scale 1), and markers for the table, cylinder, target disc and your tracked skeleton.
"""
import os
import subprocess
import tempfile
from pathlib import Path

from dex_hand import paths
from dex_hand.mujoco_scene import CYL_HEIGHT, CYL_RADIUS, TABLE_TOP, TARGET_XY

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
         (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


class RosBridge:
    def __init__(self, side, rviz=True):
        import rclpy
        from geometry_msgs.msg import Point, TransformStamped
        from sensor_msgs.msg import JointState
        from tf2_ros import TransformBroadcaster
        from visualization_msgs.msg import Marker, MarkerArray
        self.Point, self.TransformStamped, self.JointState = Point, TransformStamped, JointState
        self.Marker, self.MarkerArray = Marker, MarkerArray
        rclpy.init()
        self.rclpy, self.node = rclpy, rclpy.create_node('dexhand_bridge')
        self.joint_pub = self.node.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.node.create_publisher(MarkerArray, '/dexhand_markers', 1)
        self.tf = TransformBroadcaster(self.node)
        self.base_frame = ('R' if side == 'right' else 'L') + '_hand_base_link'
        self.procs = self._start(side, rviz)

    def _start(self, side, rviz):
        import yaml
        urdf = paths.urdf(side).read_text(encoding='utf-8').replace('package://dex_hand/urdf/meshes/',
                                                                       f'file://{paths.URDF_DIR}/meshes/')
        params = Path(tempfile.gettempdir()) / 'dexhand_rsp.yaml'
        params.write_text(yaml.safe_dump({'/**': {'ros__parameters': {'robot_description': urdf}}}))
        # OpenCV (opencv-python) points QT_QPA_PLATFORM_PLUGIN_PATH at its own Qt; RViz would load those plugins and fail
        env = {k: v for k, v in os.environ.items() if k not in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR')}
        procs = [subprocess.Popen(['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
                                   '--ros-args', '--params-file', str(params)], env=env)]
        if rviz:
            procs.append(subprocess.Popen(['rviz2', '-d', str(paths.ASSETS / 'rviz' / 'dex_hand.rviz')], env=env))
        return procs

    def _transform(self, stamp, child, position, quat_wxyz):
        t = self.TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'world', child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, position)
        w, x, y, z = map(float, quat_wxyz)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
        return t

    def _marker(self, stamp, ident, kind, position, scale, rgba, quat_wxyz=(1, 0, 0, 0)):
        m = self.Marker()
        m.header.stamp, m.header.frame_id = stamp, 'world'
        m.ns, m.id, m.type, m.action = 'scene', ident, kind, self.Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, position)
        w, x, y, z = map(float, quat_wxyz)               # ROS aborts the process on an int where it wants a float
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = w, x, y, z
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def publish(self, state):
        stamp = self.node.get_clock().now().to_msg()
        scene = state.get('scene') or {}
        if state.get('joints'):
            joints = self.JointState()
            joints.header.stamp = stamp
            joints.name, joints.position = list(state['joints']), list(state['joints'].values())
            self.joint_pub.publish(joints)
        transforms = []
        if scene:
            transforms.append(self._transform(stamp, self.base_frame, scene['hand_pos'], scene['hand_quat']))
        if state.get('goal_pos'):
            transforms.append(self._transform(stamp, 'hand_target', state['goal_pos'], state['goal_quat']))
        ghost = state.get('ghost')
        if ghost:
            transforms.append(self._transform(stamp, 'operator_hand', ghost['pos'], ghost['quat']))
        if transforms:
            self.tf.sendTransform(transforms)

        M, markers = self.Marker, self.MarkerArray()
        markers.markers.append(self._marker(stamp, 0, M.CUBE, (0.58, 0.0, TABLE_TOP - 0.02), (0.56, 1.10, 0.04),
                                            (0.55, 0.42, 0.30, 1.0)))
        markers.markers.append(self._marker(stamp, 1, M.CYLINDER, (*TARGET_XY, TABLE_TOP + 0.001), (0.10, 0.10, 0.002),
                                            (0.2, 0.85, 0.3, 1.0)))
        if scene:
            markers.markers.append(self._marker(stamp, 2, M.CYLINDER, scene['cyl_pos'],
                                                (2 * CYL_RADIUS, 2 * CYL_RADIUS, CYL_HEIGHT), (0.9, 0.25, 0.2, 1.0),
                                                scene['cyl_quat']))
        skeleton = self._skeleton(stamp, ghost, state.get('skeleton'))
        if skeleton is not None:
            markers.markers.append(skeleton)
        self.marker_pub.publish(markers)

    def _skeleton(self, stamp, ghost, skeleton):
        m = self.Marker()
        m.header.stamp, m.header.frame_id = stamp, 'world'
        m.ns, m.id = 'skeleton', 3
        if not ghost or not skeleton:
            m.action = self.Marker.DELETE
            return m
        m.type, m.action = self.Marker.LINE_LIST, self.Marker.ADD
        m.scale.x = 0.004
        m.color.g = m.color.b = 0.9
        m.color.r, m.color.a = 0.1, 1.0
        m.pose.orientation.w = 1.0
        origin = ghost['pos']
        for a, b in EDGES:
            for i in (a, b):
                m.points.append(self.Point(x=origin[0] + skeleton[i][0], y=origin[1] + skeleton[i][1],
                                           z=origin[2] + skeleton[i][2]))
        return m

    def close(self):
        for proc in self.procs:
            proc.terminate()
        if self.rclpy.ok():
            self.rclpy.shutdown()
