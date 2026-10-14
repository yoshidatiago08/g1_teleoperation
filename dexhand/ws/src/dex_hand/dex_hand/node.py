"""ROS 2 node: camera -> dex-retargeting -> JointState + TF + markers for RViz.

Publishes
    /joint_states                      the Inspire hand's joint angles (robot_state_publisher draws the hand)
    TF  world -> <P>_hand_base_link    the hand at the SCALED target (the one the robot would follow)
    TF  world -> operator_hand         your hand at scale 1 from the same anchor, to compare against
    /operator_skeleton                 your tracked landmarks (MarkerArray), at the unscaled position
Controls (the camera window, or the service): space = clutch (hold the robot hand still, reposition your
hand, release), + / - = motion scale x1.25 / x0.8. Parameters can be changed live: motion_scale, rotation_scale.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from dex_hand import frames
from dex_hand.motion import MotionScaler
from dex_hand.pipeline import HandPipeline
from dex_hand.retarget import FingerRetargeter

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
         (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
HOME_ROTATION = np.diag([1.0, -1.0, -1.0])        # fingers forward, palm down


class DexHandNode(Node):
    def __init__(self):
        super().__init__('dex_hand')
        declare = self.declare_parameter
        self.side = declare('hand', 'right').value
        source_kind = declare('source', 'webcam').value                      # webcam | realsense | synthetic
        device = declare('device', '0').value                                # camera index, device path or video file
        hfov = declare('hfov_deg', 70.0).value                               # webcam horizontal field of view
        scale = declare('motion_scale', 0.3).value
        rotation_scale = declare('rotation_scale', 1.0).value
        finger_scale = declare('finger_scale', 1.15).value
        default_home = [0.35, -0.20, 0.0] if self.side == 'right' else [0.35, 0.20, 0.0]
        home = declare('home_position', default_home).value
        self.show_window = declare('show_window', True).value and bool(os.environ.get('DISPLAY'))
        self.world_frame = declare('world_frame', 'world').value

        share = Path(get_package_share_directory('dex_hand'))
        self.prefix = 'R' if self.side == 'right' else 'L'
        retargeter = FingerRetargeter(share / 'urdf', share / 'config' / f'inspire_hand_{self.side}.yml',
                                      self.side, scaling_factor=finger_scale)
        self.source = self._open_source(source_kind, device, hfov)
        self.scaler = MotionScaler(home, HOME_ROTATION, scale, rotation_scale)
        self.pipeline = HandPipeline(self.side, retargeter, self.scaler, self.source.focal_px,
                                     self.source.cx, self.source.cy)
        self.clutch = False

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/operator_skeleton', 1)
        self.tf = TransformBroadcaster(self)
        self.create_service(Trigger, '~/toggle_clutch', self._toggle_clutch_srv)
        self.add_on_set_parameters_callback(self._on_parameters)
        self.timer = self.create_timer(0.001, self._tick)          # the camera read paces the loop
        self.get_logger().info(f'{self.side} hand, source {source_kind}, motion scale {scale}'
                               f'{"" if self.show_window else ", no window (no DISPLAY)"}')

    def _open_source(self, kind, device, hfov):
        if kind == 'synthetic':
            from dex_hand.synthetic import SyntheticHandSource
            return SyntheticHandSource(self.side)
        from dex_hand.sources import CameraHandSource
        return CameraHandSource(self.side, device=device, realsense=(kind == 'realsense'), hfov_deg=hfov)

    # -- controls -------------------------------------------------------------------------------------------
    def _toggle_clutch(self):
        self.clutch = not self.clutch
        self.get_logger().info('CLUTCH: robot hand held' if self.clutch else 'engaged')

    def _toggle_clutch_srv(self, _, response):
        self._toggle_clutch()
        response.success, response.message = True, 'held' if self.clutch else 'engaged'
        return response

    def _on_parameters(self, params):
        for p in params:
            if p.name == 'motion_scale':
                self.scaler.set_scale(scale=p.value)
            elif p.name == 'rotation_scale':
                self.scaler.set_scale(rotation_scale=p.value)
        return SetParametersResult(successful=True)

    # -- loop -----------------------------------------------------------------------------------------------
    def _tick(self):
        obs = self.source.read()
        if obs is None:
            return
        command = self.pipeline.step(obs, self.clutch)
        if command is not None:
            self._publish(command)
        if self.show_window and obs.image is not None:
            self._draw(obs, command)

    def _publish(self, c):
        now = self.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = now
        joints.name, joints.position = list(c.joints), list(c.joints.values())
        self.joint_pub.publish(joints)

        transforms = [self._transform(now, f'{self.prefix}_hand_base_link', c.position, c.rotation)]
        markers = MarkerArray()
        if c.ghost is not None:
            gp, gr = c.ghost
            transforms.append(self._transform(now, 'operator_hand', gp, gr))
            markers.markers.append(self._skeleton_marker(now, gp, gr, c.skeleton))
        else:
            delete = Marker()
            delete.action = Marker.DELETEALL
            markers.markers.append(delete)
        self.tf.sendTransform(transforms)
        self.marker_pub.publish(markers)

    def _transform(self, stamp, child, position, rotation):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.world_frame, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, position)
        q = frames.quaternion(rotation)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        return t

    def _skeleton_marker(self, stamp, position, hand_rotation, skeleton):
        """The tracked landmarks drawn at the unscaled hand position. `skeleton` is relative to the wrist in
        operator axes already, so only the position is added."""
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, self.world_frame
        m.ns, m.id, m.type, m.action = 'skeleton', 0, Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.004
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.9, 1.0
        m.pose.orientation.w = 1.0
        for a, b in EDGES:
            for i in (a, b):
                m.points.append(Point(x=float(position[0] + skeleton[i][0]), y=float(position[1] + skeleton[i][1]),
                                      z=float(position[2] + skeleton[i][2])))
        return m

    def _draw(self, obs, c):
        image = obs.image.copy()
        if obs.pixels is not None:
            for a, b in EDGES:
                cv2.line(image, tuple(obs.pixels[a].astype(int)), tuple(obs.pixels[b].astype(int)), (0, 220, 0), 2)
        if obs.world is None:
            status = 'NO HAND'
        elif self.clutch:
            status = 'CLUTCH (robot held)'
        else:
            status = 'TRACKING'
        text = f'{status}   scale {self.scaler.scale:.2f}   [space] clutch  [+/-] scale'
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('dex_hand', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self._toggle_clutch()
        elif key in (ord('+'), ord('=')):
            self.set_parameters([rclpy.parameter.Parameter('motion_scale', value=self.scaler.scale * 1.25)])
        elif key in (ord('-'), ord('_')):
            self.set_parameters([rclpy.parameter.Parameter('motion_scale', value=self.scaler.scale * 0.8)])


def main():
    rclpy.init()
    node = DexHandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.source.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
