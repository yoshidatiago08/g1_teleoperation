#!/usr/bin/env python3
"""Stand-in for the RealSense, so the streaming link can be tested without hardware.

Publishes the same three topics the RealSense driver does (color, aligned depth,
camera_info) at 640x480, 30 Hz, with a moving pattern so compression has real work to do.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class FakeCamera(Node):
    def __init__(self):
        super().__init__('fake_camera')
        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', 1)
        self.depth_pub = self.create_publisher(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', 1)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', 1)
        self.frame = 0
        self.create_timer(1.0 / 30.0, self.tick)

    def tick(self):
        h, w = 480, 640
        t = self.frame
        color = np.zeros((h, w, 3), np.uint8)
        color[:] = (40 + (t * 2) % 80, 60, 90)
        cv2.circle(color, (int(w / 2 + 200 * np.sin(t / 30)), int(h / 2 + 100 * np.cos(t / 20))),
                   60, (0, 255, 0), -1)
        cv2.putText(color, f'fake camera frame {t}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        depth = (1500 + 500 * np.sin(np.add.outer(np.arange(h), np.arange(w)) / 80.0 + t / 15)
                 ).astype(np.uint16)  # millimetres

        stamp = self.get_clock().now().to_msg()
        color_msg = self.bridge.cv2_to_imgmsg(color, 'bgr8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth, '16UC1')
        color_msg.header.stamp = depth_msg.header.stamp = stamp
        color_msg.header.frame_id = depth_msg.header.frame_id = 'camera_color_optical_frame'

        info = CameraInfo()
        info.header = color_msg.header
        info.width, info.height = w, h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [610.0, 0.0, 320.0, 0.0, 610.0, 240.0, 0.0, 0.0, 1.0]
        info.p = [610.0, 0.0, 320.0, 0.0, 0.0, 610.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)
        self.frame += 1


def main():
    rclpy.init()
    node = FakeCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
