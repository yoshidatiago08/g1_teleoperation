"""Notebook side: the camera, and nothing heavy. A frame is JPEG-compressed (and the RealSense depth shrunk to
16-bit PNG at half size) and sent to the VM, which does the tracking."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

DEPTH_DOWNSCALE = 2


@dataclass
class Frame:
    bgr: np.ndarray
    depth: Optional[np.ndarray]       # uint16, full size, or None
    focal_px: float
    cx: float
    cy: float
    depth_scale: float = 0.001        # m per depth unit


class Webcam:
    """A webcam (index or device path) or a video file. No depth: the VM estimates distance from the hand's size."""

    def __init__(self, device='0', hfov_deg=70.0, size=(640, 480)):
        self.cap = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open the camera {device!r} (is /dev/video0 passed into the container?)')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        w, h = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.focal_px = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.cx, self.cy = w / 2.0, h / 2.0

    def read(self):
        ok, bgr = self.cap.read()
        return Frame(bgr, None, self.focal_px, self.cx, self.cy) if ok else None

    def close(self):
        self.cap.release()


class RealSense:
    def __init__(self, size=(640, 480)):
        import pyrealsense2 as rs
        self.pipe, config = rs.pipeline(), rs.config()
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
        profile = self.pipe.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.focal_px, self.cx, self.cy = (k.fx + k.fy) / 2.0, k.ppx, k.ppy

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        return Frame(np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), self.focal_px,
                     self.cx, self.cy, self.depth_scale)

    def close(self):
        self.pipe.stop()


def make_capture(kind, device='0', hfov_deg=70.0):
    if kind == 'webcam':
        return Webcam(device, hfov_deg)
    if kind == 'realsense':
        return RealSense()
    return None                                   # 'none': the VM drives itself (synthetic hand)


def encode(frame, t, quality=80):
    """The message sent to the VM."""
    message = {'t': t, 'color': cv2.imencode('.jpg', frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes(),
               'fx': frame.focal_px, 'cx': frame.cx, 'cy': frame.cy}
    if frame.depth is not None:
        small = frame.depth[::DEPTH_DOWNSCALE, ::DEPTH_DOWNSCALE]
        message.update(depth=cv2.imencode('.png', small, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes(),
                       depth_scale=frame.depth_scale, depth_downscale=DEPTH_DOWNSCALE)
    return message


def decode(message):
    """VM side: (bgr, depth16 or None) from an encoded frame."""
    bgr = cv2.imdecode(np.frombuffer(message['color'], np.uint8), cv2.IMREAD_COLOR)
    depth = None
    if 'depth' in message:
        depth = cv2.imdecode(np.frombuffer(message['depth'], np.uint8), cv2.IMREAD_UNCHANGED)
    return bgr, depth
