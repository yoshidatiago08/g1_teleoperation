"""Where the hand comes from: a webcam or video file (`CameraHandSource`), an Intel RealSense (with depth),
or a made-up hand (`synthetic.SyntheticHandSource`). Every source's read() returns an Observation or None."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Observation:
    world: np.ndarray            # 21x3 MediaPipe world landmarks (m, camera axes, hand-centred)
    pixels: np.ndarray           # 21x2 landmark pixels in the full image
    depth: Optional[float]       # wrist distance from a depth sensor (m); None = estimate it from the hand's size
    image: Optional[np.ndarray]  # BGR frame for the debug window; None for the synthetic source


class HandDetector:
    """MediaPipe Hands, as in dex-retargeting's single_hand_detector: the camera is not mirrored, so a right
    hand is reported as 'Left'."""

    def __init__(self, side, min_confidence=0.6):
        import mediapipe as mp
        self.wanted = 'Left' if side == 'right' else 'Right'
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                              min_detection_confidence=min_confidence,
                                              min_tracking_confidence=min_confidence)

    def __call__(self, rgb):
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        for i, handed in enumerate(result.multi_handedness):
            if handed.classification[0].label == self.wanted:
                world = np.array([[p.x, p.y, p.z] for p in result.multi_hand_world_landmarks[i].landmark])
                pixels = np.array([[p.x * w, p.y * h] for p in result.multi_hand_landmarks[i].landmark])
                return world, pixels
        return None


class CameraHandSource:
    """A webcam (index or device path) or a video file, plus MediaPipe. Optionally a RealSense for depth."""

    def __init__(self, side, device='0', realsense=False, hfov_deg=70.0, size=(640, 480)):
        import cv2
        self.cv2, self.rs = cv2, None
        self.detector = HandDetector(side)
        if realsense:
            self._open_realsense(size)
        else:
            self.cap = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
            if not self.cap.isOpened():
                raise RuntimeError(f'cannot open the camera {device!r} (is /dev/video0 passed into the container?)')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.focal_px = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
            self.cx, self.cy = w / 2.0, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2.0

    def _open_realsense(self, size):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe, config = rs.pipeline(), rs.config()
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
        profile = self.pipe.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.focal_px, self.cx, self.cy = (k.fx + k.fy) / 2.0, k.ppx, k.ppy

    def _wrist_depth(self, depth_image, pixel):
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        h, w = depth_image.shape
        window = depth_image[max(v - 3, 0):min(v + 4, h), max(u - 3, 0):min(u + 4, w)]
        valid = window[window > 0]
        return float(np.median(valid)) * self.depth_scale if valid.size >= 10 else None

    def read(self):
        depth_image = None
        if self.rs is not None:
            frames = self.align.process(self.pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                return None
            bgr, depth_image = np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
        else:
            ok, bgr = self.cap.read()
            if not ok:
                return None
        found = self.detector(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
        if found is None:
            return Observation(world=None, pixels=None, depth=None, image=bgr)
        world, pixels = found
        depth = self._wrist_depth(depth_image, pixels[0]) if depth_image is not None else None
        return Observation(world=world, pixels=pixels, depth=depth, image=bgr)

    def close(self):
        if self.rs is not None:
            self.pipe.stop()
        else:
            self.cap.release()
