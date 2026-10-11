"""MediaPipe wrappers (the only place that imports mediapipe), behind two tiny interfaces:

    pose.process(rgb)                          -> PoseResult | None
    hands.detect(rgb, center_px, hand_px)      -> (world 21x3, image points 21x2) | None

so tests and the benchmark can plug in synthetic estimators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe Pose landmark indices
SHOULDER = {'right': 12, 'left': 11}
ELBOW = {'right': 14, 'left': 13}
WRIST = {'right': 16, 'left': 15}
# wrist, pinky, index, thumb: their mean locates the hand for the crop
HAND_GUIDE = {'right': (16, 18, 20, 22), 'left': (15, 17, 19, 21)}
HAND_LENGTH = 0.19   # m, wrist to the tip of the middle finger (sizes the crop)


@dataclass
class PoseResult:
    uv: np.ndarray        # 33x2 pixels
    vis: np.ndarray       # 33


class MediaPipePose:
    def __init__(self, complexity: int = 1, min_confidence: float = 0.5, smooth: bool = True):
        import mediapipe as mp
        # smooth=True is MediaPipe's own landmark smoothing (what the previous pipeline used). Landmarks
        # that jitter by several pixels would otherwise move the hand crops and every 3D point.
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=complexity, smooth_landmarks=smooth,
            enable_segmentation=False, min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence)

    def process(self, rgb: np.ndarray) -> Optional[PoseResult]:
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        lm = res.pose_landmarks.landmark
        uv = np.array([[p.x * w, p.y * h] for p in lm])
        vis = np.array([p.visibility for p in lm])
        return PoseResult(uv, vis)

    def close(self):
        self.pose.close()


class MediaPipeHands:
    """MediaPipe Hands on a crop around the wrist (the hand is only ~30 px wide at 2 m)."""

    CROP = 256

    def __init__(self, crop_margin: float = 2.2, min_confidence: float = 0.5):
        import mediapipe as mp
        self.crop_margin = crop_margin
        # static_image_mode: the crop moves from frame to frame, so the model's own tracking
        # between calls would not work.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, model_complexity=1,
            min_detection_confidence=min_confidence)

    def detect(self, rgb: np.ndarray, center_px, hand_px: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        h, w = rgb.shape[:2]
        side = int(np.clip(self.crop_margin * hand_px, 64, min(h, w)))
        u0 = int(np.clip(center_px[0] - side // 2, 0, w - side))
        v0 = int(np.clip(center_px[1] - side // 2, 0, h - side))
        crop = cv2.resize(rgb[v0:v0 + side, u0:u0 + side], (self.CROP, self.CROP),
                          interpolation=cv2.INTER_CUBIC)
        res = self.hands.process(crop)
        if not res.multi_hand_world_landmarks:
            return None
        world = np.array([[p.x, p.y, p.z] for p in res.multi_hand_world_landmarks[0].landmark])
        image = np.array([[u0 + p.x * side, v0 + p.y * side]
                          for p in res.multi_hand_landmarks[0].landmark])
        return world, image

    def close(self):
        self.hands.close()
