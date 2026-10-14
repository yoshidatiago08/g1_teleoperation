"""VM side: turn what the notebook sent into an Observation (MediaPipe landmarks + wrist depth)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Observation:
    world: Optional[np.ndarray]      # 21x3 MediaPipe world landmarks (m, camera axes, hand-centred); None = no hand
    pixels: Optional[np.ndarray]     # 21x2 landmark pixels in the full image
    depth: Optional[float]           # wrist distance from a depth sensor (m); None = estimate it from the hand's size
    image: Optional[np.ndarray]      # BGR frame (for the preview); None for the synthetic source


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


def wrist_depth(depth_image, pixel, depth_scale, downscale=1):
    """Median depth (m) in a small window around the wrist; the depth image may be smaller than the colour
    image by `downscale`. None if too few valid pixels."""
    u, v = int(round(pixel[0] / downscale)), int(round(pixel[1] / downscale))
    h, w = depth_image.shape[:2]
    window = depth_image[max(v - 3, 0):min(v + 4, h), max(u - 3, 0):min(u + 4, w)]
    valid = window[window > 0]
    return float(np.median(valid)) * depth_scale if valid.size >= 10 else None
