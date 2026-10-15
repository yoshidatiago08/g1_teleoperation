"""A made-up hand for trying the whole pipeline without a camera or MediaPipe.

Builds 21 MediaPipe-style landmarks (metric, camera axes) for a given finger curl and pose, and projects
them like a camera at distance z. `SyntheticHandSource` animates it: the hand drifts in a circle, turns a
little, and the fingers open and close one after another.
"""
import time

import numpy as np

from dex_hand.frames import rotation_from_vec

SEGMENTS = {'index': (0.040, 0.025, 0.020), 'middle': (0.045, 0.028, 0.022), 'ring': (0.042, 0.026, 0.021),
            'pinky': (0.034, 0.020, 0.018)}
MCP_X = 0.090
# Palm facing the camera, fingers up in the image (hand frame columns in camera axes)
FACING_CAMERA = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def make_hand(side, curls, rotation=FACING_CAMERA):
    """21x3 landmarks (m, camera axes, wrist at the origin). curls: thumb, index, middle, ring, pinky in 0..1."""
    index_side = -1.0 if side == 'right' else 1.0           # y of the index finger in the hand frame
    h = np.zeros((21, 3))
    for k, (finger, y) in enumerate({'index': 0.030, 'middle': 0.010, 'ring': -0.010, 'pinky': -0.030}.items()):
        y *= index_side
        base = 5 + 4 * k
        pos, angle = np.array([MCP_X, y, 0.0]), 0.0
        h[base] = pos
        for i, length in enumerate(SEGMENTS[finger]):
            angle += curls[1 + k] * (1.35 if i == 0 else 1.1)      # each joint bends towards the palm (+z)
            pos = pos + length * np.array([np.cos(angle), 0.0, np.sin(angle)])
            h[base + 1 + i] = pos
    pos, angle = np.array([0.020, index_side * 0.025, 0.0]), 0.0     # thumb: starts beside the palm, bends in and across
    h[1] = pos
    for i, length in enumerate((0.040, 0.032, 0.027)):
        direction = np.array([np.cos(0.7 + 0.5 * curls[0] * i), index_side * np.sin(0.7) * (1 - 0.6 * curls[0]),
                              np.sin(1.0 * curls[0] * (i + 1) * 0.6)])
        pos = pos + length * direction / np.linalg.norm(direction)
        h[2 + i] = pos
    return h @ np.asarray(rotation).T


class SyntheticHandSource:
    """Same interface as sources.CameraHandSource: read() -> Observation or None."""

    def __init__(self, side, focal_px=600.0, size=(640, 480), fps=30.0):
        from dex_hand.sources import Observation
        self._obs = Observation
        self.side, self.focal_px = side, focal_px
        self.cx, self.cy = size[0] / 2.0, size[1] / 2.0
        self.fps, self._t0 = fps, time.monotonic()
        self.stretched = False            # True: the calibration pose (flat open hand, palm to the camera) instead of the animation

    def read(self):
        time.sleep(1.0 / self.fps)
        t = time.monotonic() - self._t0
        curls = [float(np.clip(0.5 + 0.7 * np.sin(0.9 * t + 0.6 * k), 0.0, 1.0)) for k in range(5)]
        rotation = rotation_from_vec([0.0, 0.0, 0.4 * np.sin(0.5 * t)]) @ FACING_CAMERA
        if self.stretched:
            curls, rotation = [0.0] * 5, FACING_CAMERA
        world = make_hand(self.side, curls, rotation)
        wrist = np.array([0.35 * np.sin(0.4 * t), 0.20 * np.sin(0.8 * t), 1.6 + 0.25 * np.sin(0.3 * t)])
        points = world + wrist
        pixels = np.column_stack([self.cx + self.focal_px * points[:, 0] / points[:, 2],
                                  self.cy + self.focal_px * points[:, 1] / points[:, 2]])
        return self._obs(world=world, pixels=pixels, depth=float(wrist[2]), image=None)

    def close(self):
        pass
