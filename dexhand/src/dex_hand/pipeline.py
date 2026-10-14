"""One observation in, one robot hand command out. No ROS, so it runs in tests and in the node alike.

    landmarks ─► hand frame ─► wrist -> fingertip vectors ─► dex-retargeting ─► Inspire joint angles
        └──────► wrist pose (camera) ─► operator axes ─► MotionScaler (scale, clutch) ─► hand target
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dex_hand import frames


@dataclass
class Command:
    joints: dict                      # joint name -> angle (rad), every joint of the hand
    position: np.ndarray              # robot hand base link in the world frame (scaled)
    rotation: np.ndarray              # world <- base link
    ghost: Optional[tuple]            # (position, hand-frame rotation) at scale 1, from the same anchor
    skeleton: np.ndarray              # 21x3 landmarks, operator axes, relative to the wrist
    raw_position: np.ndarray          # wrist in operator axes (camera at the origin), calibrated
    raw_rotation: np.ndarray          # your hand's rotation in operator axes BEFORE the calibration (what calibrating measures)
    engaged: bool
    depth_from: str


class HandPipeline:
    def __init__(self, side, retargeter, scaler, focal_px, cx, cy):
        self.side, self.retargeter, self.scaler = side, retargeter, scaler
        self.focal_px, self.cx, self.cy = focal_px, cx, cy
        self.last = None                                # the last Command (held while the hand is lost)
        self.camera_to_operator = frames.CAMERA_TO_OPERATOR   # replaced by the IMU's gravity when the camera has one
        self.calibration = np.eye(3)                    # operator frame -> calibrated frame (see frames.calibration_from)
        self._t = None

    def set_camera_up(self, up_camera):
        """Level the operator frame with gravity: `up_camera` is 'up' in camera axes from the IMU."""
        axes = frames.gravity_aligned_axes(up_camera)
        if axes is not None:
            self.camera_to_operator = axes

    def set_calibration(self, rotation):
        """Install a calibration; the robot re-anchors on the next engaged frame."""
        self.calibration = np.asarray(rotation, dtype=float)
        self.scaler.release()

    def step(self, obs, clutch=False) -> Optional[Command]:
        """`obs`: sources.Observation (world/pixels None = no hand). `clutch`: True holds the robot hand still.
        Returns None until a hand has been seen once."""
        rotation = None if obs.world is None else frames.hand_rotation(obs.world, self.side)
        if rotation is None:                            # lost: hold, and re-anchor when the hand returns
            self.scaler.release()
            if self.last is not None:
                self.last.engaged, self.last.ghost = False, None
            return self.last

        joints = self.retargeter(frames.tip_vectors(obs.world, rotation))
        depth, depth_from = obs.depth, 'sensor'
        if depth is None:
            depth, depth_from = frames.depth_from_scale(obs.world, obs.pixels, self.focal_px), 'hand size'
        if depth is None:
            return self.last
        wrist = frames.pixel_to_camera(*obs.pixels[0], depth, self.focal_px, self.cx, self.cy)
        p_raw, r_raw = frames.to_operator(wrist, rotation, self.camera_to_operator)
        p_op, r_op = self.calibration @ p_raw, self.calibration @ r_raw
        now = time.monotonic()
        dt = None if self._t is None else min(now - self._t, 0.1)
        self._t = now

        position, hand_rotation = self.scaler.update(p_op, r_op, engaged=not clutch, dt=dt)
        skeleton = (obs.world - obs.world[0]) @ self.camera_to_operator.T @ self.calibration.T
        self.last = Command(joints=joints, position=position,
                            rotation=hand_rotation @ self.retargeter.r_bh.T,
                            ghost=self.scaler.ghost(p_op, r_op), skeleton=skeleton, raw_position=p_op, raw_rotation=r_raw,
                            engaged=self.scaler.engaged, depth_from=depth_from)
        return self.last
