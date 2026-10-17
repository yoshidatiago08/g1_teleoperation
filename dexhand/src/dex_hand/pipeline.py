"""One observation in, one robot hand command out. No ROS, so it runs in tests and in the node alike.

    landmarks ─► hand frame ─► wrist -> fingertip vectors ─► dex-retargeting ─► Inspire joint angles
        └──────► wrist pose (camera) ─► operator axes ─► MotionScaler (scale, clutch) ─► hand target

With a calibrated hand shape (`set_shape`) the hand frame, the fingertip vectors and the wrist position do not come
from MediaPipe's 3D landmarks directly but from a model of your hand fitted to its 2D landmarks (hand_model.py), which
keeps the depth of hidden landmarks out of the control. Without a shape (before the calibration, or TRACKING=raw)
the MediaPipe landmarks are used as they are.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dex_hand import frames
from dex_hand.hand_model import HandFit
from dex_hand.sources import Observation
from dex_hand.smoothing import LandmarkFilter, PoseSmoother


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
    fit_error_px: Optional[float] = None   # hand model: median distance to MediaPipe's pixels (None = raw landmarks)


class HandPipeline:
    def __init__(self, side, retargeter, scaler, focal_px, cx, cy, smooth=True):
        self.side, self.retargeter, self.scaler = side, retargeter, scaler
        self.focal_px, self.cx, self.cy = focal_px, cx, cy
        self.last = None                                # the last Command (held while the hand is lost)
        self.camera_to_operator = frames.CAMERA_TO_OPERATOR   # replaced by the IMU's gravity when the camera has one
        self.calibration = np.eye(3)                    # operator frame -> calibrated frame (see frames.calibration_from)
        self._t = None
        self.fitter = None                              # HandFit, once a hand shape is installed
        self.unreliable = False                         # the hand model does not agree with the image: the robot holds
        self._lost = 0                                  # consecutive frames without a hand
        self.landmark_filter = LandmarkFilter() if smooth else None   # MediaPipe's landmarks, before anything is computed from them
        self.smoother = PoseSmoother() if smooth else None   # palm rotation and wrist position, weighted by what the image determines
        self.rot_sigma_deg = None                       # hand model: uncertainty of the palm rotation in this frame

    def set_camera_up(self, up_camera):
        """Level the operator frame with gravity: `up_camera` is 'up' in camera axes from the IMU."""
        axes = frames.gravity_aligned_axes(up_camera)
        if axes is not None:
            self.camera_to_operator = axes

    def set_shape(self, shape, fitter=None):
        """Install the calibrated hand geometry (hand_model.HandShape), or None to use MediaPipe's landmarks as they are.
        `fitter`: a HandFit for it that is already tracking (the calibration frames)."""
        self.fitter = None if shape is None else (fitter or HandFit(shape))
        self.unreliable, self._lost = False, 0
        if self.smoother is not None:
            self.smoother.reset()
            self.landmark_filter.reset()

    def set_calibration(self, rotation):
        """Install a calibration; the robot re-anchors on the next engaged frame."""
        self.calibration = np.asarray(rotation, dtype=float)
        self.scaler.align()                                      # the first imitation after a calibration matches your orientation

    def step(self, obs, clutch=False) -> Optional[Command]:
        """`obs`: sources.Observation (world/pixels None = no hand). `clutch`: True holds the robot hand still.
        Returns None until a hand has been seen once."""
        fit, self.unreliable, self.rot_sigma_deg = None, False, None
        now = time.monotonic()
        dt = None if self._t is None else min(now - self._t, 0.1)
        self._t = now
        smoothed = None
        if obs.world is not None and self.landmark_filter is not None:
            world, pixels = self.landmark_filter.update(obs.world, obs.pixels, dt)
            obs = Observation(world, pixels, obs.depth, obs.image)
        elif self.landmark_filter is not None:
            self.landmark_filter.reset()
        if obs.world is not None and self.fitter is not None:
            fit = self.fitter.update(obs.world, obs.pixels, obs.depth, self.focal_px, self.cx, self.cy)
            if not fit.ok:                              # the model cannot explain the image: do not follow it
                fit, self.unreliable = None, True
                rotation = None
            else:
                rotation = fit.rotation
                self.rot_sigma_deg = fit.rot_sigma_deg
                if self.smoother is not None:
                    if fit.reinitialised:
                        self.smoother.reset()
                    smoothed = self.smoother.update(fit.rotation, fit.wrist, fit.cov_rot, fit.cov_pos, dt)
                    rotation = smoothed[0]
        else:
            rotation = None if obs.world is None else frames.hand_rotation(obs.world, self.side)
        self._lost = self._lost + 1 if obs.world is None else 0
        if self._lost == 15 and self.fitter is not None:    # gone for half a second: the hand model starts from MediaPipe's pose again
            self.fitter.reset()
        if rotation is None:                            # lost: hold, and re-anchor when the hand returns
            if self.smoother is not None:
                self.smoother.reset()
            self.scaler.release()
            if self.last is not None:
                self.last.engaged, self.last.ghost = False, None
            return self.last

        if fit is not None:
            tips, wrist, depth_from, landmarks = fit.tip_vectors, (fit.wrist if smoothed is None else smoothed[1]), 'hand model', fit.landmarks
        else:
            tips = frames.tip_vectors(obs.world, rotation)
            depth, depth_from = obs.depth, 'sensor'
            if depth is None:
                depth, depth_from = frames.depth_from_scale(obs.world, obs.pixels, self.focal_px), 'hand size'
            if depth is None:
                return self.last
            wrist = frames.pixel_to_camera(*obs.pixels[0], depth, self.focal_px, self.cx, self.cy)
            landmarks = obs.world
        joints = self.retargeter(tips)
        p_raw, r_raw = frames.to_operator(wrist, rotation, self.camera_to_operator)
        p_op, r_op = self.calibration @ p_raw, self.calibration @ r_raw

        position, hand_rotation = self.scaler.update(p_op, r_op, engaged=not clutch, dt=dt)
        skeleton = (landmarks - landmarks[0]) @ self.camera_to_operator.T @ self.calibration.T
        self.last = Command(joints=joints, position=position,
                            rotation=hand_rotation @ self.retargeter.r_bh.T,
                            ghost=self.scaler.ghost(p_op, r_op), skeleton=skeleton, raw_position=p_op, raw_rotation=r_raw,
                            engaged=self.scaler.engaged, depth_from=depth_from,
                            fit_error_px=None if fit is None else fit.error_px)
        return self.last
