"""Per-arm tracking: noisy 3D joint measurements -> a clean, bone-consistent arm in the body frame.

What it does with each frame's shoulder, elbow and wrist measurements:
  1. rejects physically impossible jumps (speed in m/s, not "cm per frame": it does not depend on fps)
  2. enforces the operator's bone lengths: a joint that is too far from / too near its parent is
     moved onto the sphere of the right radius, and a joint with a good 2D landmark but no usable
     depth is placed on its camera ray at bone length from its parent
  3. expresses elbow and wrist relative to the shoulder in the fixed body frame
  4. smooths with One Euro filters (still = smooth, moving = responsive)
  5. keeps a status: TRACKING / HELD (briefly unmeasured) / LOST
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from g1tele import protocol
from g1tele.config import PerceptionCfg
from g1tele.filters import OneEuro, OneEuroRotation
from g1tele.geometry import unit
from g1tele.hand import DEFAULT_CLOSED, DEFAULT_OPEN, curls, hand_rotation, raw_features
from g1tele.perception.fusion import FusionParams, fuse_arm
from g1tele.protocol import ArmObs, BoneLengths


@dataclass
class Measure:
    """One landmark in one frame."""
    uv: np.ndarray                       # pixel
    vis: float                           # landmark visibility 0..1
    ray: np.ndarray                      # unit camera ray through the pixel
    P: Optional[np.ndarray] = None       # 3D joint centre, camera frame; None = no usable depth


class ArmTracker:
    def __init__(self, side: str, cfg: PerceptionCfg):
        self.side, self.cfg = side, cfg
        f = cfg.filters
        self.f_shoulder = OneEuro(f.shoulder_min_cutoff, f.shoulder_beta)
        self.f_elbow = OneEuro(f.elbow_min_cutoff, f.elbow_beta)
        self.f_wrist = OneEuro(f.wrist_min_cutoff, f.wrist_beta)
        self.fusion = FusionParams(depth_sigma=cfg.depth_sigma, bone_sigma=cfg.bone_sigma)
        self.reset()

    def reset(self):
        for filt in (self.f_shoulder, self.f_elbow, self.f_wrist):
            filt.reset()
        self.prev_depths = None                     # (elbow, wrist) distances along their rays
        self.prev_w: Optional[np.ndarray] = None
        self.prev_w_t = -1e9
        self.jump_rejects = 0
        self.bad_depth_frames = 0
        self.last_t: Optional[float] = None         # last time the wrist was measured
        self.last_obs: Optional[ArmObs] = None
        self.shoulder_cam: Optional[np.ndarray] = None
        self.shoulder_t = -1e9
        self.flags = {'E_corrected': False, 'W_corrected': False}

    # ------------------------------------------------------------------ update
    def update(self, t: float, shoulder: Optional[np.ndarray], elbow: Optional[Measure],
               wrist: Optional[Measure], frame_rot: np.ndarray,
               bones: Optional[BoneLengths]) -> ArmObs:
        """`shoulder`: 3D (camera frame), possibly predicted. `frame_rot`: columns forward, left,
        up of the body frame in camera coordinates."""
        cfg = self.cfg
        for k in self.flags:
            self.flags[k] = False
        held_shoulder = False
        if shoulder is not None:
            s = self.f_shoulder(shoulder, t)
            self.shoulder_cam, self.shoulder_t = s, t
        elif self.shoulder_cam is not None and t - self.shoulder_t < cfg.shoulder_hold:
            # the shoulders barely move: when they are hidden (arm reaching at the camera), keep
            # the last known position instead of dropping the whole arm
            s, held_shoulder = self.shoulder_cam, True
        else:
            return self._unmeasured(t)
        lu, lf = (bones.upper, bones.fore) if bones is not None and bones.valid() else (0.0, 0.0)

        usable_e = elbow is not None and elbow.vis >= cfg.min_visibility
        usable_w = wrist is not None and wrist.vis >= cfg.min_visibility
        e_pt = w_pt = None
        e_how = w_how = 'measured'
        if cfg.fusion and usable_w and lu > 0 and lf > 0:
            res = fuse_arm(s, elbow.ray if usable_e else None,
                           None if not usable_e or elbow.P is None else float(np.linalg.norm(elbow.P)),
                           wrist.ray, None if wrist.P is None else float(np.linalg.norm(wrist.P)),
                           lu, lf, self.prev_depths, self.fusion)
            # A wrist that moved faster than any human arm is a wrong solution (e.g. the other
            # intersection of a ray with a bone sphere): redo the fusion anchored at the previous
            # depths. After a few frames of this the new solution is accepted (the old one was wrong).
            if (res is not None and self.prev_w is not None and self.prev_depths is not None
                    and t - self.prev_w_t < 0.5):
                limit = cfg.max_joint_speed * max(t - self.prev_w_t, 1e-3) + 0.06
                if np.linalg.norm(res.wrist - self.prev_w) > limit and self.jump_rejects < 8:
                    self.jump_rejects += 1
                    anchored = fuse_arm(s, elbow.ray if usable_e else None,
                                        self.prev_depths[0] if usable_e and self.prev_depths[0] else None,
                                        wrist.ray, self.prev_depths[1], lu, lf, self.prev_depths, self.fusion)
                    if anchored is not None:
                        res = anchored
                else:
                    self.jump_rejects = 0
            if res is not None:
                e_pt, w_pt = res.elbow, res.wrist
                self.prev_w, self.prev_w_t = w_pt.copy(), t
                self.prev_depths = (float(np.linalg.norm(e_pt)) if e_pt is not None else None,
                                    float(np.linalg.norm(w_pt)))
                if self.prev_depths[0] is None:
                    self.prev_depths = None
                # Stuck on a wrong solution? Depth readings that disagree for a whole second mean the
                # temporal history is what is wrong: forget it and start over from the measurement.
                if wrist.P is not None and res.wrist_depth_weight < 0.2:
                    self.bad_depth_frames += 1
                    if self.bad_depth_frames > 30:
                        self.bad_depth_frames = 0
                        self.prev_depths = self.prev_w = None
                        self.f_wrist.reset()
                        self.f_elbow.reset()
                else:
                    self.bad_depth_frames = 0
                if res.wrist_depth_weight < 0.5:
                    w_how = 'corrected'
                if res.elbow_depth_weight < 0.5:
                    e_how = 'corrected'
        elif usable_w and wrist.P is not None:
            w_pt = wrist.P                       # not calibrated yet: raw measurement
            if usable_e and elbow.P is not None:
                e_pt = elbow.P
        if w_pt is None:
            return self._unmeasured(t)

        self.last_t = t
        w_rel = self.f_wrist(frame_rot.T @ (w_pt - s), t)
        e_rel = None
        if e_pt is not None:
            e_rel = self.f_elbow(frame_rot.T @ (e_pt - s), t)
        else:
            self.f_elbow.reset()

        q = 1.0
        if wrist is not None:
            q *= float(np.clip(wrist.vis, 0.0, 1.0))
        if w_how in ('ray', 'corrected'):
            q *= 0.6
        if held_shoulder:
            q *= 0.7
        if e_pt is None:
            q *= 0.5
        elif e_how in ('ray', 'corrected'):
            q *= 0.8
        obs = ArmObs(status=protocol.TRACKING, quality=q, wrist=w_rel, elbow=e_rel)
        self.last_obs = obs
        return obs

    def _unmeasured(self, t: float) -> ArmObs:
        """The wrist was not measured this frame: repeat the last value briefly, then give up."""
        if self.last_obs is None or self.last_t is None:
            return ArmObs(status=protocol.LOST)
        age = t - self.last_t
        if age <= self.cfg.hold_time:
            held = ArmObs(status=protocol.HELD, quality=self.last_obs.quality * float(np.exp(-4 * age)),
                          wrist=self.last_obs.wrist, elbow=self.last_obs.elbow)
            return held
        return ArmObs(status=protocol.LOST)


class HandTrack:
    """Palm orientation and finger curls of one hand, filtered."""

    def __init__(self, side: str, cfg: PerceptionCfg, open_=None, closed=None):
        self.side, self.cfg = side, cfg
        f = cfg.filters
        self.f_rot = OneEuroRotation(f.hand_rot_min_cutoff, f.hand_rot_beta)
        self.f_fingers = [OneEuro(f.finger_min_cutoff, f.finger_beta) for _ in range(6)]
        self.open = DEFAULT_OPEN.copy() if open_ is None else np.asarray(open_)
        self.closed = DEFAULT_CLOSED.copy() if closed is None else np.asarray(closed)
        self.last_t: Optional[float] = None
        self.R: Optional[np.ndarray] = None
        self.fingers: Optional[np.ndarray] = None
        self.last_raw: Optional[np.ndarray] = None

    def reset(self):
        self.f_rot.reset()
        for f in self.f_fingers:
            f.reset()
        self.last_t = self.R = self.fingers = self.last_raw = None

    def set_range(self, open_, closed):
        self.open, self.closed = np.asarray(open_, dtype=float), np.asarray(closed, dtype=float)

    def update(self, t: float, world: np.ndarray, frame_rot: np.ndarray) -> bool:
        """world: 21x3 hand landmarks in camera axes. Returns True if a reading was accepted."""
        world = world * np.asarray(self.cfg.hand_axes_sign)
        r_cam = hand_rotation(world, self.side)
        if r_cam is None:
            return False
        r_body = frame_rot.T @ r_cam
        self.R = self.f_rot(r_body, t)
        raw = raw_features(world)
        self.last_raw = raw
        c = curls(raw, self.open, self.closed)
        self.fingers = np.array([flt(v, t) for flt, v in zip(self.f_fingers, c)]).ravel()
        self.last_t = t
        return True

    def fresh(self, t: float, timeout: float = 0.3) -> bool:
        return self.last_t is not None and t - self.last_t < timeout


class WaistEstimator:
    """Waist yaw, roll and pitch from the shoulders, relative to the calibration pose. Optional."""

    def __init__(self, cfg: PerceptionCfg, torso_length: float = 0.45, deadband_deg: float = 3.0):
        f = cfg.filters
        self.filt = OneEuro(f.waist_min_cutoff, f.waist_beta)
        self.torso_length = torso_length
        self.deadband = np.radians(deadband_deg)
        self.mid0: Optional[np.ndarray] = None   # shoulder midpoint at calibration, body frame

    def reset(self):
        self.filt.reset()
        self.mid0 = None

    def update(self, t: float, r_sh: np.ndarray, l_sh: np.ndarray, frame_rot: np.ndarray):
        mid_b = frame_rot.T @ ((r_sh + l_sh) / 2)
        if self.mid0 is None:
            self.mid0 = mid_b
        line = unit(frame_rot.T @ (l_sh - r_sh))
        if line is None:
            return None
        yaw = np.arctan2(-line[0], line[1])
        roll = np.arcsin(np.clip(line[2], -1.0, 1.0))
        off = mid_b - (self.mid0 - [0.0, 0.0, self.torso_length])
        pitch = np.arctan2(off[0], off[2])
        a = self.filt(np.array([yaw, roll, pitch]), t)
        return np.sign(a) * np.maximum(np.abs(a) - self.deadband, 0.0)
