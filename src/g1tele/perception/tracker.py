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
from g1tele.geometry import rotation_angle, unit
from g1tele.hand import DEFAULT_CLOSED, DEFAULT_OPEN, curls, hand_rotation, raw_features
from g1tele.perception.fusion import FusionParams, circle_nearest_to_ray, circle_point, fuse_arm, wrist_first
from g1tele.protocol import ArmObs, BoneLengths


@dataclass
class Measure:
    """One landmark in one frame."""
    uv: np.ndarray                       # pixel
    vis: float                           # landmark visibility 0..1
    ray: np.ndarray                      # unit camera ray through the pixel
    P: Optional[np.ndarray] = None       # 3D joint centre, camera frame; None = no usable depth
    clusters: Optional[list] = None      # distinct surfaces in the depth window, nearest first: [(z, share)]


class ArmTracker:
    def __init__(self, side: str, cfg: PerceptionCfg):
        self.side, self.cfg = side, cfg
        f = cfg.filters
        self.f_shoulder = OneEuro(f.shoulder_min_cutoff, f.shoulder_beta)
        self.f_elbow = OneEuro(f.elbow_min_cutoff, f.elbow_beta)
        self.f_wrist = OneEuro(f.wrist_min_cutoff, f.wrist_beta)        # the wrist's DIRECTION from the camera (accurate)
        self.f_wrist_depth = OneEuro(f.wrist_depth_min_cutoff, f.wrist_depth_beta)   # its DISTANCE (noisy)
        self.fusion = FusionParams(depth_sigma=cfg.depth_sigma, bone_sigma=cfg.bone_sigma)
        self.reset()

    def reset(self):
        for filt in (self.f_shoulder, self.f_elbow, self.f_wrist, self.f_wrist_depth):
            filt.reset()
        self.prev_depths = None                     # (elbow, wrist) distances along their rays
        self.prev_w: Optional[np.ndarray] = None
        self.prev_w_t = -1e9
        self.jump_rejects = 0
        self.bad_depth_frames = 0
        self.fused_points = None
        self.prev_elbow: Optional[np.ndarray] = None
        self.c_ema: Optional[float] = None
        self.e_q = 0.0
        self.last_t: Optional[float] = None         # last time the wrist was measured
        self.last_obs: Optional[ArmObs] = None
        self.shoulder_cam: Optional[np.ndarray] = None
        self.shoulder_t = -1e9
        self.flags = {'E_corrected': False, 'W_corrected': False}

    # ------------------------------------------------------------------ update
    @staticmethod
    def _smooth(x, lo, hi):
        x = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
        return x * x * (3 - 2 * x)

    def _upper_arm_toward_camera(self, shoulder) -> float:
        """0..1: how much the upper arm (from the PREVIOUS frame, so it is stable) points at the camera."""
        if self.prev_elbow is None or self.shoulder_cam is None:
            return 0.0
        ray = shoulder / (np.linalg.norm(shoulder) + 1e-9)
        d = unit(self.prev_elbow - self.shoulder_cam)
        return 0.0 if d is None else max(0.0, float(-(d @ ray)))

    def _wrist_depth(self, wrist, s, radius: float):
        """Distance of the wrist along its ray, following ONE surface from frame to frame.

        The window around the wrist holds the hand and the wrist/forearm when the hand points at the camera;
        which one a percentile picks flips with a pixel of noise (the glitch). Every surface is a candidate
        (the nearest one in two versions: it is the wrist itself, or the hand in front of the wrist, then
        corrected by the end-on bias); the candidate closest to the previous depth wins."""
        if wrist.P is None:
            return None
        ray = wrist.ray
        bias = self._endon_bias(s, wrist.P, ray)
        cands = []
        for k, (z, _) in enumerate(wrist.clusters or []):
            b = z / max(ray[2], 0.2) + radius
            if k == 0:
                cands += [(b + bias, 0), (b, 1)]       # (distance, version)
            else:
                cands.append((b, 1))
        if not cands:
            return float(np.linalg.norm(wrist.P)) + bias
        prev_b = None if self.prev_depths is None else self.prev_depths[1]
        if prev_b is None:
            return cands[0][0]                          # first frame: the nearest surface, corrected
        # prefer the candidate closest to where the wrist was; it is a tracked value, not a vote per frame
        best = min(cands, key=lambda c: abs(c[0] - prev_b))
        return best[0]

    def _endon_bias(self, s, w_raw, ray) -> float:
        """How much too near the wrist depth reads because the HAND is in front of the wrist.

        With the arm pointing at the camera the hand protrudes toward it, so the nearest surface around the
        wrist pixel is the hand, not the wrist. The correction grows with how much the shoulder-to-wrist
        direction points at the camera (cos^2: none from the side, the full bias end on). The direction
        comes from the previous frame and is smoothed, so the correction itself adds no noise."""
        ref = self.prev_w if self.prev_w is not None else w_raw
        d = unit(ref - s)
        if d is None:
            return 0.0
        c_now = max(0.0, float(-(d @ ray)))
        self.c_ema = c_now if self.c_ema is None else 0.85 * self.c_ema + 0.15 * c_now
        return self.cfg.endon_depth_bias * self.c_ema * self.c_ema

    def _elbow_quality(self, s_cam, elbow, wrist, lu, lf, fx) -> float:
        """How far the elbow estimate can be trusted, 0..1. A forearm (or upper arm) pointing at the camera
        looks very short in the image for its length, and then its far end is hidden behind the near
        end: MediaPipe guesses the elbow and the depth under it belongs to the forearm in front."""
        if elbow is None or wrist is None or elbow.vis < self.cfg.min_visibility or lu <= 0 or lf <= 0:
            return 0.0
        z = wrist.P[2] if wrist.P is not None else (s_cam[2] if s_cam is not None else 1.8)
        fore_px = float(np.linalg.norm(wrist.uv - elbow.uv))
        ratio = fore_px / max(fx * lf / max(z, 0.3), 1.0)          # 1 = seen from the side, 0 = end on
        return self._smooth(ratio, 0.35, 0.70) * float(np.clip(elbow.vis, 0.0, 1.0))

    def update(self, t: float, shoulder: Optional[np.ndarray], elbow: Optional[Measure],
               wrist: Optional[Measure], frame_rot: np.ndarray,
               bones: Optional[BoneLengths], fx: float = 600.0) -> ArmObs:
        """`shoulder`: 3D (camera frame), possibly predicted. `frame_rot`: columns forward, left,
        up of the body frame in camera coordinates."""
        cfg = self.cfg
        for k in self.flags:
            self.flags[k] = False
        held_shoulder = False
        if shoulder is not None and self.shoulder_cam is not None:
            # the more the upper arm points at the camera, the more it sits in front of the shoulder and the
            # less the depth there is the shoulder's: fade the new reading out (the shoulder hardly moves)
            w = 1.0 - self._smooth(self._upper_arm_toward_camera(shoulder), 0.5, 0.85)
            shoulder = self.shoulder_cam + w * (shoulder - self.shoulder_cam)
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
        e_q_raw = self._elbow_quality(s, elbow, wrist, lu, lf, fx) if usable_e else 0.0
        # Trust drops at once but recovers slowly: MediaPipe's visibility of an occluded elbow flickers from
        # frame to frame, and a quality that flickers would flicker the wrist target that is blended with it.
        prev_q = self.e_q
        self.e_q = e_q_raw if e_q_raw < prev_q else prev_q + 0.08 * (e_q_raw - prev_q)
        e_q = self.e_q
        e_pt = w_pt = None
        e_how = w_how = 'measured'
        if cfg.fusion and usable_w and lu > 0 and lf > 0:
            # Two estimators, blended by how far the elbow can be trusted (so there is never a switch):
            #  - wrist-first: the wrist from its own ray and depth only (immune to a bad elbow)
            #  - full fusion: depth + bones + rays for both joints (better when the elbow is good)
            prev_b = None if self.prev_depths is None else self.prev_depths[1]
            b_m = self._wrist_depth(wrist, s, cfg.radii.wrist)
            w_first = wrist_first(s, wrist.ray, b_m, lu, lf, prev_b)
            alpha = self._smooth(e_q, 0.45, 0.80)
            res = None
            if alpha > 0.0:
                res = fuse_arm(s, elbow.ray if usable_e else None,
                               None if not usable_e or elbow.P is None else float(np.linalg.norm(elbow.P)),
                               wrist.ray, b_m, lu, lf, self.prev_depths, self.fusion)
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
                if res is not None and wrist.P is not None:
                    if res.wrist_depth_weight < 0.2:
                        self.bad_depth_frames += 1
                        if self.bad_depth_frames > 30:      # stuck on a wrong solution: forget the history
                            self.bad_depth_frames = 0
                            self.prev_depths = self.prev_w = None
                            self.f_wrist.reset()
                            self.f_elbow.reset()
                    else:
                        self.bad_depth_frames = 0
            if res is not None and w_first is not None:
                w_pt = alpha * res.wrist + (1.0 - alpha) * w_first
            elif res is not None:
                w_pt = res.wrist
            else:
                w_pt = w_first
            if w_pt is not None:
                # elbow: on the circle the bones allow for THIS wrist, at the swivel the evidence suggests
                hint = self.prev_elbow
                if res is not None and res.elbow is not None:
                    hint = res.elbow if hint is None else alpha * res.elbow + (1.0 - alpha) * hint
                if usable_e and alpha < 1.0:
                    # the elbow's 2D landmark (its line of sight) picks the swivel; its depth is not trusted
                    from_ray = circle_nearest_to_ray(s, w_pt, lu, lf, elbow.ray)
                    w2d = 0.35 + 0.65 * e_q                           # a hidden elbow is still a hint
                    meas = from_ray if hint is None else w2d * from_ray + (1.0 - w2d) * hint
                    hint = meas if hint is None else alpha * hint + (1.0 - alpha) * meas
                e_pt = circle_point(s, w_pt, lu, lf, hint)
                self.prev_elbow = e_pt.copy()
                self.prev_w, self.prev_w_t = w_pt.copy(), t
                self.prev_depths = (float(np.linalg.norm(e_pt)), float(np.linalg.norm(w_pt)))
                w_how = 'measured' if alpha >= 0.8 else 'wrist_first'
        elif usable_w and wrist.P is not None:
            w_pt = wrist.P                       # not calibrated yet: raw measurement
            if usable_e and elbow.P is not None:
                e_pt = elbow.P
        if w_pt is None:
            return self._unmeasured(t)

        self.last_t = t
        self.fused_points = (s, e_pt, w_pt)         # camera frame, for the debug overlay
        # The wrist's direction from the camera comes from the 2D landmark (accurate); its distance comes
        # from depth, which is noisy and far noisier when the arm points at the camera. Filtering the
        # body-frame vector as one thing lets that noise inflate the speed estimate and defeats the
        # smoothing, so the two parts are filtered separately, the distance harder end on.
        f = cfg.filters
        dist, ray = float(np.linalg.norm(w_pt)), w_pt / (np.linalg.norm(w_pt) + 1e-9)
        c = self.c_ema if self.c_ema is not None else 0.0
        self.f_wrist_depth.min_cutoff = f.wrist_depth_min_cutoff + (f.wrist_depth_endon_cutoff
                                                                      - f.wrist_depth_min_cutoff) * min(1.0, c / 0.8)
        ray_f = self.f_wrist(1.8 * ray, t)                       # x 1.8 m: metres at a typical distance
        w_pt = (ray_f / (np.linalg.norm(ray_f) + 1e-9)) * float(self.f_wrist_depth(dist, t))
        w_rel = frame_rot.T @ (w_pt - s)
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
        obs = ArmObs(status=protocol.TRACKING, quality=q, wrist=w_rel, elbow=e_rel, elbow_quality=e_q)
        self.last_obs = obs
        return obs

    def _unmeasured(self, t: float) -> ArmObs:
        """The wrist was not measured this frame: repeat the last value briefly, then give up."""
        if self.last_obs is None or self.last_t is None:
            return ArmObs(status=protocol.LOST)
        age = t - self.last_t
        if age <= self.cfg.hold_time:
            held = ArmObs(status=protocol.HELD, quality=self.last_obs.quality * float(np.exp(-4 * age)),
                          wrist=self.last_obs.wrist, elbow=self.last_obs.elbow,
                          elbow_quality=self.last_obs.elbow_quality)
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
        self.rejects = 0
        self.quality = 0.0              # share of recent readings that were accepted, 0..1
        self.reject_reason = ''

    def reset(self):
        self.f_rot.reset()
        self.rejects, self.quality = 0, 0.0
        for f in self.f_fingers:
            f.reset()
        self.last_t = self.R = self.fingers = self.last_raw = None

    def set_range(self, open_, closed):
        self.open, self.closed = np.asarray(open_, dtype=float), np.asarray(closed, dtype=float)

    @staticmethod
    def _angle(a, b) -> float:
        a, b = unit(a), unit(b)
        return 180.0 if a is None or b is None else float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))

    def update(self, t: float, world: np.ndarray, frame_rot: np.ndarray, arm_dir=None, pose_dir=None,
               model_dir=None) -> bool:
        """world: 21x3 hand landmarks in camera axes. Returns True if the reading was accepted.

        A reading is rejected (and neither orientation nor fingers use it) when
          * its finger axis points far from the forearm (arm_dir, camera frame): a flipped palm or the other hand
          * its image direction wrist -> knuckles disagrees with the Pose model's (pose_dir vs model_dir, 2D)
          * the palm would have turned faster than a hand can (hand_max_rate): flips are instantaneous
        and after 8 rejections in a row the new readings are believed (the old reference was wrong)."""
        cfg = self.cfg
        world = world * np.asarray(cfg.hand_axes_sign)
        r_cam = hand_rotation(world, self.side)
        self.reject_reason = ''
        if r_cam is None:
            self.reject_reason = 'degenerate'
        elif arm_dir is not None and self._angle(r_cam[:, 0], arm_dir) > cfg.hand_arm_max_deg:
            self.reject_reason = 'not along the forearm'
        elif pose_dir is not None and model_dir is not None and self._angle(pose_dir, model_dir) > cfg.hand_pose_max_deg:
            self.reject_reason = 'disagrees with the pose model'
        if not self.reject_reason and self.R is not None and self.last_t is not None:
            r_body_new = frame_rot.T @ r_cam
            limit = cfg.hand_max_rate * max(t - self.last_t, 1e-3) + 0.3
            if rotation_angle(self.R, r_body_new) > limit:
                self.rejects += 1
                if self.rejects < 8:
                    self.reject_reason = 'turned too fast'
        if self.reject_reason:
            self.quality *= 0.8
            return False
        self.rejects = 0
        self.quality = 0.8 * self.quality + 0.2
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
