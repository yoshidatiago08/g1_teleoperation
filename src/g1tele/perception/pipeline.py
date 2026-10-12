"""The perception pipeline: RGB-D frame in, OperatorFrame out. Runs where the camera is.

    pose (2D landmarks) -> lift to 3D (depth, joint radius) -> calibration | tracking
                                                                   |
        shoulders -> body frame;  per arm: bone constraints + One Euro filters;  hands; waist

It has no networking and no GUI, so the same object runs live, on a recording (`g1tele-bench`) and
in the tests on a synthetic operator.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import logging

from g1tele import protocol
from g1tele.config import Config
from g1tele.geometry import estimate_shoulders, unit
from g1tele.perception.calibration import Calibrator, Profile, validate_profile
from g1tele.perception.monitor import CalibrationMonitor
from g1tele.perception.camera import RGBDFrame
from g1tele.perception.gravity import estimate_up_from_floor
from g1tele.hand import hand_rotation
from g1tele.perception.gestures import NodDetector, head_width
from g1tele.perception.hand_depth import occluded_landmarks, palm_normal_from_depth
from g1tele.perception.hands_calib import HandsEnterCalibration
from g1tele.perception.lifting import depth_clusters, lift, pixel_ray
from g1tele.perception.pose_mediapipe import ELBOW, HAND_GUIDE, HAND_LENGTH, SHOULDER, WRIST
from g1tele.perception.tracker import ArmTracker, HandTrack, Measure, WaistEstimator
from g1tele.protocol import ArmObs, OperatorFrame


log = logging.getLogger('g1tele.perception')


class PerceptionPipeline:
    def __init__(self, cfg: Config, pose, hands=None, profile: Optional[Profile] = None,
                 calibrated: bool = False, hands_full=None):
        """`calibrated=True` with a profile that has a body frame starts tracking at once (replaying a recording
        that began after the calibration)."""
        self.cfg = cfg
        self.pcfg = cfg.perception
        self.pose = pose
        # One hand estimator PER HAND: a MediaPipe graph must not be called from two threads at once,
        # and the two hands are processed in parallel.
        if hands is not None and not isinstance(hands, dict):
            hands = {side: hands for side in cfg.perception.arms}
        self.hands = hands
        self.hands_full = hands_full         # whole-image hand detector for the hands-only mode (control = HANDS)
        self.stage = 'reach'                     # reach -> enter (calibration, robot holds) -> hands -> exit (match, robot holds) -> reach
        self.nod = NodDetector(self.pcfg.nod_amplitude, self.pcfg.nod_count, self.pcfg.nod_window_s,
                               self.pcfg.nod_cooldown_s)
        self.enter_cal: Optional[HandsEnterCalibration] = None
        self.entry_wrist: Dict[str, np.ndarray] = {}         # the wrists (body frame, from the shoulder) when reach mode ended
        self._last_wrists: Dict[str, np.ndarray] = {}
        self._pose_t = -1e9                                  # last time the body model ran in hands-only mode
        self._match_since: Optional[float] = None
        self._toggle = False                                 # a request from the keyboard (another thread)
        self.sides = list(self.pcfg.arms)
        self.trackers = {s: ArmTracker(s, self.pcfg) for s in self.sides}
        self.hand_tracks = {s: HandTrack(s, self.pcfg) for s in self.sides}
        self.waist = WaistEstimator(self.pcfg) if self.pcfg.waist else None
        self.profile: Optional[Profile] = profile
        self.monitor = CalibrationMonitor(cfg.calibration, self.sides)
        self.notice = ''                 # why a calibration was (re)started, shown with the prompt
        self.notice_until = 0.0
        self.drift_events: list = []
        self.rejections = 0
        self.calibrator: Optional[Calibrator] = None
        self.seq = 0
        self._up_guess: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(0)
        self.left_vec: Optional[np.ndarray] = None       # last measured shoulder line, camera frame
        self.sh_prev: Dict[str, np.ndarray] = {}
        self.sh_prev_t: Dict[str, float] = {}
        self.sh_rejects: Dict[str, int] = {}
        self.shoulder_width: Optional[float] = profile.shoulder_width if profile else None
        self._pool = ThreadPoolExecutor(max_workers=2) if hands is not None else None
        self.hand_points: Dict[str, Optional[np.ndarray]] = {s: None for s in self.sides}
        self.last_debug: dict = {}
        if profile is not None:
            self._apply_profile_hands(profile)
            bad = validate_profile(Profile(bones=profile.bones, shoulder_width=profile.shoulder_width,
                                           frame=np.column_stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]]),
                                           up=np.array([0.0, -1.0, 0.0])), self.sides) \
                if self._profile_has_bones(profile) else ['no arm lengths']
            if bad:                      # a stored calibration that is implausible must not be reused
                self.notice, self.notice_until = f'saved calibration ignored ({bad[0]})', 1e18
                profile = self.profile = None
        if calibrated and profile is not None and profile.frame is not None:
            self.profile, self.calibrator = profile, None
            self.shoulder_width = profile.shoulder_width or None
        else:
            self.start_calibration(full=profile is None or not self._profile_has_bones(profile))

    # ------------------------------------------------------------------ calibration
    def _profile_has_bones(self, p: Profile) -> bool:
        return all(s in p.bones and p.bones[s].valid() for s in self.sides)

    def start_calibration(self, full: bool = True):
        """Begin a calibration. `full` walks through all the poses; otherwise only the rest pose is
        repeated (to re-establish the body frame) and the saved bones are kept."""
        self.stage = 'reach'
        self.enter_cal = None
        base = None
        if self.profile is not None:
            base = Profile(bones=dict(self.profile.bones), shoulder_width=self.profile.shoulder_width,
                           hand_open=list(self.profile.hand_open), hand_closed=list(self.profile.hand_closed))
        self.calibrator = Calibrator(self.cfg.calibration, self.pcfg, existing=base, full=full)
        self.profile = self.calibrator.profile
        self.profile.frame = None
        self.monitor.reset()
        for t in self.trackers.values():
            t.reset()
        for h in self.hand_tracks.values():
            h.reset()
        if self.waist is not None:
            self.waist.reset()

    def _apply_profile_hands(self, p: Profile):
        for h in self.hand_tracks.values():
            h.set_range(p.hand_open, p.hand_closed)

    @property
    def calibrating(self) -> bool:
        return self.calibrator is not None and not self.calibrator.done

    # ------------------------------------------------------------------ control stages
    @property
    def control(self) -> str:
        """What the robot is told to do: REACH (follow the arms), HANDS (follow the hands), HOLD (stay put)."""
        return {'reach': protocol.REACH, 'hands': protocol.HANDS}.get(self.stage, protocol.HOLD)

    def request_toggle(self):
        """The keyboard's 'h' (any thread): handled at the next frame."""
        self._toggle = True

    def _can_enter(self) -> bool:
        return not (self.hands_full is None or self.profile is None or self.profile.frame is None or self.calibrating)

    def _begin_enter(self, t: float, why: str = 'nod') -> bool:
        if not self._can_enter():
            self.notice, self.notice_until = 'hands-only mode needs a calibration and the hand model', t + 4.0
            return False
        self.entry_wrist = {s: w.copy() for s, w in self._last_wrists.items()}
        self.enter_cal = HandsEnterCalibration(self.sides, self.pcfg.hands_calib_range, self.pcfg.hands_hold_s,
                                               self.pcfg.hands_calib_timeout_s)
        self.stage = 'enter'
        self.nod.reset()
        self._pose_t = -1e9
        for h in self.hand_tracks.values():
            h.reset_orientation()
        log.info('hands-only mode: calibration starts (%s); the robot holds', why)
        return True

    def _begin_exit(self, t: float, why: str = 'nod'):
        self.stage = 'exit'
        self.enter_cal = None
        self._match_since = None
        self.nod.reset()
        self.nod.blocked_until = t + self.pcfg.nod_cooldown_s
        for tr in self.trackers.values():
            tr.reset()
        log.info('leaving hands-only mode (%s): match the robot, then reach mode resumes', why)

    def _finish_exit(self, t: float):
        self.stage = 'reach'
        self.nod.reset()
        self.nod.blocked_until = t + self.pcfg.nod_cooldown_s
        if self.profile is not None:
            self._apply_profile_hands(self.profile)
        log.info('reach mode')

    def _handle_toggle(self, t: float):
        if not self._toggle:
            return
        self._toggle = False
        if self.stage == 'reach':
            self._begin_enter(t, 'key')
        elif self.stage in ('enter', 'hands'):
            self._begin_exit(t, 'key')
        else:
            self._finish_exit(t)

    def _check_nod(self, t: float, pose, tracking_ok: bool) -> bool:
        if pose is None or not tracking_ok:
            self.nod.update(t, np.zeros(2), None)
            return False
        return self.nod.update(t, pose.uv[0], head_width(pose.uv, pose.vis))

    def _match_exit(self, t: float, out: OperatorFrame, debug):
        """Leaving hands-only mode: the robot still stands where it froze; reach mode resumes when your wrists are
        back where they were when you left it (the circles in the window)."""
        k = debug.get('intrinsics')
        errs, shown = [], {}
        for s in self.sides:
            target, arm = self.entry_wrist.get(s), out.arms.get(s)
            if target is None:
                continue
            sh = self.trackers[s].shoulder_cam
            if sh is not None and k is not None:
                p = sh + self.profile.frame @ target
                if p[2] > 0.05:
                    shown[s] = (int(k.fx * p[0] / p[2] + k.cx), int(k.fy * p[1] / p[2] + k.cy))
            e = None if arm is None or arm.wrist is None or arm.status != protocol.TRACKING \
                else float(np.linalg.norm(arm.wrist - target))
            errs.append(e)
            debug.setdefault('match', {})[s] = (shown.get(s), e)
        good = bool(errs) and all(e is not None and e < self.pcfg.match_tol_m for e in errs)
        if good:
            self._match_since = self._match_since or t
            left = self.pcfg.match_dwell_s - (t - self._match_since)
            if left <= 0:
                self._finish_exit(t)
                out.prompt = 'REACH mode'
                return
            out.prompt = f'Matched. Hold... {left:.1f} s'
        else:
            self._match_since = None
            worst = max((e for e in errs if e is not None), default=None)
            out.prompt = ('Match the robot: put your wrists on the circles' +
                          ('' if worst is None else f' ({worst * 100:.0f} cm off)') if errs
                          else 'No arm was tracked when you left reach mode: press h to resume')

    def _process_hands(self, frame: RGBDFrame, rgb, out: OperatorFrame, timing, debug):
        """Hands-only (and its calibration): MediaPipe Hands on the whole image, no body. The hand's pose goes out as
        a "wrist" (the palm position in body axes) and a hand frame; the retargeter turns it into the robot's wrist
        motion. The body model runs at a low rate, only to see the head nod that leaves the mode."""
        t = frame.t_capture
        frame_rot = self.profile.frame
        t0 = time.perf_counter()
        detections = {side: (world, image) for world, image, side in self.hands_full.detect(rgb)}
        timing['hands'] = (time.perf_counter() - t0) * 1e3
        out.mode = protocol.TRACKING
        debug.update(hand_axes={}, hand_center={}, hand_reject={}, hand_palm_source={})
        info = {}
        for s in self.sides:
            ht = self.hand_tracks[s]
            obs = ArmObs(status=protocol.LOST)
            out.arms[s] = obs
            if s not in detections:
                continue
            world, image = detections[s]
            palm, hidden = self._hand_depth_cues(frame, (world, image), s, ht, frame_rot)
            if not ht.update(t, world, frame_rot, None, None, None, palm, hidden) and not ht.fresh(t):
                debug['hand_reject'][s] = ht.reject_reason
                continue
            centre = self._palm_centre(frame, image, palm)
            if centre is None or ht.R is None:
                continue
            obs.status, obs.quality = protocol.TRACKING, 1.0
            obs.wrist = frame_rot.T @ centre
            obs.hand_R, obs.fingers, obs.hand_quality = ht.R.copy(), ht.fingers.copy(), float(ht.quality)
            obs.tips = None if ht.tips is None else ht.tips.copy()
            info[s] = {'depth': float(centre[2]), 'raw': ht.last_raw, 'p': centre}
            debug['hands'][s] = image
            debug['hand_axes'][s] = frame_rot @ ht.R
            debug['hand_center'][s] = centre
            debug['hand_reject'][s] = ht.reject_reason
            debug['hand_palm_source'][s] = ht.palm_source
            debug['status'][s] = obs.status
        # the head, a few times a second
        face = None
        if t - self._pose_t >= 1.0 / max(self.pcfg.hands_pose_hz, 1e-3):
            self._pose_t = t
            pose = self.pose.process(rgb)
            width = None if pose is None else head_width(pose.uv, pose.vis)
            face = width is not None
            self._face_seen = face
            nodded = pose is not None and self.nod.update(t, pose.uv[0], width)
            if width is None:
                self.nod.update(t, np.zeros(2), None)
        else:
            nodded = False
        if self.stage == 'enter':
            prompt, done, failed = self.enter_cal.update(t, info)
            out.prompt = prompt
            if done:
                for s, ht in self.hand_tracks.items():
                    if s in self.enter_cal.open_raw and s in self.enter_cal.closed_raw:
                        ht.set_range(self.enter_cal.open_raw[s], self.enter_cal.closed_raw[s])
                    ht.reset_orientation()
                self.stage = 'hands'
                self.nod.reset()
                self.nod.blocked_until = t + self.pcfg.nod_cooldown_s
                out.prompt = 'HANDS-ONLY mode (two nods to leave)'
                log.info('hands-only mode: calibrated%s', ' (' + self.enter_cal.warning + ')' if self.enter_cal.warning else '')
            elif failed:
                self._begin_exit(t, 'calibration timed out')
            elif self.enter_cal.step >= 1 and not getattr(self, '_face_seen', True):
                out.prompt += '   (your face is not in view: you will need the keyboard to leave)'
        elif nodded:
            self._begin_exit(t, 'nod')

    def _palm_centre(self, frame: RGBDFrame, image, palm) -> Optional[np.ndarray]:
        """Camera-frame position of the palm: the centroid of the fitted palm plane, else the depth under the knuckles."""
        if palm is not None:
            return palm.plane.centroid.copy()
        u, v = np.mean(image[[0, 5, 9, 13, 17]], axis=0)
        return lift(u, v, frame.depth, frame.intrinsics, 0.0, self.pcfg.depth_window, self.pcfg.depth_percentile,
                    *self.pcfg.depth_range)

    # ------------------------------------------------------------------ per frame
    def _measure(self, pose, frame: RGBDFrame, index: int, radius: float) -> Measure:
        uv, vis = pose.uv[index], float(pose.vis[index])
        k = frame.intrinsics
        if not (0 <= uv[0] < k.width and 0 <= uv[1] < k.height):
            vis = 0.0          # a landmark outside the image is a guess, whatever its score says
        p = None
        if vis >= self.pcfg.min_visibility:
            p = lift(uv[0], uv[1], frame.depth, k, radius, self.pcfg.depth_window,
                     self.pcfg.depth_percentile, *self.pcfg.depth_range)
        clusters = None
        if p is not None and index in (WRIST['right'], WRIST['left']):
            clusters = depth_clusters(frame.depth, uv[0], uv[1], self.pcfg.depth_window, *self.pcfg.depth_range)
        return Measure(uv=uv, vis=vis, ray=pixel_ray(uv[0], uv[1], k), P=p, clusters=clusters)

    def process(self, frame: RGBDFrame) -> OperatorFrame:
        cfg, pcfg = self.cfg, self.pcfg
        t_start = time.perf_counter()
        t = frame.t_capture
        timing: Dict[str, float] = {}
        rgb = cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)
        self._handle_toggle(t)
        if self.stage in ('enter', 'hands'):                                  # no body tracking in these stages
            out = OperatorFrame(seq=self.seq, t_capture=t, mode=protocol.IDLE, bones=dict(self.profile.bones))
            self.seq += 1
            debug = {'pose': None, 'measures': {}, 'hands': {}, 'status': {}, 'intrinsics': frame.intrinsics}
            self.last_debug = debug
            self._process_hands(frame, rgb, out, timing, debug)
            out.control = self.control
            timing['total'] = (time.perf_counter() - t_start) * 1e3
            out.timing_ms = {k: round(v, 2) for k, v in timing.items()}
            return out
        pose = self.pose.process(rgb)
        timing['pose'] = (time.perf_counter() - t_start) * 1e3
        gravity_up = None if frame.gravity is None else -np.asarray(frame.gravity)

        out = OperatorFrame(seq=self.seq, t_capture=t, mode=protocol.IDLE,
                            bones=dict(self.profile.bones) if self.profile else {})
        self.seq += 1
        debug = {'pose': pose, 'measures': {}, 'hands': {}, 'status': {}, 'intrinsics': frame.intrinsics}
        self.last_debug = debug

        t1 = time.perf_counter()
        meas: Dict[str, Dict[str, Optional[Measure]]] = {s: {'S': None, 'E': None, 'W': None}
                                                           for s in self.sides}
        shoulder_meas: Dict[str, Optional[Measure]] = {'right': None, 'left': None}
        if pose is not None:
            r = pcfg.radii
            for s in ('right', 'left'):       # both shoulders always: they define the body frame
                shoulder_meas[s] = self._measure(pose, frame, SHOULDER[s], r.shoulder)
            for s in self.sides:
                meas[s]['S'] = shoulder_meas[s]
                meas[s]['E'] = self._measure(pose, frame, ELBOW[s], r.elbow)
                meas[s]['W'] = self._measure(pose, frame, WRIST[s], r.wrist)
        debug['measures'] = meas
        timing['lift'] = (time.perf_counter() - t1) * 1e3

        # both shoulders (right, left), one possibly predicted from the other
        sh = self._shoulders(shoulder_meas, t)

        if self.calibrating:
            out.mode = protocol.CALIBRATING
            self._calibrate(t, rgb, pose, frame, meas, sh, gravity_up, out, timing)
        elif self.profile is not None and self.profile.frame is not None:
            out.mode = protocol.TRACKING
            self._track(t, rgb, pose, frame, meas, sh, out, timing, debug)
        tracking = out.mode == protocol.TRACKING
        if self.stage == 'reach':
            self._last_wrists = {s: a.wrist.copy() for s, a in out.arms.items()
                                 if a.status == protocol.TRACKING and a.wrist is not None}
            if self._check_nod(t, pose, tracking and bool(self._last_wrists)):
                self._begin_enter(t, 'nod')
        elif self.stage == 'exit':
            self._match_exit(t, out, debug)
        out.control = self.control
        timing['total'] = (time.perf_counter() - t_start) * 1e3
        out.timing_ms = {k: round(v, 2) for k, v in timing.items()}
        return out

    # ------------------------------------------------------------------ shoulders
    def _gate_shoulder(self, side, p, t):
        """Reject a shoulder reading that moved faster than a body can (an arm reaching at the
        camera hides the shoulder behind it and the depth jumps to the arm)."""
        if p is None:
            return None
        prev, prev_t = self.sh_prev.get(side), self.sh_prev_t.get(side)
        if prev is not None:
            limit = self.pcfg.shoulder_speed * max(t - prev_t, 1e-3) + 0.04
            if np.linalg.norm(p - prev) > limit:
                self.sh_rejects[side] = self.sh_rejects.get(side, 0) + 1
                if self.sh_rejects[side] < 12:
                    return None
        self.sh_rejects[side] = 0
        self.sh_prev[side], self.sh_prev_t[side] = p.copy(), t
        return p

    def _shoulders(self, shoulder_meas, t):
        """(right, left, right_predicted, left_predicted) camera points, or None."""
        r_m, l_m = shoulder_meas['right'], shoulder_meas['left']
        r_sh = self._gate_shoulder('right', None if r_m is None else r_m.P, t)
        l_sh = self._gate_shoulder('left', None if l_m is None else l_m.P, t)
        width = self.shoulder_width
        if r_sh is not None and l_sh is not None and width is not None and not self.calibrating:
            if abs(np.linalg.norm(l_sh - r_sh) - width) > 0.25 * width:
                # inconsistent pair: keep the one that stayed closer to where it was
                dr = np.linalg.norm(r_sh - self.sh_prev.get('right', r_sh))
                dl = np.linalg.norm(l_sh - self.sh_prev.get('left', l_sh))
                if dr > dl:
                    r_sh = None
                else:
                    l_sh = None
        est = estimate_shoulders(r_sh, l_sh, self.left_vec, self.shoulder_width)
        if est is None:
            return None
        r_sh, l_sh, rp, lp = est
        if not (rp or lp):
            self.left_vec = unit(l_sh - r_sh)
            w = float(np.linalg.norm(l_sh - r_sh))
            self.shoulder_width = w if self.shoulder_width is None else 0.98 * self.shoulder_width + 0.02 * w
        return r_sh, l_sh, rp, lp

    # ------------------------------------------------------------------ calibration
    def _calibrate(self, t, rgb, pose, frame, meas, sh, gravity_up, out, timing):
        cal = self.calibrator
        pts = {s: {k: (None if meas[s][k] is None else meas[s][k].P) for k in 'SEW'} for s in self.sides}
        shoulders = None if sh is None or sh[2] or sh[3] else (sh[0], sh[1])
        up0 = gravity_up if gravity_up is not None else (
            self._up_guess if self._up_guess is not None else cal._default_up())
        hand_raw = {s: None for s in self.sides}
        if cal.stage and cal.stage.startswith('hand') and self.hands is not None and pose is not None:
            from g1tele.hand import raw_features
            for s in self.sides:
                world = self._detect_hand(rgb, pose, frame, s, meas)
                if world is not None:
                    hand_raw[s] = raw_features(world[0] * np.asarray(self.pcfg.hand_axes_sign))
        floor_up = None
        if gravity_up is None and self.seq % 6 == 0:
            res = estimate_up_from_floor(frame.depth, frame.intrinsics, up0, self._rng)
            floor_up = None if res is None else res[0]
            if floor_up is not None:     # the pose checks below use the best "up" known so far
                self._up_guess = floor_up if self._up_guess is None else unit(0.7 * self._up_guess + 0.3 * floor_up)
        out.prompt = cal.update(t, pts, shoulders, up0, hand_raw, gravity_up, floor_up)
        if self.notice and t < self.notice_until:
            out.prompt = f'{self.notice} | {out.prompt}'
        if cal.done:
            self.profile = cal.profile
            self._apply_profile_hands(self.profile)
            if self.profile.frame is None:
                # shoulders were never seen both: cannot continue; try the rest pose again
                self.start_calibration(full=False)
                return
            problems = validate_profile(self.profile, self.sides)
            if problems:
                self.rejections += 1
                log.warning('calibration rejected: %s', '; '.join(problems))
                self.notice, self.notice_until = f'Calibration rejected: {problems[0]}. Doing it again.', t + 10.0
                self.profile = None
                self.start_calibration(full=True)
                out.prompt = self.notice
                return
            self.shoulder_width = self.profile.shoulder_width or self.shoulder_width
            try:
                self.profile.save(self.cfg.calibration.profile_dir, self.cfg.calibration.operator)
            except OSError:
                pass
            out.prompt = 'Calibration done.'
            out.bones = dict(self.profile.bones)

    # ------------------------------------------------------------------ hands
    def _detect_hand(self, rgb, pose, frame, side, meas):
        """(world landmarks, image points) of THIS side's hand, or None.

        Two hands close together (both on the table, or holding one object) share a crop, and the model
        may return either or both: the one whose wrist is at this side's wrist pixel is kept, and a hand
        that sits closer to the other wrist is refused. A left hand read as a right one would turn the
        palm normal around: thumbs up becomes thumbs down."""
        wm = meas[side]['W']
        if wm is None or wm.P is None or wm.vis < 0.3:
            return None
        pts = [pose.uv[i] for i in HAND_GUIDE[side] if pose.vis[i] > 0.3]
        if not pts:
            return None
        center = np.mean(pts, axis=0)
        hand_px = HAND_LENGTH * frame.intrinsics.fx / wm.P[2]
        detector = self.hands[side]
        found = detector.detect_all(rgb, center, hand_px) if hasattr(detector, 'detect_all') else \
            [r for r in [detector.detect(rgb, center, hand_px)] if r is not None]
        if not found:
            return None
        own = pose.uv[WRIST[side]]
        other_side = 'left' if side == 'right' else 'right'
        other = pose.uv[WRIST[other_side]] if pose.vis[WRIST[other_side]] > 0.3 else None
        best, best_d = None, 1e9
        for world, image in found:
            d_own = float(np.linalg.norm(image[0] - own))
            if other is not None and float(np.linalg.norm(image[0] - other)) < d_own:
                continue                                   # that is the other hand
            if d_own < best_d:
                best, best_d = (world, image), d_own
        if best is None or best_d > 0.9 * hand_px + 8.0:
            return None
        self.hand_points[side] = best[1]
        return best

    def _hand_depth_cues(self, frame, found, side, ht, frame_rot):
        """(palm cue, hidden landmarks) of one hand from the depth image."""
        pc = self.pcfg
        world, image = found
        lo, hi = pc.depth_range
        prev = None if ht.R is None else frame_rot @ ht.R[:, 2]
        n_lm = hand_rotation(world * np.asarray(pc.hand_axes_sign), side)
        cue = palm_normal_from_depth(frame.depth, frame.intrinsics, image, side, lo, hi, prev,
                                     None if n_lm is None else n_lm[:, 2]) if pc.hand_depth else None
        hidden = None
        if pc.hand_occlusion and cue is not None:
            hidden = occluded_landmarks(frame.depth, image, world * np.asarray(pc.hand_axes_sign), cue.plane,
                                        frame.intrinsics, pc.occ_front_margin, lo)
        return cue, hidden

    # ------------------------------------------------------------------ tracking
    def _track(self, t, rgb, pose, frame, meas, sh, out, timing, debug):
        frame_rot = self.profile.frame
        t0 = time.perf_counter()
        # hands run in parallel threads (MediaPipe releases the GIL while it infers)
        futures = {}
        if self.hands is not None and pose is not None and self.pcfg.reach_hands:
            for s in self.sides:
                if self._pool is not None:
                    futures[s] = self._pool.submit(self._detect_hand, rgb, pose, frame, s, meas)
        for s in self.sides:
            tr = self.trackers[s]
            shoulder = None
            if sh is not None:
                shoulder = sh[0] if s == 'right' else sh[1]
            m = meas[s]
            obs = tr.update(t, shoulder, m['E'], m['W'], frame_rot, self.profile.bones.get(s),
                            frame.intrinsics.fx)
            out.arms[s] = obs
            debug['status'][s] = obs.status
            debug.setdefault('fused', {})[s] = tr.fused_points if obs.status == protocol.TRACKING else None
        timing['track'] = (time.perf_counter() - t0) * 1e3
        reason = self.monitor.update(t, self.profile, {s: self.trackers[s].shoulder_cam for s in self.sides},
                                     meas, frame)
        if reason is not None and self.cfg.calibration.auto_recalibrate:
            text = ('the arm lengths no longer match the calibration' if reason == 'bones'
                    else 'the camera seems to have moved')
            log.warning('recalibrating: %s', text)
            self.drift_events.append((t, reason))
            self.notice, self.notice_until = f'Recalibrating: {text}.', t + 15.0
            self.start_calibration(full=(reason == 'bones'))
            out.mode = protocol.CALIBRATING
            out.arms = {}
            out.prompt = self.notice
            return
        if self.monitor.status != 'ok':
            out.prompt = 'WARNING: calibration may be off (press v to redo the rest pose)'
        t1 = time.perf_counter()
        for s, fut in futures.items():
            found = fut.result()
            ht = self.hand_tracks[s]
            obs = out.arms[s]
            if found is not None:
                arm_dir = None
                fp = self.trackers[s].fused_points
                if fp is not None:
                    _, e3, w3 = fp
                    ref = e3 if (e3 is not None and obs.elbow_quality > 0.5) else fp[0]
                    arm_dir = unit(w3 - ref)
                gi = HAND_GUIDE[s]
                palm_2d = (pose.uv[gi[2]] + pose.uv[gi[1]]) / 2 - pose.uv[gi[0]] if pose is not None else None
                ok2d = pose is not None and min(pose.vis[gi[1]], pose.vis[gi[2]]) > 0.3
                palm, hidden = self._hand_depth_cues(frame, found, s, ht, frame_rot)
                ht.update(t, found[0], frame_rot, arm_dir, palm_2d if ok2d else None,
                          (found[1][9] - found[1][0]) if ok2d else None, palm, hidden)
                debug.setdefault('hand_reject', {})[s] = ht.reject_reason
                debug.setdefault('hand_palm_source', {})[s] = ht.palm_source
                debug.setdefault('hand_held', {})[s] = ht.held.copy()
            if ht.fresh(t) and obs.status in (protocol.TRACKING, protocol.HELD):
                obs.hand_R = ht.R.copy()
                obs.fingers = ht.fingers.copy()
                obs.tips = None if ht.tips is None else ht.tips.copy()
                obs.hand_quality = float(ht.quality)
            debug['hands'][s] = self.hand_points.get(s) if found is not None else None
            debug.setdefault('hand_axes', {})[s] = None if obs.hand_R is None else frame_rot @ ht.R
        timing['hands'] = (time.perf_counter() - t1) * 1e3
        if self.waist is not None and sh is not None and not (sh[2] or sh[3]):
            out.waist = self.waist.update(t, sh[0], sh[1], frame_rot)
        out.bones = dict(self.profile.bones)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        closed = set()
        for est in [self.pose] + list((self.hands or {}).values()):
            if est is not None and id(est) not in closed and hasattr(est, 'close'):
                closed.add(id(est))
                est.close()
