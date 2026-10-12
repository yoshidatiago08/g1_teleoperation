"""Operator state -> robot joint targets.

Runs next to the simulator. It knows the robot (from the URDF) and nothing about cameras.

Mapping of the operator's arm onto the robot's (IKCfg.mapping):
    bone_scaled  Each bone keeps its DIRECTION and takes the robot's length: the robot's elbow sits
                 where your upper arm points, its wrist where your forearm points. The posture
                 is copied exactly and a fully stretched arm is a fully stretched robot arm.
                 Without an elbow reading, falls back to `uniform`.
    uniform      The whole arm is scaled by (robot reach / your reach).
    metric       No scaling: your wrist position, clamped to the robot's reach. Use it only if the
                 robot's arm is about as long as yours.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from g1tele import protocol
from g1tele.config import Config
from g1tele.geometry import from_rotvec, rotvec, unit, waist_matrix
from g1tele.hand import joint_to_curl, robot_hand_in_link
from g1tele.protocol import ArmObs, OperatorFrame
from g1tele.perception.fusion import circle_point
from g1tele.retarget.arm_ik import ArmIK, ArmKinematics
from g1tele.retarget.hand_ik import HandIK
from g1tele.retarget.urdf import WAIST_JOINTS, Urdf, hand_joint_names

# Human bone lengths assumed when the operator has not been calibrated (m)
DEFAULT_BONES = (0.30, 0.27)


@dataclass
class RobotTargets:
    """What the backends apply. Joint values by URDF joint name."""
    arm_q: Dict[str, float] = field(default_factory=dict)     # 14 arm joints
    waist_q: Dict[str, float] = field(default_factory=dict)   # 3 waist joints
    hand: Dict[str, np.ndarray] = field(default_factory=dict)  # side -> 6 curls (0 open ... 1 closed)
    status: Dict[str, str] = field(default_factory=dict)       # side -> TRACKING | HELD | LOST | ...
    ik_error: Dict[str, float] = field(default_factory=dict)   # side -> wrist position error (m)
    t_capture: float = 0.0                                    # when the camera frame was captured
    seq: int = 0
    mode: str = protocol.IDLE


class Retargeter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.urdf = Urdf(cfg.robot.urdf)
        self.sides = list(cfg.perception.arms)
        self.kin = {s: ArmKinematics(self.urdf, s) for s in self.sides}
        self.ik = {s: ArmIK(self.kin[s], cfg.ik) for s in self.sides}
        self.hand_in_link = {s: robot_hand_in_link(s) for s in self.sides}
        self.waist_limits = {n: (self.urdf.joints[n].lower, self.urdf.joints[n].upper)
                             for n in WAIST_JOINTS if n in self.urdf.joints}
        self.waist = {n: 0.0 for n in WAIST_JOINTS}
        self.hand_names = {s: hand_joint_names(s) for s in self.sides}
        self.fingers = {s: np.zeros(6) for s in self.sides}
        self.hand_ik = {s: HandIK(self.urdf, s, cfg.ik.hand_scale, cfg.ik.hand_huber, cfg.ik.hand_reg)
                        for s in self.sides} if cfg.ik.hand_mode == 'vector' else {}
        self.hand_q = {s: np.zeros(6) for s in self.sides}
        self.last_time: Optional[float] = None
        self.status = {s: protocol.IDLE for s in self.sides}
        self.ik_error = {s: 0.0 for s in self.sides}
        self.last_seq = -1
        self.rot_conf = {s: 0.0 for s in self.sides}      # how much the hand orientation is trusted, 0..1
        self.hand_R_last = {s: None for s in self.sides}
        self.R_target = {s: None for s in self.sides}      # the wrist-orientation target, speed limited
        self.was_tracking = {s: False for s in self.sides}
        self.ease = {s: np.zeros(3) for s in self.sides}   # offset that fades out after the arm (re)engages
        self.ease_f = {s: 0.0 for s in self.sides}         # 1 at engage, fading to 0
        self.lost_since = {s: None for s in self.sides}
        self.control = protocol.REACH
        self.clutch = {s: None for s in self.sides}        # hands-only mode: the operator's and the robot's hand at the switch
        self._compute_rest()

    def _compute_rest(self):
        """The starting pose: arms hanging at the sides, hands pointing down, away from the table."""
        for side in self.sides:
            kin, ik = self.kin[side], self.ik[side]
            sign = -1.0 if side == 'right' else 1.0
            w = kin.shoulder + np.array([0.03, sign * 0.05, -0.93 * (kin.upper_arm + kin.forearm)])
            x_axis = np.array([0.0, 0.0, -1.0])                 # fingers down
            z_axis = np.array([0.0, -sign, 0.0])                # palm toward the body
            r_hand = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
            r_link = r_hand @ self.hand_in_link[side].T
            ik.reset(np.zeros(kin.n))
            res = None
            for _ in range(40):
                res = ik.solve(w, r_link, None)
            ik.q_rest = res.q.copy()
            ik.q = res.q.copy()

    def snapshot(self, mode=protocol.IDLE) -> 'RobotTargets':
        """The current joint targets, without advancing anything (used to place the robot at start)."""
        out = RobotTargets(status=dict(self.status), ik_error=dict(self.ik_error), mode=mode)
        for side in self.sides:
            for name, value in zip(self.kin[side].names, self.ik[side].q):
                out.arm_q[name] = float(value)
            out.hand[side] = self.fingers[side].copy()
        out.waist_q = dict(self.waist)
        return out

    # ------------------------------------------------------------------ mapping
    def _targets(self, side: str, obs: ArmObs, bones, r_waist):
        """(wrist target, elbow hint or None, elbow trust 0..1) for the operator's arm, in torso_link coordinates.

        The wrist target blends two mappings by how far the elbow can be trusted (obs.elbow_quality):
          bone_scaled  wrist = elbow + forearm * direction of YOUR forearm: copies the posture exactly, but
                       inherits every error of the elbow estimate
          uniform      wrist = shoulder + (robot reach / your reach) * YOUR wrist vector: depends only on
                       the wrist, which is what the hand actually needs
        With the forearm pointing at the camera the elbow is a guess, so the wrist comes from the uniform
        mapping alone. The elbow target is only ever a hint for the swivel (see step()).
        """
        kin = self.kin[side]
        upper_h, fore_h = (bones.upper, bones.fore) if bones is not None and bones.valid() \
            else DEFAULT_BONES
        mode = self.cfg.ik.mapping
        s = kin.shoulder
        wrist, elbow = obs.wrist, obs.elbow
        reach = kin.upper_arm + kin.forearm
        scale = 1.0 if mode == 'metric' else reach / (upper_h + fore_h)
        w_uniform = s + scale * wrist
        e_uniform = None if elbow is None else s + scale * elbow
        w_t, e_hint, trust = w_uniform, e_uniform, (0.0 if elbow is None else float(obs.elbow_quality))
        if mode == 'bone_scaled' and elbow is not None:
            u, f = unit(elbow), unit(wrist - elbow)
            if u is not None and f is not None:
                e_bone = s + kin.upper_arm * u
                w_bone = e_bone + kin.forearm * f
                w_t = trust * w_bone + (1.0 - trust) * w_uniform
                e_hint = e_bone
        d = w_t - s
        n = np.linalg.norm(d)
        if n > 0.99 * reach:      # keep the target reachable: the IK would otherwise stretch toward it
            w_t = s + d * (0.99 * reach / n)
        to_torso = lambda p: None if p is None else self._to_torso(s, p, r_waist)
        return to_torso(w_t), to_torso(e_hint), trust

    @staticmethod
    def _to_torso(shoulder, p, r_waist):
        """Point in the zero-waist (upright) torso frame -> torso_link, so the hand stays where the
        fixed-frame target says even when the waist turns. Only the arm vector is rotated."""
        return shoulder + r_waist.T @ (p - shoulder)

    def _hands_only(self, side: str, obs: ArmObs, r_waist):
        """Hands-only mode: the arm stays where it was; the wrist turns and the fingers move as your hand does.

        At the first frame the operator's hand and the robot's hand are paired (a clutch). From then on the robot's
        wrist is turned by the SAME rotation your hand has turned since, whatever its absolute orientation: an
        error in the hand model's absolute orientation does not matter, only changes of it do. The wrist moves
        by ik.hands_gain times your hand's movement (0: the arm holds still).
        Returns (wrist target, elbow hint, trust, obs) with obs.hand_R replaced by the hand frame that gives that
        wrist orientation through the normal orientation path."""
        kin, ik = self.kin[side], self.ik[side]
        c = self.clutch[side]
        if c is None:
            c = self.clutch[side] = dict(op_R=obs.hand_R.copy() if obs.hand_R is not None else None,
                                         op_p=obs.wrist.copy(), w=kin.fk(ik.q)[0], R=kin.fk(ik.q)[1])
        if c['op_R'] is None and obs.hand_R is not None:
            c['op_R'] = obs.hand_R.copy()
        w_t = c['w'] + self.cfg.ik.hands_gain * (obs.wrist - c['op_p'])
        if obs.hand_R is not None and c['op_R'] is not None:
            r_link = (obs.hand_R @ c['op_R'].T) @ c['R']            # the robot's wrist link, turned like the hand
            obs = ArmObs(status=obs.status, quality=obs.quality, wrist=obs.wrist,
                         hand_R=r_waist @ r_link @ self.hand_in_link[side], fingers=obs.fingers,
                         hand_quality=obs.hand_quality, tips=obs.tips)
        return w_t, None, 0.0, obs

    # ------------------------------------------------------------------ main step
    def step(self, frame: Optional[OperatorFrame], now: Optional[float] = None) -> RobotTargets:
        """One control update. `frame` is the newest operator frame, or None when nothing new has arrived.

        Behaviour when an arm has no usable data (calibrating, idle, lost): the arm HOLDS where it is. It
        used to be sent back to the rest pose, which swept it across the table; now it only drifts slowly
        to rest after robot.lost_rest_after seconds. When an arm (re)engages it starts from where it is
        and the offset to your pose fades out (ik.engage_tau), so it never jumps.
        """
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.last_time is None else float(np.clip(now - self.last_time, 0.0, 0.2))
        self.last_time = now
        cfg = self.cfg
        r_waist = waist_matrix(self.waist['waist_yaw_joint'], self.waist['waist_roll_joint'],
                               self.waist['waist_pitch_joint'])
        mode = protocol.IDLE if frame is None else frame.mode
        control = protocol.REACH if frame is None else frame.control
        if control != self.control:                        # a mode change: nothing carries over
            self.control = control
            for side in self.sides:
                self.clutch[side] = None
                self.was_tracking[side] = False            # the arm re-engages smoothly (ik.engage_tau)
                self.R_target[side] = None
                self.hand_R_last[side] = None
                self.rot_conf[side] = 0.0

        waist_goal = {n: 0.0 for n in WAIST_JOINTS}
        if frame is not None and frame.waist is not None and mode == protocol.TRACKING:
            for n, v in zip(WAIST_JOINTS, frame.waist):
                if n in self.waist_limits:
                    lo, hi = self.waist_limits[n]
                    waist_goal[n] = float(np.clip(v, lo, hi))
        for n in WAIST_JOINTS:
            step = cfg.robot.max_waist_speed * dt
            self.waist[n] += float(np.clip(waist_goal[n] - self.waist[n], -step, step))

        for side in self.sides:
            obs = None if frame is None else frame.arms.get(side)
            ik, kin = self.ik[side], self.kin[side]
            if (frame is not None and mode == protocol.TRACKING and obs is not None
                    and obs.status in (protocol.TRACKING, protocol.HELD) and obs.wrist is not None):
                self.lost_since[side] = None
                bones = frame.bones.get(side)
                hands_only = control == protocol.HANDS
                if hands_only:
                    w_t, e_hint, trust, obs = self._hands_only(side, obs, r_waist)
                else:
                    w_t, e_hint, trust = self._targets(side, obs, bones, r_waist)
                if not self.was_tracking[side]:
                    self.ease[side] = kin.fk(ik.q)[0] - w_t        # start from where the hand is now
                    self.ease_f[side] = 1.0
                self.was_tracking[side] = True
                decay = float(np.exp(-dt / cfg.ik.engage_tau))
                self.ease[side] = self.ease[side] * decay
                self.ease_f[side] *= decay
                w_t = w_t + self.ease[side]
                reach = kin.upper_arm + kin.forearm
                d = w_t - kin.shoulder
                if np.linalg.norm(d) > 0.99 * reach:
                    w_t = kin.shoulder + d * (0.99 * reach / np.linalg.norm(d))
                # The elbow target lies on the circle the bone lengths allow for THIS wrist target: it
                # can only choose the swivel, so it can never pull the wrist away.
                if e_hint is not None and self.ease_f[side] > 1e-3:      # the swivel eases in too
                    e_hint = self.ease_f[side] * kin.fk(ik.q)[2] + (1.0 - self.ease_f[side]) * e_hint
                e_t = None if e_hint is None else circle_point(kin.shoulder, w_t, kin.upper_arm, kin.forearm, e_hint)
                # Hand orientation: fade it in and out instead of switching, and hold the last one
                # briefly when the hand model drops a frame.
                if obs.hand_R is not None and obs.hand_quality > 0.3:
                    self.hand_R_last[side] = obs.hand_R
                    self.rot_conf[side] = min(1.0, self.rot_conf[side] + dt / 0.4)
                else:
                    self.rot_conf[side] = max(0.0, self.rot_conf[side] - dt / 0.6)
                r_t = None
                if self.hand_R_last[side] is not None and self.rot_conf[side] > 0.0:
                    r_new = r_waist.T @ self.hand_R_last[side] @ self.hand_in_link[side].T
                    cur = self.R_target[side]
                    if cur is None:
                        cur = kin.fk(ik.q)[1]                   # start from the hand's current orientation
                    delta = rotvec(cur.T @ r_new)
                    limit = cfg.robot.max_rot_speed * (dt if dt > 0 else 1.0 / 30.0)
                    n = float(np.linalg.norm(delta))
                    if n > limit:
                        delta *= limit / n
                    r_t = cur @ from_rotvec(delta)
                    self.R_target[side] = r_t
                q_before = ik.q.copy()
                res = ik.solve(w_t, r_t, e_t, rot_scale=self.rot_conf[side] * float(np.clip(obs.hand_quality, 0.0, 1.0)),
                               elbow_scale=trust)
                ik.q = q_before + self._clamp_speed(res.q - q_before, dt)
                self.ik_error[side] = res.pos_error
                self.status[side] = obs.status
                if side in self.hand_ik and obs.tips is not None:
                    self.hand_q[side] = self.hand_ik[side].solve(obs.tips, self.hand_q[side])
                    self.fingers[side] = joint_to_curl(self.hand_q[side])
                elif obs.fingers is not None:
                    self.fingers[side] = np.clip(obs.fingers, 0.0, 1.0)
                    self.hand_q[side] = HandIK.joint_from_curl(self.fingers[side])
            else:
                # No usable data: HOLD. Only after a while does the arm drift slowly back to the rest pose.
                self.was_tracking[side] = False
                self.rot_conf[side] = 0.0
                self.R_target[side] = None
                if self.lost_since[side] is None:
                    self.lost_since[side] = now
                if (mode == protocol.TRACKING or frame is None) and now - self.lost_since[side] > cfg.robot.lost_rest_after:
                    q_before = ik.q.copy()
                    ik.q = q_before + np.clip(ik.q_rest - q_before, -cfg.robot.rest_speed * dt, cfg.robot.rest_speed * dt)
                self.status[side] = obs.status if obs is not None else mode

        out = self.snapshot(mode)
        out.status, out.ik_error = dict(self.status), dict(self.ik_error)
        out.t_capture = 0.0 if frame is None else frame.t_capture
        out.seq = -1 if frame is None else frame.seq
        return out

    def _clamp_speed(self, delta, dt, factor=1.0):
        limit = self.cfg.robot.max_joint_speed * factor * (dt if dt > 0.0 else 1.0 / 30.0)
        return np.clip(delta, -limit, limit)
