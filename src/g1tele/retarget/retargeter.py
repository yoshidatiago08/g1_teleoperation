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
from g1tele.geometry import unit, waist_matrix
from g1tele.hand import robot_hand_in_link
from g1tele.protocol import ArmObs, OperatorFrame
from g1tele.retarget.arm_ik import ArmIK, ArmKinematics
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
        self.last_time: Optional[float] = None
        self.status = {s: protocol.IDLE for s in self.sides}
        self.ik_error = {s: 0.0 for s in self.sides}
        self.last_seq = -1
        self.rot_conf = {s: 0.0 for s in self.sides}      # how much the hand orientation is trusted, 0..1
        self.hand_R_last = {s: None for s in self.sides}

    # ------------------------------------------------------------------ mapping
    def _targets(self, side: str, obs: ArmObs, bones, r_waist):
        """(wrist target, elbow target) of the operator's arm, in the robot's torso frame."""
        kin = self.kin[side]
        upper_h, fore_h = (bones.upper, bones.fore) if bones is not None and bones.valid() \
            else DEFAULT_BONES
        mode = self.cfg.ik.mapping
        s = kin.shoulder
        wrist, elbow = obs.wrist, obs.elbow
        if mode == 'bone_scaled' and elbow is not None:
            u = unit(elbow)
            f = unit(wrist - elbow)
            if u is not None and f is not None:
                e_t = s + kin.upper_arm * u
                w_t = e_t + kin.forearm * f
                return self._to_torso(s, w_t, r_waist), self._to_torso(s, e_t, r_waist)
        if mode == 'metric':
            scale = 1.0
        else:
            scale = (kin.upper_arm + kin.forearm) / (upper_h + fore_h)
        w_t = s + scale * wrist
        e_t = None if elbow is None else s + scale * elbow
        reach = kin.upper_arm + kin.forearm
        d = w_t - s
        n = np.linalg.norm(d)
        if n > 0.99 * reach:      # keep the target reachable: the IK would otherwise stretch toward it
            w_t = s + d * (0.99 * reach / n)
        return (self._to_torso(s, w_t, r_waist),
                None if e_t is None else self._to_torso(s, e_t, r_waist))

    @staticmethod
    def _to_torso(shoulder, p, r_waist):
        """Point in the zero-waist (upright) torso frame -> torso_link, so the hand stays where the
        fixed-frame target says even when the waist turns. Only the arm vector is rotated."""
        return shoulder + r_waist.T @ (p - shoulder)

    # ------------------------------------------------------------------ main step
    def step(self, frame: Optional[OperatorFrame], now: Optional[float] = None) -> RobotTargets:
        """One control update. `frame` is the newest operator frame, or None when nothing new has
        arrived (the targets are then held or eased to rest)."""
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.last_time is None else float(np.clip(now - self.last_time, 0.0, 0.2))
        self.last_time = now
        cfg = self.cfg
        r_waist = waist_matrix(self.waist['waist_yaw_joint'], self.waist['waist_roll_joint'],
                               self.waist['waist_pitch_joint'])
        mode = protocol.IDLE if frame is None else frame.mode

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
            ik = self.ik[side]
            if (frame is not None and mode == protocol.TRACKING and obs is not None
                    and obs.status in (protocol.TRACKING, protocol.HELD) and obs.wrist is not None):
                bones = frame.bones.get(side)
                w_t, e_t = self._targets(side, obs, bones, r_waist)
                # Hand orientation: fade it in and out instead of switching, and hold the last one
                # briefly when the hand model drops a frame (a flickering detection must not make
                # the robot's wrist snap between "follow the palm" and "ignore it").
                if obs.hand_R is not None and obs.hand_quality > 0.3:
                    self.hand_R_last[side] = obs.hand_R
                    self.rot_conf[side] = min(1.0, self.rot_conf[side] + dt / 0.4)
                else:
                    self.rot_conf[side] = max(0.0, self.rot_conf[side] - dt / 0.6)
                r_t = None
                if self.hand_R_last[side] is not None and self.rot_conf[side] > 0.0:
                    r_t = r_waist.T @ self.hand_R_last[side] @ self.hand_in_link[side].T
                q_before = ik.q.copy()
                res = ik.solve(w_t, r_t, e_t, rot_scale=self.rot_conf[side])
                q = q_before + self._clamp_speed(res.q - q_before, dt)
                ik.q = q
                self.ik_error[side] = res.pos_error
                self.status[side] = obs.status
                if obs.fingers is not None:
                    self.fingers[side] = np.clip(obs.fingers, 0.0, 1.0)
            else:
                # No usable data (lost, calibrating, idle): ease back to the rest pose
                target = ik.q_rest
                q_before = ik.q.copy()
                ik.q = q_before + self._clamp_speed(target - q_before, dt, factor=0.5)
                self.status[side] = obs.status if obs is not None else mode
                self.fingers[side] = self.fingers[side] * (1.0 - min(1.0, 3.0 * dt))

        out = RobotTargets(status=dict(self.status), ik_error=dict(self.ik_error),
                           t_capture=0.0 if frame is None else frame.t_capture,
                           seq=-1 if frame is None else frame.seq, mode=mode)
        for side in self.sides:
            for name, value in zip(self.kin[side].names, self.ik[side].q):
                out.arm_q[name] = float(value)
            out.hand[side] = self.fingers[side].copy()
        out.waist_q = dict(self.waist)
        return out

    def _clamp_speed(self, delta, dt, factor=1.0):
        limit = self.cfg.robot.max_joint_speed * factor * dt
        if dt <= 0.0:
            return delta
        return np.clip(delta, -limit, limit)
