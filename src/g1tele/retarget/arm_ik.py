"""Damped least-squares IK for one 7-DoF G1 arm.

Everything is analytic numpy: forward kinematics and Jacobians of a 7-joint chain are tiny, so a
full solve takes well under a millisecond and needs no optimiser library. The solver is warm
started from the previous solution, which also keeps consecutive frames continuous.

Tasks (weights in IKCfg): wrist position, elbow position (chooses the arm posture: a 7-DoF arm
can reach a wrist target with the elbow in many places), wrist orientation, closeness to the
previous solution, and a faint pull toward a relaxed posture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from g1tele.config import IKCfg
from g1tele.geometry import rotvec
from g1tele.retarget.urdf import Urdf, arm_joint_names


def _rodrigues(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


@dataclass
class IKResult:
    q: np.ndarray
    pos_error: float        # m, wrist
    elbow_error: float      # m (0 if no elbow target)
    rot_error: float        # rad (0 if no orientation target)
    iterations: int


class ArmKinematics:
    """Kinematic chain of one arm, from the torso link to the wrist link."""

    def __init__(self, urdf: Urdf, side: str, base: str = 'torso_link'):
        self.side = side
        chain = urdf.chain(base, f'{side}_wrist_yaw_link')
        movable = urdf.movable(chain)
        expected = arm_joint_names(side)
        if [j.name for j in movable] != expected:
            raise RuntimeError(f'{side} arm: expected joints {expected}, URDF has {[j.name for j in movable]}')
        # Fold fixed joints into the next movable joint's origin; trailing ones become the tip offset
        self.origins: List[np.ndarray] = []
        pending = np.eye(4)
        for j in chain:
            pending = pending @ j.origin
            if j.type in ('revolute', 'continuous', 'prismatic'):
                self.origins.append(pending)
                pending = np.eye(4)
        self.tip_offset = pending
        self.names = expected
        self.axes = [j.axis for j in movable]
        self.lower = np.array([j.lower for j in movable])
        self.upper = np.array([j.upper for j in movable])
        self.n = len(movable)
        self.elbow_idx = self.names.index(f'{side}_elbow_joint')
        # The robot's "wrist" is the origin of the wrist-pitch joint: it sits on the wrist-roll
        # axis, so it depends only on the shoulder and elbow joints and its distance to the elbow
        # is constant. The wrist joints then only decide the hand's orientation.
        self.wrist_idx = self.names.index(f'{side}_wrist_pitch_joint')
        zero = self.fk(np.zeros(self.n))
        self.rest_wrist = zero[0]
        self.forearm = float(np.linalg.norm(zero[0] - zero[2]))
        self.shoulder, self.upper_arm = self._fit_shoulder_sphere()

    def _fit_shoulder_sphere(self):
        """Centre and radius of the sphere the elbow moves on when only the three shoulder joints
        turn. The G1's shoulder axes do not meet in one point, so there is no exact centre; the
        best-fit sphere is the "shoulder" the operator's upper arm is mapped onto."""
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(400):
            q = np.zeros(self.n)
            q[:3] = rng.uniform(0.5 * self.lower[:3], 0.5 * self.upper[:3])
            pts.append(self.fk(q)[2])
        pts = np.array(pts)
        a = np.hstack([2 * pts, np.ones((len(pts), 1))])
        b = (pts ** 2).sum(axis=1)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        center = x[:3]
        radius = float(np.sqrt(x[3] + center @ center))
        return center, radius

    def fk(self, q):
        """(wrist position, hand-link rotation, elbow position, joint positions (n,3),
        joint axes (n,3), hand-link position)."""
        t = np.eye(4)
        pos = np.empty((self.n, 3))
        axes = np.empty((self.n, 3))
        for i in range(self.n):
            t = t @ self.origins[i]
            pos[i] = t[:3, 3]
            axes[i] = t[:3, :3] @ self.axes[i]
            t[:3, :3] = t[:3, :3] @ _rodrigues(self.axes[i], q[i])
        t = t @ self.tip_offset
        return (pos[self.wrist_idx].copy(), t[:3, :3].copy(), pos[self.elbow_idx].copy(), pos, axes,
                t[:3, 3].copy())


class ArmIK:
    def __init__(self, kin: ArmKinematics, cfg: IKCfg, q_rest: Optional[np.ndarray] = None):
        self.kin = kin
        self.cfg = cfg
        # A relaxed posture: slightly bent elbow, arm down. Only a faint pull in the null space.
        self.q_rest = np.zeros(kin.n) if q_rest is None else q_rest
        self.q = np.clip(self.q_rest.copy(), kin.lower, kin.upper)

    def reset(self, q=None):
        self.q = np.clip(self.q_rest.copy() if q is None else q, self.kin.lower, self.kin.upper)

    def solve(self, p_target, r_target=None, elbow_target=None, rot_scale: float = 1.0) -> IKResult:
        cfg, kin = self.cfg, self.kin
        q = self.q.copy()
        q_prev = self.q.copy()
        n = kin.n
        free = np.ones(n, dtype=bool)
        iters = 0
        for iters in range(1, cfg.iterations + 1):
            p, r, p_el, pos, axes, _ = kin.fk(q)
            rows, res = [], []
            # wrist position (only the joints before the wrist move it)
            j_p = np.cross(axes, (p - pos)).T
            j_p[:, kin.wrist_idx:] = 0.0
            rows.append(cfg.pos_weight * j_p)
            res.append(cfg.pos_weight * (p_target - p))
            if elbow_target is not None and cfg.elbow_weight > 0:
                j_e = np.zeros((3, n))
                k = kin.elbow_idx
                j_e[:, :k] = np.cross(axes[:k], (p_el - pos[:k])).T
                rows.append(cfg.elbow_weight * j_e)
                res.append(cfg.elbow_weight * (elbow_target - p_el))
            w_rot = cfg.rot_weight * rot_scale
            if r_target is not None and w_rot > 1e-6:
                rows.append(w_rot * axes.T)
                res.append(w_rot * rotvec(r_target @ r.T))
            jac = np.vstack(rows)
            err = np.concatenate(res)
            a = jac.T @ jac + (cfg.damping + cfg.smooth_weight + cfg.posture_weight) * np.eye(n)
            b = jac.T @ err + cfg.smooth_weight * (q_prev - q) + cfg.posture_weight * (self.q_rest - q)
            dq = np.zeros(n)
            # Joints at a limit that would be pushed further out are frozen, then re-solved.
            for _ in range(3):
                idx = np.where(free)[0]
                dq[:] = 0.0
                dq[idx] = np.linalg.solve(a[np.ix_(idx, idx)], b[idx])
                at_lo = (q <= kin.lower + 1e-6) & (dq < 0)
                at_hi = (q >= kin.upper - 1e-6) & (dq > 0)
                stuck = (at_lo | at_hi) & free
                if not stuck.any():
                    break
                free &= ~stuck
            free[:] = True
            step = np.max(np.abs(dq))
            if step > cfg.max_step:
                dq *= cfg.max_step / step
            q = np.clip(q + dq, kin.lower, kin.upper)
            if step < 1e-5:
                break
        p, r, p_el, _, _, _ = kin.fk(q)
        self.q = q
        return IKResult(
            q=q.copy(),
            pos_error=float(np.linalg.norm(p_target - p)),
            elbow_error=0.0 if elbow_target is None else float(np.linalg.norm(elbow_target - p_el)),
            rot_error=0.0 if r_target is None else float(np.linalg.norm(rotvec(r_target @ r.T))),
            iterations=iters)
