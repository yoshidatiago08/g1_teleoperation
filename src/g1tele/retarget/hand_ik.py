"""Fingers: the operator's fingertip positions -> the six Inspire actuator angles.

This is the "vector" retargeting of dex-retargeting (Qin et al., "AnyTeleop"), written in numpy on our own
URDF reader so nothing else has to be installed. The operator's wrist -> fingertip vectors (in the hand
frame, in palm lengths) are scaled to the robot's palm, and the actuator angles are found so that the robot's
own fingertips, through the real finger kinematics (including the `<mimic>` joints and the thumb's two
axes), end up on those vectors. Compared with mapping a flexion sum to an angle linearly this:
  * follows the thumb opposition, which is a position, not a curl;
  * is steady when one landmark is bad (a Huber loss, not a square);
  * cannot ask for an impossible hand (joint limits are hard bounds, the solution is the closest reachable).

Solver: damped Gauss-Newton with analytic Jacobians and iteratively reweighted Huber, warm started from the
previous angles, so a solve is a fraction of a millisecond.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from g1tele.hand import HAND_JOINT_RANGE, TIP_ORDER, curl_to_joint, robot_hand_in_link
from g1tele.retarget.urdf import HAND_PREFIX, Urdf, hand_joint_names

# Last link of each finger, and the fingertip in that link's frame (the extreme point of its mesh)
LAST_LINK = {'thumb': 'thumb_distal', 'index': 'index_intermediate', 'middle': 'middle_intermediate',
             'ring': 'ring_intermediate', 'pinky': 'pinky_intermediate'}
TIP_OFFSET = {
    'right': {'thumb': (0.0223, 0.0104, -0.0060), 'index': (0.0025, 0.0452, -0.0050),
              'middle': (0.0024, 0.0487, -0.0050), 'ring': (0.0025, 0.0453, -0.0050),
              'pinky': (0.0024, 0.0375, -0.0050)},
    'left': {'thumb': (0.0223, -0.0104, -0.0060), 'index': (0.0027, -0.0453, -0.0050),
             'middle': (0.0024, -0.0487, -0.0050), 'ring': (0.0027, -0.0453, -0.0050),
             'pinky': (0.0031, -0.0375, -0.0050)},
}


def _rot(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class HandIK:
    def __init__(self, urdf: Urdf, side: str, scale: float = 1.0, huber: float = 0.02, reg: float = 0.0003,
                 iterations: int = 4):
        self.side, self.scale, self.huber, self.reg, self.iterations = side, scale, huber, reg, iterations
        prefix = HAND_PREFIX[side]
        base = prefix + 'hand_base_link'
        actuators = hand_joint_names(side)
        # per finger: [(origin 4x4, axis, actuator index, multiplier, offset)], tip offset
        self.chains: List[Tuple[list, np.ndarray]] = []
        for finger in TIP_ORDER:
            self.chains.append((self._fold(urdf, base, prefix + LAST_LINK[finger], actuators),
                                np.array(TIP_OFFSET[side][finger])))
        # base-frame vectors -> the hand frame of g1tele.hand (x fingers, z palm normal)
        r_base_in_wrist = urdf.joints[prefix + 'base_link_joint'].origin[:3, :3]
        self.to_hand = robot_hand_in_link(side).T @ r_base_in_wrist
        # palm length of the robot: wrist -> middle knuckle
        mid = urdf.joints[prefix + 'middle_proximal_joint'].origin[:3, 3]
        self.palm = float(np.linalg.norm(mid))
        self.lo, self.hi = HAND_JOINT_RANGE[:, 0], HAND_JOINT_RANGE[:, 1]

    @staticmethod
    def joint_from_curl(curls):
        return curl_to_joint(curls)

    @staticmethod
    def _fold(urdf, base, tip_link, actuators):
        """Per movable joint: (R_origin, p_origin, axis, K, K@K, actuator index, multiplier, offset), with the fixed
        joints folded into the next origin. K is the cross-product matrix of the axis (Rodrigues)."""
        out, pending = [], np.eye(4)
        for j in urdf.chain(base, tip_link):
            pending = pending @ j.origin
            if j.type in ('revolute', 'continuous', 'prismatic'):
                master, mult, off = (j.name, 1.0, 0.0) if j.mimic is None else j.mimic
                a = j.axis
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                out.append((pending[:3, :3].copy(), pending[:3, 3].copy(), a, K, K @ K, actuators.index(master), mult, off))
                pending = np.eye(4)
        return out

    def fingertips(self, q, jacobian=False):
        """Fingertip positions (5x3, hand frame, m, from the wrist) and optionally d tips / d q (5x3x6)."""
        q = np.asarray(q, dtype=float)
        tips = np.zeros((5, 3))
        jac = np.zeros((5, 3, 6)) if jacobian else None
        eye = np.eye(3)
        for f, (joints, offset) in enumerate(self.chains):
            R, p = eye, np.zeros(3)
            if jacobian:
                axes_w, points_w = [], []
            for Ro, po, axis, K, KK, a, m, off in joints:
                Rj = R @ Ro
                pj = p + R @ po
                if jacobian:
                    axes_w.append(Rj @ axis)
                    points_w.append(pj)
                th = m * q[a] + off
                R = Rj @ (eye + np.sin(th) * K + (1.0 - np.cos(th)) * KK)
                p = pj
            tip = R @ offset + p
            tips[f] = tip
            if jacobian:
                for ax, pt, (_, _, _, _, _, a, m, _) in zip(axes_w, points_w, joints):
                    d = tip - pt
                    jac[f, :, a] += m * np.array([ax[1] * d[2] - ax[2] * d[1], ax[2] * d[0] - ax[0] * d[2],
                                                  ax[0] * d[1] - ax[1] * d[0]])
        tips = tips @ self.to_hand.T
        if jacobian:
            return tips, np.einsum('ij,fjk->fik', self.to_hand, jac)
        return tips

    def solve(self, tip_vectors, q_prev) -> np.ndarray:
        """Actuator angles (rad, FINGERS order) whose fingertips best match `tip_vectors` (5x3, palm lengths)."""
        target = np.asarray(tip_vectors, dtype=float) * self.palm * self.scale
        q = np.clip(np.asarray(q_prev, dtype=float), self.lo, self.hi)
        q_ref = q.copy()
        for _ in range(self.iterations):
            tips, J = self.fingertips(q, jacobian=True)
            err = tips - target                                   # 5x3
            dist = np.linalg.norm(err, axis=1)
            w = np.where(dist <= self.huber, 1.0, self.huber / np.maximum(dist, 1e-9))   # Huber, by reweighting
            Jm = (J * w[:, None, None]).reshape(-1, 6)
            A = J.reshape(-1, 6).T @ Jm + (self.reg + 1e-4) * np.eye(6)
            b = Jm.T @ err.reshape(-1) + self.reg * (q - q_ref)
            dq = -np.linalg.solve(A, b)
            q_new = np.clip(q + np.clip(dq, -0.5, 0.5), self.lo, self.hi)
            done = np.max(np.abs(q_new - q)) < 2e-3
            q = q_new
            if done:
                break
        return q
