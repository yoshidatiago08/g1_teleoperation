"""Scaled, clutched hand motion: move your hand a lot, the robot hand moves a little.

    robot = anchor_robot + scale * (hand - anchor_hand)

Relative, not absolute: when engaged the robot hand continues from where it is, so there is never a
jump. Releasing the clutch (or losing the hand) holds the robot; re-engaging re-anchors, like lifting
a mouse and putting it down elsewhere. Rotation uses the same idea on the rotation vector, with its
own gain (1.0 = the robot hand turns exactly like yours).
"""
import numpy as np

from dex_hand.frames import rotation_from_vec, rotvec


class MotionScaler:
    def __init__(self, home_position, home_rotation=None, scale=0.3, rotation_scale=1.0):
        self.scale, self.rotation_scale = float(scale), float(rotation_scale)
        self.position = np.asarray(home_position, dtype=float).copy()      # the robot hand target
        self.rotation = np.eye(3) if home_rotation is None else np.asarray(home_rotation, dtype=float).copy()
        self._anchor = None                                                # (hand p, hand R, robot p, robot R, k, ks)
        self.engaged = False

    def set_scale(self, scale=None, rotation_scale=None):
        """Change the gains live. The anchor is retaken so the target does not jump."""
        if scale is not None:
            self.scale = float(scale)
        if rotation_scale is not None:
            self.rotation_scale = float(rotation_scale)
        self._anchor = None

    def release(self):
        """Hold the robot where it is until the next engaged update."""
        self.engaged = False
        self._anchor = None

    def update(self, hand_position, hand_rotation, engaged):
        """Feed the operator's hand pose (operator frame); returns the robot hand target (p, R)."""
        if not engaged:
            self.release()
            return self.position.copy(), self.rotation.copy()
        p, r = np.asarray(hand_position, dtype=float), np.asarray(hand_rotation, dtype=float)
        if self._anchor is None:
            self._anchor = (p.copy(), r.copy(), self.position.copy(), self.rotation.copy())
        self.engaged = True
        p0, r0, q0, s0 = self._anchor
        self.position = q0 + self.scale * (p - p0)
        self.rotation = rotation_from_vec(self.rotation_scale * rotvec(r @ r0.T)) @ s0
        return self.position.copy(), self.rotation.copy()

    def ghost(self, hand_position, hand_rotation):
        """Where the robot hand would be with scale 1 (and rotation gain 1) from the same anchor: the unscaled reference,
        for comparing against the scaled target. None while not engaged."""
        if self._anchor is None:
            return None
        p0, r0, q0, s0 = self._anchor
        return q0 + (np.asarray(hand_position, dtype=float) - p0), np.asarray(hand_rotation) @ r0.T @ s0
