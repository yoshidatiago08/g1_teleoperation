"""Scaled, clutched hand motion: move your hand a lot, the robot hand moves a little.

Position is relative, so a big motion becomes a small precise one and the robot never jumps, and it is expressed in the hand's own
axes, so moving your hand along its fingers moves the robot along ITS fingers, whatever the robot's orientation:

    robot = anchor_robot + scale * M * (hand - anchor_hand)        M = robot orientation * (your orientation)^-1, both taken once at the anchor

For the first engagement after a calibration the robot is about to match your orientation, so M is the identity (plain world
directions). After a pause (when the robot may be turned relative to your hand, e.g. for a side grasp) M carries your hand's directions
onto the robot hand's. M is fixed for the whole engagement (your orientation at the anchor is averaged over the last few frames), so
nothing is integrated frame by frame and the mapping cannot drift or accumulate noise.

Releasing the clutch (or losing the hand) holds the robot; re-engaging re-anchors, like lifting a mouse and putting it down
elsewhere. Orientation works in two modes:
  * the first engagement after a calibration (or after align()) is ABSOLUTE: the robot hand points where yours points (pointing at
    the floor is pointing at the floor). The calibration (pipeline.py) makes "where yours points" agree with the simulator's axes.
  * every later engagement (after a pause, or after the hand was lost) is RELATIVE, like the position: the robot keeps the orientation
    it has and applies only how much you rotate your hand from then on, about the world axes. So a hand that turns to the side and
    leaves the camera's view can be turned a little, paused, flattened again, resumed and turned some more: the rotation accumulates.
`rotation_scale` scales the rotation (the deviation from home in absolute mode, the change since the anchor in relative mode); 1.0
copies your hand exactly. The orientation change is rate-limited, so re-engaging swings smoothly instead of snapping.
"""
from collections import deque

import numpy as np

from dex_hand.frames import mean_rotation, rotation_from_vec, rotvec


class MotionScaler:
    def __init__(self, home_position, home_rotation=None, scale=0.3, rotation_scale=1.0, max_rotation_rate=None):
        self.scale, self.rotation_scale = float(scale), float(rotation_scale)
        self.max_rotation_rate = max_rotation_rate          # rad/s, None = no limit
        self.home_rotation = np.eye(3) if home_rotation is None else np.asarray(home_rotation, dtype=float).copy()
        self.position = np.asarray(home_position, dtype=float).copy()      # the robot hand target
        self.rotation = self.home_rotation.copy()
        self._anchor = None                                                # see update()
        self._recent = deque(maxlen=6)                                     # your last hand rotations: the anchor's frame is their mean
        self.align_next = True          # the next engagement matches your absolute orientation (the first one after a calibration)
        self.engaged = False

    def set_scale(self, scale=None, rotation_scale=None):
        """Change the gains live. The position anchor is retaken so the target does not jump."""
        if scale is not None:
            self.scale = float(scale)
        if rotation_scale is not None:
            self.rotation_scale = float(rotation_scale)
        self._anchor = None

    def align(self):
        """The next engaged frame takes your absolute orientation again (the robot swings there, rate-limited)."""
        self.align_next = True
        self._anchor = None

    def release(self):
        """Hold the robot where it is until the next engaged update; also forget the recent hand rotations (the hand was lost)."""
        self._recent.clear()
        self._hold()

    def _hold(self):
        self.engaged = False
        self._anchor = None

    def update(self, hand_position, hand_rotation, engaged, dt=None):
        """Feed the operator's (calibrated) hand pose in the operator frame; returns the robot hand target (p, R)."""
        p, r = np.asarray(hand_position, dtype=float), np.asarray(hand_rotation, dtype=float)
        self._recent.append(r)
        if not engaged:
            self._hold()
            return self.position.copy(), self.rotation.copy()
        if self._anchor is None:
            absolute, self.align_next = self.align_next, False
            frame = np.eye(3) if absolute else self.rotation @ mean_rotation(list(self._recent)).T
            self._anchor = (p.copy(), self.position.copy(), r.copy(), self.rotation.copy(), absolute, frame)
        self.engaged = True
        p0, q0, r0, s0, absolute, frame = self._anchor
        self.position = q0 + self.scale * (frame @ (p - p0))

        if absolute:
            target = rotation_from_vec(self.rotation_scale * rotvec(r @ self.home_rotation.T)) @ self.home_rotation
        else:
            target = rotation_from_vec(self.rotation_scale * rotvec(r @ r0.T)) @ s0
        step = rotvec(target @ self.rotation.T)
        angle = np.linalg.norm(step)
        if self.max_rotation_rate is not None and dt is not None and angle > self.max_rotation_rate * dt:
            step *= self.max_rotation_rate * dt / angle
        self.rotation = rotation_from_vec(step) @ self.rotation
        return self.position.copy(), self.rotation.copy()

    def limit_position(self, reachable):
        """The robot could only reach `reachable` instead of the target: move the anchor so the target is there. Without this the target keeps
        running away beyond the workspace and you have to retrace the whole excursion before the robot answers again."""
        if self._anchor is None:
            return
        shift = np.asarray(reachable, dtype=float) - self.position
        p0, q0, *rest = self._anchor
        self._anchor = (p0, q0 + shift, *rest)
        self.position = np.asarray(reachable, dtype=float).copy()

    def ghost(self, hand_position, hand_rotation):
        """Where the robot hand would be at position scale 1 from the same anchor, with your (unscaled) rotation: the reference for
        comparing against the scaled target. None while not engaged."""
        if self._anchor is None:
            return None
        p0, q0, r0, s0, absolute, frame = self._anchor
        r = np.asarray(hand_rotation, dtype=float)
        return q0 + frame @ (np.asarray(hand_position, dtype=float) - p0), r if absolute else rotation_from_vec(rotvec(r @ r0.T)) @ s0
