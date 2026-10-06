"""Torso and pelvis frames, hidden-shoulder prediction and waist angles.

Pure numpy (no ROS, no MediaPipe), so it can be tested with synthetic skeletons.

Frames follow REP-103 and are written as rotation matrices whose columns are
(forward, left, up), expressed in the coordinates of the input points (the camera frame).
"""
import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def frame_from(left, up):
    """Rotation matrix (columns forward, left, up) from a left direction and an approximate up.

    `up` only needs to point roughly up: the part of it along `left` is removed.
    """
    left = unit(left)
    if left is None:
        return None
    up = unit(up - np.dot(up, left) * left)
    if up is None:
        return None
    return np.column_stack([np.cross(left, up), left, up])


def estimate_shoulders(r_sh, l_sh, r_hp, l_hp, up, left_prev, width, torso_len):
    """Fill in a shoulder that was not measured, from the rest of the torso.

    The shoulders and hips are a rigid shape, so a hidden shoulder sits one shoulder-width away
    from the other one along the body's left-right axis, and with both shoulders hidden the pair
    sits one torso length above the hips. Each argument is a 3D point or None, except `up` (unit
    vector, the torso's last known up direction), `left_prev` (unit vector, the last known left
    direction), `width` (shoulder width) and `torso_len` (hip midpoint to shoulder midpoint).

    Returns (r_sh, l_sh, right_was_predicted, left_was_predicted), or None when there is too
    little left to predict from.
    """
    if r_sh is not None and l_sh is not None:
        return r_sh, l_sh, False, False
    if width is None:
        return None
    hips = r_hp is not None and l_hp is not None
    left = unit(l_hp - r_hp) if hips else left_prev
    if left is None:
        return None
    if r_sh is not None:
        return r_sh, r_sh + left * width, False, True
    if l_sh is not None:
        return l_sh - left * width, l_sh, True, False
    if hips and torso_len is not None and up is not None:
        mid = (r_hp + l_hp) / 2 + up * torso_len
        return mid - left * width / 2, mid + left * width / 2, True, True
    return None


def waist_angles(r_rel):
    """(yaw, roll, pitch) of the G1's waist from the torso's rotation relative to the pelvis.

    The waist chain is Rz(yaw) Rx(roll) Ry(pitch), so r_rel = Rz Rx Ry.
    """
    roll = np.arcsin(np.clip(r_rel[2, 1], -1.0, 1.0))
    yaw = np.arctan2(-r_rel[0, 1], r_rel[1, 1])
    pitch = np.arctan2(-r_rel[2, 0], r_rel[2, 2])
    return float(yaw), float(roll), float(pitch)
