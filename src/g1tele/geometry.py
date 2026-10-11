"""Rotations and body frames. Pure numpy/scipy: no camera, no ROS, no simulator.

Frames follow REP-103: rotation matrices whose columns are (forward, left, up).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def unit(v):
    """v / |v|, or None if v is (almost) zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def frame_from(left, up):
    """Rotation (columns forward, left, up) from a left direction and an approximate up.

    `up` only needs to point roughly up: its part along `left` is removed.
    """
    left = unit(left)
    if left is None:
        return None
    up = unit(up - np.dot(up, left) * left)
    if up is None:
        return None
    return np.column_stack([np.cross(left, up), left, up])


def orthonormalize(m):
    """Nearest rotation matrix (SVD)."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rotation_angle(a, b):
    """Angle (rad) of the rotation taking a to b."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def rotvec(r):
    """Rotation vector (axis * angle) of a rotation matrix."""
    return Rotation.from_matrix(r).as_rotvec()


def from_rotvec(v):
    return Rotation.from_rotvec(v).as_matrix()


def matrix_to_quat(r):
    """(x, y, z, w)."""
    return tuple(float(v) for v in Rotation.from_matrix(r).as_quat())


def quat_to_matrix(q):
    return Rotation.from_quat(q).as_matrix()


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def waist_matrix(yaw, roll, pitch):
    """Rotation of the G1's torso relative to the pelvis: Rz(yaw) Rx(roll) Ry(pitch)."""
    return rot_z(yaw) @ rot_x(roll) @ rot_y(pitch)


def estimate_shoulders(r_sh, l_sh, left_prev, width):
    """Fill in one hidden shoulder from the other, one shoulder-width along the last left axis.

    Returns (r_sh, l_sh, right_predicted, left_predicted) or None if there is not enough.
    """
    if r_sh is not None and l_sh is not None:
        return r_sh, l_sh, False, False
    if width is None or left_prev is None:
        return None
    if r_sh is not None:
        return r_sh, r_sh + left_prev * width, False, True
    if l_sh is not None:
        return l_sh - left_prev * width, l_sh, True, False
    return None


def project_on_sphere(center, point, radius, blend):
    """Move `point` toward the sphere of `radius` around `center` by a fraction `blend`.

    Used for bone lengths: a joint is `radius` away from its parent joint, whatever the depth
    sensor says. blend 1 puts the point exactly on the sphere, 0 leaves it alone.
    """
    d = point - center
    n = np.linalg.norm(d)
    if n < 1e-9:
        return point
    return point + blend * (center + d / n * radius - point)
