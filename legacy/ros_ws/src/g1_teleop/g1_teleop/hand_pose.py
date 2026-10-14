"""Hand orientation and finger curls from 21 MediaPipe hand landmarks.

Pure numpy: no ROS and no MediaPipe, so it can be tested with synthetic hands.

The hand frame H used everywhere in this project:
    x = fingers direction (wrist -> middle-finger knuckle)
    z = palm normal (pointing out of the palm)
    y = z cross x
Both the right and the left hand follow the same rule, so the same robot code handles both.
"""
import numpy as np

# MediaPipe Hands landmark indices
WRIST = 0
THUMB = (1, 2, 3, 4)    # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)    # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# Names and order of the six Inspire hand actuators; values are 0 = open ... 1 = closed.
ACTUATORS = ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']

# Total finger flexion (radians, summed over the three joints) that counts as fully open / closed.
FINGER_OPEN, FINGER_CLOSED = 0.4, 3.6
THUMB_OPEN, THUMB_CLOSED = 0.2, 1.8
# Thumb tip to pinky knuckle, in palm lengths: far apart = thumb out, close = thumb across the palm.
THUMB_ROT_OPEN, THUMB_ROT_CLOSED = 1.1, 0.5


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _angle(a, b):
    a, b = _unit(a), _unit(b)
    if a is None or b is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _ramp(value, lo, hi):
    """0 at lo, 1 at hi, clipped."""
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def hand_rotation(points, side):
    """Rotation matrix whose columns are the hand frame axes (x fingers, y, z palm normal),
    expressed in the axes of `points` (21x3). None if the landmarks are degenerate.

    The palm normal follows the anatomy: with t pointing from the pinky side to the index side,
    the normal is t x f for a right hand and f x t for a left hand.
    """
    p = np.asarray(points, dtype=float)
    f = _unit(p[MIDDLE[0]] - p[WRIST])
    t = p[INDEX[0]] - p[PINKY[0]]
    if f is None:
        return None
    t = _unit(t - np.dot(t, f) * f)  # the part of "pinky -> index" that is across the hand
    if t is None:
        return None
    n = np.cross(t, f) if side == 'right' else np.cross(f, t)
    n = _unit(n)
    return np.column_stack([f, np.cross(n, f), n])


def _finger_flexion(p, finger, base):
    """Sum of the three joint flexion angles of a finger (radians)."""
    mcp, pip, dip, tip = (p[i] for i in finger)
    return (_angle(mcp - base, pip - mcp) + _angle(pip - mcp, dip - pip)
            + _angle(dip - pip, tip - dip))


def finger_curl(points, finger):
    p = np.asarray(points, dtype=float)
    return _ramp(_finger_flexion(p, finger, p[WRIST]), FINGER_OPEN, FINGER_CLOSED)


def hand_state(points):
    """Actuator values (0 open ... 1 closed) for the Inspire hand, as a list in ACTUATORS order.

    The pinky's curl drives the pinky, ring and middle fingers together; the index and the thumb
    drive their own. Per-finger control would just use finger_curl() for each one.
    """
    p = np.asarray(points, dtype=float)
    pinky = finger_curl(p, PINKY)
    index = finger_curl(p, INDEX)
    cmc, mcp, ip, tip = (p[i] for i in THUMB)
    bend = _ramp(_angle(mcp - cmc, ip - mcp) + _angle(ip - mcp, tip - ip), THUMB_OPEN, THUMB_CLOSED)
    palm = np.linalg.norm(p[MIDDLE[0]] - p[WRIST]) + 1e-9
    rotation = 1.0 - _ramp(np.linalg.norm(p[THUMB[3]] - p[PINKY[0]]) / palm,
                           THUMB_ROT_CLOSED, THUMB_ROT_OPEN)
    return [pinky, pinky, pinky, index, bend, rotation]


def orthonormalize(m):
    """Nearest rotation matrix (SVD), used after averaging rotation matrices."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rotation_angle(a, b):
    """Angle (rad) of the rotation taking a to b."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def matrix_to_quaternion(r):
    """(x, y, z, w) of a rotation matrix."""
    t = np.trace(r)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return tuple(q)


def quaternion_to_matrix(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return None  # an all-zero quaternion means "no orientation"
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
