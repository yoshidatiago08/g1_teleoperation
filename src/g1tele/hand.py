"""Hand orientation and finger curls from 21 MediaPipe hand landmarks.

Pure numpy, shared by perception (measures the operator's hand) and retargeting (needs the robot
hand's frame in its wrist link, built with the same function so the two definitions agree).

The hand frame used everywhere:
    x = fingers direction (wrist -> middle-finger knuckle)
    z = palm normal (out of the palm)
    y = z cross x
Right and left hands follow the same rule, so the same robot code handles both.
"""
from __future__ import annotations

import numpy as np

from g1tele.geometry import unit

# MediaPipe Hands landmark indices
WRIST = 0
THUMB = (1, 2, 3, 4)    # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)    # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# Output channels, in Inspire actuator order (see protocol.FINGERS)
CHANNELS = ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']
# Landmarks that must be visible for each output channel to be measured
CHANNEL_LANDMARKS = (PINKY, RING, MIDDLE, INDEX, THUMB, THUMB + (17,))
_FINGER_OF = {'pinky': PINKY, 'ring': RING, 'middle': MIDDLE, 'index': INDEX}

# Default raw ranges (radians of summed flexion; palm lengths for the thumb rotation) that count
# as fully open / fully closed. A calibration replaces them with this person's own range.
DEFAULT_OPEN = np.array([0.4, 0.4, 0.4, 0.4, 0.2, 1.1])
DEFAULT_CLOSED = np.array([3.6, 3.6, 3.6, 3.6, 1.8, 0.5])

# Knuckle positions of the Inspire hand in its wrist_yaw_link frame (from the URDF). Landmark
# indices as above: 0 wrist, 5 index, 9 middle, 17 pinky knuckle.
_ROBOT_HAND = np.zeros((21, 3))
_ROBOT_HAND[0], _ROBOT_HAND[5], _ROBOT_HAND[9], _ROBOT_HAND[17] = (
    (0.042, 0, 0), (0.178, 0, 0.032), (0.179, 0, 0.013), (0.177, 0, -0.025))


# Inspire hand actuator ranges as the hardware (and unitree_sim_isaaclab) defines them, in radians
# of the driven joint, in CHANNELS order. A curl of 0 is the first number, a curl of 1 the second.
HAND_JOINT_RANGE = np.array([[0.0, 1.7], [0.0, 1.7], [0.0, 1.7], [0.0, 1.7], [0.0, 0.5], [-0.1, 1.3]])


def curl_to_joint(curls) -> np.ndarray:
    """Curls (0 open ... 1 closed) -> Inspire joint angles in radians."""
    c = np.clip(np.asarray(curls, dtype=float), 0.0, 1.0)
    return HAND_JOINT_RANGE[:, 0] + c * (HAND_JOINT_RANGE[:, 1] - HAND_JOINT_RANGE[:, 0])


def joint_to_curl(q) -> np.ndarray:
    """Inspire joint angles -> curls (the inverse of curl_to_joint)."""
    lo, hi = HAND_JOINT_RANGE[:, 0], HAND_JOINT_RANGE[:, 1]
    return np.clip((np.asarray(q, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


# Fingertip order of the vectors the perception side sends (ArmObs.tips) and the finger of each channel
TIP_ORDER = ('thumb', 'index', 'middle', 'ring', 'pinky')
TIP_ROW_OF_CHANNEL = (4, 3, 2, 1, 0, 0)          # CHANNELS -> row in TIP_ORDER


def fingertip_vectors(points, rotation, palm_length):
    """Wrist -> fingertip vectors of the 21 landmarks in the hand frame `rotation` (columns = axes),
    divided by `palm_length` so a big hand and a small hand give the same numbers: 5x3, TIP_ORDER."""
    p = np.asarray(points, dtype=float)
    tips = p[[THUMB[3], INDEX[3], MIDDLE[3], RING[3], PINKY[3]]] - p[WRIST]
    return (tips @ np.asarray(rotation)) / max(float(palm_length), 1e-6)


def _angle(a, b):
    a, b = unit(a), unit(b)
    if a is None or b is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def hand_rotation(points, side):
    """Rotation matrix whose columns are the hand frame axes, in the axes of `points` (21x3).
    None if the landmarks are degenerate.

    With t pointing from the pinky side to the index side, the palm normal is t x f for a right
    hand and f x t for a left hand.
    """
    p = np.asarray(points, dtype=float)
    f = unit(p[MIDDLE[0]] - p[WRIST])
    if f is None:
        return None
    t = p[INDEX[0]] - p[PINKY[0]]
    t = unit(t - np.dot(t, f) * f)
    if t is None:
        return None
    n = np.cross(t, f) if side == 'right' else np.cross(f, t)
    n = unit(n)
    if n is None:
        return None
    return np.column_stack([f, np.cross(n, f), n])


def robot_hand_in_link(side):
    """Rotation from the operator's hand frame to the robot wrist link frame."""
    return hand_rotation(_ROBOT_HAND, side)


def _flexion(p, finger):
    """Summed flexion angle (rad) of the three joints of one finger."""
    mcp, pip, dip, tip = (p[i] for i in finger)
    return (_angle(mcp - p[WRIST], pip - mcp) + _angle(pip - mcp, dip - pip)
            + _angle(dip - pip, tip - dip))


def raw_features(points):
    """The six raw measurements, in CHANNELS order: four flexion sums (rad), the thumb's bend (rad)
    and the thumb tip's distance to the pinky knuckle in palm lengths (big = thumb out)."""
    p = np.asarray(points, dtype=float)
    out = [_flexion(p, _FINGER_OF[name]) for name in CHANNELS[:4]]
    cmc, mcp, ip, tip = (p[i] for i in THUMB)
    out.append(_angle(mcp - cmc, ip - mcp) + _angle(ip - mcp, tip - ip))
    palm = np.linalg.norm(p[MIDDLE[0]] - p[WRIST]) + 1e-9
    out.append(float(np.linalg.norm(p[THUMB[3]] - p[PINKY[0]]) / palm))
    return np.array(out)


def curls(raw, open_=DEFAULT_OPEN, closed=DEFAULT_CLOSED):
    """Raw features -> 0 (open) ... 1 (closed) per channel, with this person's range."""
    return np.clip((np.asarray(raw) - open_) / (closed - open_), 0.0, 1.0)
