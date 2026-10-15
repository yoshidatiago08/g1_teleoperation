"""Hand pose from MediaPipe landmarks, as plain numpy (no ROS, no MediaPipe: testable with synthetic hands).

Frames used here
    camera    x right, y down, z away from the camera (also MediaPipe's image and world-landmark axes)
    operator  x forward (towards the camera), y left, z up: what the operator would call "in front of me"
    hand H    x = fingers (wrist -> middle knuckle), z = palm normal, y = z cross x. Right and left
              hands follow the same rule, so one robot-side mapping serves both.
"""
import numpy as np

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
TIPS = (4, 8, 12, 16, 20)            # thumb, index, middle, ring, pinky

# camera -> operator axes (operator faces the camera: its forward is the camera's -z, its left the camera's +x)
CAMERA_TO_OPERATOR = np.array([[0.0, 0.0, -1.0],
                               [1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def hand_rotation(points, side):
    """Columns = hand frame axes (x fingers, y, z palm normal), in the axes of `points` (21x3).
    None if the landmarks are degenerate. With t from the pinky side to the index side, the palm
    normal is t x f for a right hand and f x t for a left hand."""
    p = np.asarray(points, dtype=float)
    f = unit(p[MIDDLE_MCP] - p[WRIST])
    if f is None:
        return None
    t = p[INDEX_MCP] - p[PINKY_MCP]
    t = unit(t - np.dot(t, f) * f)
    if t is None:
        return None
    n = unit(np.cross(t, f) if side == 'right' else np.cross(f, t))
    if n is None:
        return None
    return np.column_stack([f, np.cross(n, f), n])


def tip_vectors(world, rotation):
    """Wrist -> fingertip vectors (5x3, metres) in the hand frame, from the camera-axes landmarks."""
    w = np.asarray(world, dtype=float)
    return (w[list(TIPS)] - w[WRIST]) @ rotation


def depth_from_scale(world, pixels, focal_px):
    """Distance of the hand from the camera (m) from how big it looks, without a depth sensor.

    The landmarks' metric layout `world` (21x3, camera axes) projects to pixel offsets
    d_pix = (focal / Z) * d_world_xy, so Z is the least-squares fit over all landmark pairs with the
    wrist. Works for any hand orientation because it uses the x/y extent of every landmark."""
    w = np.asarray(world, dtype=float)[1:, :2] - np.asarray(world, dtype=float)[0, :2]
    d = np.asarray(pixels, dtype=float)[1:] - np.asarray(pixels, dtype=float)[0]
    denom = float(np.sum(w * d))
    if denom < 1e-9:
        return None
    return focal_px * float(np.sum(w * w)) / denom


def pixel_to_camera(u, v, z, focal_px, cx, cy):
    return np.array([(u - cx) * z / focal_px, (v - cy) * z / focal_px, z])


def to_operator(p_camera=None, r_camera=None, matrix=CAMERA_TO_OPERATOR):
    """Position and/or rotation from camera axes to operator axes (`matrix`: CAMERA_TO_OPERATOR for a level camera,
    gravity_aligned_axes(up) for a tilted one)."""
    p = None if p_camera is None else matrix @ np.asarray(p_camera, dtype=float)
    r = None if r_camera is None else matrix @ np.asarray(r_camera, dtype=float)
    return p, r


def gravity_aligned_axes(up_camera):
    """Camera axes -> operator axes (forward, left, up) from the camera's IMU, whatever the camera's tilt.

    `up_camera`: unit vector pointing up, in camera axes. Up is true up; forward is the camera's optical axis projected on
    the horizontal plane, reversed (the operator faces the camera, so forward is towards it). For a level camera, up =
    (0, -1, 0), this is exactly CAMERA_TO_OPERATOR. None if the camera looks almost straight up or down."""
    up = unit(np.asarray(up_camera, dtype=float))
    if up is None:
        return None
    forward = unit(-(np.array([0.0, 0.0, 1.0]) - up[2] * up))
    if forward is None or abs(up[2]) > 0.95:
        return None
    return np.vstack([forward, np.cross(up, forward), up])


def rotvec(r):
    """Rotation matrix -> rotation vector (axis * angle, the shortest way round). Goes through the quaternion, which stays accurate
    all the way to 180 degrees (an axis from sin(angle) does not)."""
    x, y, z, w = quaternion(np.asarray(r, dtype=float))
    v = np.array([x, y, z])
    if w < 0:
        v, w = -v, -w
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros(3)
    return v / norm * 2.0 * np.arctan2(norm, w)


def rotation_from_vec(v):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def quaternion(r):
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
    return tuple(float(x) for x in q)


# Calibration pose: the palm faces the camera, fingers up. In operator axes (x forward, y left, z up) the hand frame is
# x = fingers = up, z = palm normal = towards the camera = forward, y = z cross x = right.
CALIBRATION_REFERENCE = np.column_stack([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def rotation_angle(a, b):
    """Angle (rad) of the rotation taking a to b."""
    return float(np.arccos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0, -1.0, 1.0)))


def mean_rotation(rotations):
    """The rotation closest to all of `rotations` (sum, then the nearest rotation matrix by SVD)."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def calibration_from(samples, quantile=None):
    """From operator-frame hand rotations measured while the operator held the calibration pose.

    Returns (C, spread): C rotates everything measured in the operator frame (positions and orientations) so that the
    calibration pose reads exactly CALIBRATION_REFERENCE: it removes the camera's tilt and the operator's heading.
    `spread` (rad) is the largest deviation of a sample from the mean (how still the hand was), or with `quantile` (0.9) that
    quantile of the deviations: the hand model's tilt of a flat palm jitters, and a few noisy frames are not a moving hand."""
    mean = mean_rotation(samples)
    deviations = [rotation_angle(mean, r) for r in samples]
    return CALIBRATION_REFERENCE @ mean.T, float(max(deviations) if quantile is None else np.quantile(deviations, quantile))
