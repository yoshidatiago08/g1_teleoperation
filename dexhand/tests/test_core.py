import numpy as np
import pytest

from dex_hand import frames
from dex_hand.motion import MotionScaler


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def synthetic_hand(side='right'):
    """A flat hand in camera axes: wrist at the origin, fingers along -y (up in the image), palm facing the camera."""
    p = np.zeros((21, 3))
    sign = 1.0 if side == 'right' else -1.0
    for finger, (mcp, x) in {'index': (5, -0.03), 'middle': (9, -0.01), 'ring': (13, 0.01), 'pinky': (17, 0.03)}.items():
        for k in range(4):
            p[mcp + k] = [x * sign * -1, -(0.09 + 0.03 * k), 0.0]
    p[1:5] = [[-0.03 * sign, -0.02, 0], [-0.05 * sign, -0.05, 0], [-0.07 * sign, -0.07, 0], [-0.09 * sign, -0.09, 0]]
    return p


@pytest.mark.parametrize('side', ['right', 'left'])
def test_hand_rotation_is_a_rotation_and_points_along_the_fingers(side):
    r = frames.hand_rotation(synthetic_hand(side), side)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r[:, 0], [0, -1, 0], atol=0.2)         # x = fingers = up in the image


def test_left_hand_gives_the_mirror_image_vectors_the_left_urdf_expects():
    """Same rule for both hands (x fingers, z palm normal); a mirrored hand differs only by the sign of y."""
    right, left = synthetic_hand('right'), synthetic_hand('left')
    tr = frames.tip_vectors(right, frames.hand_rotation(right, 'right'))
    tl = frames.tip_vectors(left, frames.hand_rotation(left, 'left'))
    assert np.allclose(tr * [1, -1, 1], tl, atol=1e-9)


def test_depth_from_scale_recovers_distance_for_a_rotated_hand():
    world = synthetic_hand() @ rot_z(0.7).T
    world[:, 2] += np.linspace(-0.02, 0.02, 21)               # some depth relief
    focal, z = 600.0, 1.7
    pix = focal * world[:, :2] / z + np.array([320.0, 240.0])
    assert frames.depth_from_scale(world, pix, focal) == pytest.approx(z, rel=0.01)


def test_operator_axes():
    forward_in_camera = np.array([0.0, 0.0, -1.0])             # towards the camera
    p, _ = frames.to_operator(forward_in_camera)
    assert np.allclose(p, [1, 0, 0])
    up_in_camera = np.array([0.0, -1.0, 0.0])                   # image y is down
    assert np.allclose(frames.to_operator(up_in_camera)[0], [0, 0, 1])
    assert np.allclose(frames.to_operator(np.array([1.0, 0, 0]))[0], [0, 1, 0])   # camera right = operator left


def test_rotvec_roundtrip_incl_near_180():
    for v in ([0.3, -0.2, 0.5], [0, 0, np.pi - 1e-3], [np.pi - 1e-3, 0, 0], [1e-12, 0, 0]):
        r = frames.rotation_from_vec(v)
        assert np.allclose(frames.rotation_from_vec(frames.rotvec(r)), r, atol=1e-6)


def test_scaler_moves_a_fraction_and_never_jumps_on_engage():
    s = MotionScaler(home_position=[0.3, -0.2, 0.1], scale=0.25)
    p, _ = s.update([1.0, 0.0, 0.0], np.eye(3), engaged=True)
    assert np.allclose(p, [0.3, -0.2, 0.1])                    # first frame anchors, no jump
    p, _ = s.update([1.4, 0.2, -0.4], np.eye(3), engaged=True)
    assert np.allclose(p, [0.4, -0.15, 0.0])                   # 25% of the hand motion


def test_clutch_holds_then_resumes_from_where_the_robot_is():
    s = MotionScaler(home_position=[0, 0, 0], scale=0.5)
    s.update([0, 0, 0], np.eye(3), True)
    s.update([0.2, 0, 0], np.eye(3), True)                      # robot at 0.1
    held, _ = s.update([5.0, 5.0, 5.0], np.eye(3), False)       # hand moves a lot while clutched
    assert np.allclose(held, [0.1, 0, 0])
    p, _ = s.update([9.0, 0, 0], np.eye(3), True)               # re-engage somewhere else
    assert np.allclose(p, [0.1, 0, 0])
    p, _ = s.update([9.2, 0, 0], np.eye(3), True)
    assert np.allclose(p, [0.2, 0, 0])


def test_changing_the_scale_live_does_not_jump():
    s = MotionScaler(home_position=[0, 0, 0], scale=1.0)
    s.update([0, 0, 0], np.eye(3), True)
    s.update([0.3, 0, 0], np.eye(3), True)
    s.set_scale(0.1)
    p, _ = s.update([0.3, 0, 0], np.eye(3), True)
    assert np.allclose(p, [0.3, 0, 0])
    p, _ = s.update([0.5, 0, 0], np.eye(3), True)
    assert np.allclose(p, [0.32, 0, 0])


def test_rotation_gain():
    s = MotionScaler(home_position=[0, 0, 0], scale=1.0, rotation_scale=0.5)
    s.update([0, 0, 0], np.eye(3), True)
    _, r = s.update([0, 0, 0], rot_z(0.8), True)
    assert np.allclose(r, rot_z(0.4), atol=1e-9)
