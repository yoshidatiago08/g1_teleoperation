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


def test_rotvec_is_accurate_at_and_around_180_degrees():
    rng = np.random.RandomState(1)
    axis = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    for angle in (np.pi, np.pi - 1e-4, np.pi - 1e-2):
        r = frames.rotation_from_vec(axis * angle) + rng.randn(3, 3) * 1e-4          # landmark noise
        r = frames.mean_rotation([r])
        assert np.allclose(frames.rotation_from_vec(frames.rotvec(r)), r, atol=1e-6)


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


R_FLOOR = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])      # fingers down, palm towards the camera


def test_orientation_is_absolute_not_relative_to_where_tracking_started():
    home = np.diag([1.0, -1.0, -1.0])
    for first in (home, frames.CALIBRATION_REFERENCE, R_FLOOR):          # however the hand was held when tracking began
        s = MotionScaler(home_position=[0, 0, 0], home_rotation=home, scale=0.3)
        s.update([0, 0, 0], first, True)
        _, r = s.update([0, 0, 0], R_FLOOR, True)
        assert np.allclose(r, R_FLOOR, atol=1e-9)                        # pointing at the floor = pointing at the floor


def test_rotation_scale_scales_the_deviation_from_home():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home, rotation_scale=0.5)
    _, r = s.update([0, 0, 0], rot_z(0.8) @ home, True)
    assert np.allclose(r, rot_z(0.4) @ home, atol=1e-9)


def test_rotation_is_rate_limited():
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=np.eye(3), max_rotation_rate=2.0)
    _, r = s.update([0, 0, 0], rot_z(1.5), True, dt=0.1)                 # wants 1.5 rad, may turn 0.2 rad
    assert frames.rotation_angle(np.eye(3), r) == pytest.approx(0.2, abs=1e-6)
    for _ in range(20):
        _, r = s.update([0, 0, 0], rot_z(1.5), True, dt=0.1)
    assert np.allclose(r, rot_z(1.5), atol=1e-6)


def test_calibration_removes_a_camera_tilt_and_reports_stillness():
    tilt = frames.rotation_from_vec([0.35, 0.0, 0.1])
    measured = [tilt @ frames.CALIBRATION_REFERENCE] * 5
    c, spread = frames.calibration_from(measured)
    assert np.allclose(c @ measured[0], frames.CALIBRATION_REFERENCE, atol=1e-9) and spread == pytest.approx(0, abs=1e-6)
    wobbly = [frames.rotation_from_vec([0.0, 0.0, a]) @ measured[0] for a in (-0.1, 0.0, 0.1)]
    assert frames.calibration_from(wobbly)[1] == pytest.approx(0.1, abs=0.01)


def test_gravity_axes_equal_the_level_camera_axes_when_the_camera_is_level():
    assert np.allclose(frames.gravity_aligned_axes([0.0, -1.0, 0.0]), frames.CAMERA_TO_OPERATOR)


@pytest.mark.parametrize('pitch, roll', [(0.35, 0.0), (-0.5, 0.0), (0.0, 0.4), (0.3, -0.3)])
def test_gravity_axes_undo_a_camera_pitch_and_roll(pitch, roll):
    """A camera pitched/rolled sees the world rotated by `tilt`; with its IMU's up vector the operator axes come out the same."""
    tilt = frames.rotation_from_vec([0, 0, roll]) @ frames.rotation_from_vec([pitch, 0, 0])      # pitch, then roll about the optical axis
    up_seen = tilt @ np.array([0.0, -1.0, 0.0])
    axes = frames.gravity_aligned_axes(up_seen)
    assert np.allclose(axes @ tilt, frames.CAMERA_TO_OPERATOR, atol=1e-9)


def test_gravity_axes_are_undefined_when_the_camera_looks_straight_down():
    assert frames.gravity_aligned_axes([0.0, 0.0, 1.0]) is None


def test_after_a_pause_the_orientation_is_kept_and_only_your_rotation_from_then_on_is_applied():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home)
    flat = home                                                       # the hand held flat
    turned = rot_z(0.5) @ flat                                        # turned to the side
    _, r = s.update([0, 0, 0], flat, True)
    _, r = s.update([0, 0, 0], turned, True)
    assert np.allclose(r, turned, atol=1e-9)                          # first engagement: absolute
    s.update([0, 0, 0], turned, False)                                # pause
    _, r = s.update([0, 0, 0], flat, True)                            # resume with the hand flat again
    assert np.allclose(r, turned, atol=1e-9)                          # the robot did NOT go back to flat
    _, r = s.update([0, 0, 0], rot_z(0.3) @ flat, True)               # turn 0.3 rad more
    assert np.allclose(r, rot_z(0.3) @ turned, atol=1e-9)             # it accumulates: 0.8 rad in total
    s.update([0, 0, 0], rot_z(0.3) @ flat, False)
    s.update([0, 0, 0], flat, True)
    _, r = s.update([0, 0, 0], rot_z(0.3) @ flat, True)
    assert frames.rotation_angle(home, r) == pytest.approx(1.1, abs=1e-6)   # and again


def test_align_makes_the_next_engagement_absolute_again():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home)
    s.update([0, 0, 0], rot_z(0.5) @ home, True)
    s.update([0, 0, 0], home, False)
    s.align()
    _, r = s.update([0, 0, 0], home, True)
    assert np.allclose(r, home, atol=1e-9)                            # swung back to match the hand


def test_changing_the_scale_does_not_realign_the_orientation():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home)
    s.update([0, 0, 0], rot_z(0.5) @ home, True)
    s.set_scale(scale=0.1)
    _, r = s.update([0, 0, 0], home, True)
    assert np.allclose(r, rot_z(0.5) @ home, atol=1e-9)


def test_ghost_follows_the_same_orientation_rule():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home)
    s.update([0, 0, 0], rot_z(0.5) @ home, True)
    s.update([0, 0, 0], home, False)
    s.update([0, 0, 0], home, True)
    _, r = s.ghost([0, 0, 0], rot_z(0.2) @ home)
    assert np.allclose(r, rot_z(0.2) @ rot_z(0.5) @ home, atol=1e-9)


def test_the_first_engagement_uses_plain_world_directions():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home, scale=0.5)
    s.update([0, 0, 0], rot_z(0.9) @ home, True)                         # however the hand is held, the robot is about to match it
    p, _ = s.update([0.1, 0.0, 0.0], rot_z(0.9) @ home, True)
    assert np.allclose(p, [0.05, 0, 0])


def test_after_a_pause_moving_along_your_fingers_moves_the_robot_along_its_fingers():
    home = np.diag([1.0, -1.0, -1.0])                                   # fingers along +x (forward), palm down
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home, scale=0.5)
    turned = rot_z(np.pi / 2) @ home                                    # robot (and hand) turned left: fingers along +y
    s.update([0, 0, 0], turned, True)
    s.update([0, 0, 0], turned, False)                                  # pause, bring the hand back flat (fingers along +x)
    for _ in range(6):
        s.update([0, 0, 0], home, True)
    s.update([0, 0, 0], home, False)
    s.update([0, 0, 0], home, True)
    p, r = s.update([0.1, 0.0, 0.0], home, True)                        # hand moves 10 cm along its fingers
    assert np.allclose(r[:, 0], [0, 1, 0], atol=1e-9)                   # the robot's fingers still point along +y
    assert np.allclose(p, [0.0, 0.05, 0.0], atol=1e-9)                  # and it moved 5 cm along them
    p, _ = s.update([0.1, 0.1, 0.0], home, True)                        # then 10 cm to the hand's sideways (+y world = hand left)
    assert np.allclose(p, [-0.05, 0.05, 0.0], atol=1e-9)                # = the robot's sideways (-x world, since it faces +y)


def test_the_frame_is_fixed_for_the_whole_engagement():
    home = np.diag([1.0, -1.0, -1.0])
    s = MotionScaler(home_position=[0, 0, 0], home_rotation=home, scale=1.0)
    s.update([0, 0, 0], rot_z(np.pi / 2) @ home, True)
    s.update([0, 0, 0], home, False)
    s.update([0, 0, 0], home, True)
    a, _ = s.update([0.1, 0, 0], home, True)
    b, _ = s.update([0.1, 0, 0], rot_z(0.8) @ home, True)               # turning the hand does not change where the same displacement goes
    assert np.allclose(a, b)


def test_limit_position_pulls_the_target_back_so_the_mapping_does_not_wind_up():
    import numpy as np
    import pytest
    from dex_hand.motion import MotionScaler
    m = MotionScaler([0.3, 0.0, 1.0], scale=1.0)
    m.update([0, 0, 0], np.eye(3), True)
    out, _ = m.update([1.0, 0, 0], np.eye(3), True)             # the hand ran far past the reachable edge: target x 1.3
    assert out[0] == pytest.approx(1.3)
    m.limit_position([0.4, 0.0, 1.0])                           # the arm could only get to x 0.4
    out, _ = m.update([0.9, 0, 0], np.eye(3), True)             # the hand moves back 10 cm: the target follows at once
    assert out[0] == pytest.approx(0.3)
