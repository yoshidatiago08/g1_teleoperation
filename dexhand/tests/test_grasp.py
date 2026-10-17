"""Picking up the cylinder in MuJoCo: the fingers must be able to hold it without flinging it away."""
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('dex_retargeting')

from dex_hand import frames, paths
from dex_hand.mujoco_scene import CYL_HEIGHT, CYL_START, FINGER_MAX_TORQUE, TABLE_TOP, FloatingHandScene
from dex_hand.retarget import FingerRetargeter, robot_hand_rotation
from dex_hand.synthetic import FACING_CAMERA, make_hand


def wxyz(rotation):
    x, y, z, w = frames.quaternion(rotation)
    return np.array([w, x, y, z])


def side_grasp(finger_torque, dx=0.09, dy=0.048):
    """Dex5-1 beside the upright cylinder, palm towards it (fingers forward, curling round it), fingers closed with the real retargeting,
    then lifted 12 cm. Returns (how far the cylinder rose, its final height)."""
    fr = FingerRetargeter(paths.URDF_DIR, paths.retarget_config('right', 'dex5'), 'right', 1.0)
    rot = frames.hand_rotation(make_hand('right', [0.0] * 5, FACING_CAMERA), 'right')

    def pose(curl):
        for _ in range(60):
            joints = fr(frames.tip_vectors(make_hand('right', [curl] * 5, FACING_CAMERA), rot))
        return joints

    opened, closed = pose(0.0), pose(0.75)
    quat = wxyz(np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ robot_hand_rotation(paths.urdf('right', 'dex5'), 'right').T)
    scene = FloatingHandScene(paths.urdf('right', 'dex5'), 'right', finger_torque=finger_torque)
    cylinder = np.array([CYL_START[0], CYL_START[1], TABLE_TOP + CYL_HEIGHT / 2 + 0.002])
    wrist = cylinder - [dx, dy, 0.0]
    scene.set_target(wrist, quat)
    scene.snap_to_target()
    scene.set_fingers(opened)
    scene.step(400)
    scene.set_fingers(closed)
    scene.step(1000)
    start = scene.data.xpos[scene.cyl_id].copy()
    for k in range(300):
        scene.set_target(wrist + [0, 0, 0.12 * (k + 1) / 300], quat)
        scene.step(3)
    scene.step(500)
    z = scene.data.xpos[scene.cyl_id][2]
    return z - start[2], z


def test_the_limited_fingers_pick_the_cylinder_up():
    rise, z = side_grasp(FINGER_MAX_TORQUE)
    assert rise > 0.09 and z > TABLE_TOP + 0.1


def test_unlimited_servo_fingers_fling_the_cylinder_instead():
    rise, z = side_grasp(None)
    assert rise < 0.05 or z < 0.5                                           # it is knocked off, not held (the reason for the limit)


def test_only_the_finger_joints_are_force_limited_in_the_arm_scene():
    from dex_hand.mujoco_arm_scene import ARM_JOINTS, ArmScene
    scene = ArmScene(urdf_path=paths.g1_urdf('dex5'))
    m = scene.model
    for name, i in scene.act.items():
        limited = bool(m.actuator_forcelimited[i])
        assert limited == name.startswith(('R_', 'L_')), name
        if limited:
            assert m.actuator_forcerange[i][1] == pytest.approx(FINGER_MAX_TORQUE)
    assert all(not m.actuator_forcelimited[scene.act[n]] for n in ARM_JOINTS)
