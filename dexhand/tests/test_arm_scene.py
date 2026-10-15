"""The G1 arm scene in MuJoCo, and that MuJoCo and CuRobo agree on where the hand is for the same joints."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

from dex_hand.mujoco_arm_scene import ARM_JOINTS, PELVIS_Z, ArmScene
from dex_hand.mujoco_scene import CYL_START, TABLE_TOP

ARM_POSE = dict(zip(ARM_JOINTS, [-0.43, -0.68, 0.40, -0.08, -0.90, 0.57, 0.24]))


@pytest.fixture
def scene():
    return ArmScene(arm_start=ARM_POSE)


def test_builds_with_servos_for_the_arm_and_the_fingers(scene):
    assert all(n in scene.act for n in ARM_JOINTS)
    assert 'R_index_proximal_joint' in scene.act and 'R_index_intermediate_joint' not in scene.act     # the coupled one has no servo
    assert scene.status()['lifted_cm'] < 1.0


def test_arm_servos_reach_their_targets_and_hold(scene):
    target = {**ARM_POSE, 'right_shoulder_pitch_joint': -0.9, 'right_elbow_joint': 0.6}
    scene.set_arm_targets(target)
    scene.step(800)
    for name in ('right_shoulder_pitch_joint', 'right_elbow_joint'):
        assert scene.data.qpos[scene.qadr[name]] == pytest.approx(target[name], abs=0.03)


def test_fingers_follow_and_the_coupled_joints_move_with_them(scene):
    scene.set_fingers({'R_index_proximal_joint': 1.2})
    scene.step(600)
    q = scene.robot_joints()
    assert q['R_index_proximal_joint'] == pytest.approx(1.2, abs=0.1)
    assert q['R_index_intermediate_joint'] == pytest.approx(q['R_index_proximal_joint'], abs=0.1)


def test_the_hand_is_where_the_pelvis_height_says(scene):
    assert scene.hand_position()[2] > PELVIS_Z                       # the hand is up at the table, not at the floor


def test_mujoco_and_curobo_agree_on_where_the_hand_is(scene):
    torch = pytest.importorskip('torch')
    pytest.importorskip('curobo')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    from dex_hand.arm_ik import ArmIK
    ik = ArmIK(avoid_table=False)
    position, quat = ik.forward(ARM_POSE)
    assert np.allclose(scene.hand_position(), position, atol=0.002)
    assert abs(float(np.dot(scene.data.xquat[scene.base_id], quat))) > 0.9995


def test_the_hand_can_touch_the_cylinder(scene):
    scene.step(200)
    assert np.allclose(scene.data.xpos[scene.cyl_id][:2], CYL_START, atol=0.005)
    assert scene.data.xpos[scene.cyl_id][2] > TABLE_TOP
