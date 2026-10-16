"""Both robot hands (Inspire, Unitree Dex5-1) behave the same way for everything above them: same hand frame, fingers curl towards the
palm, the retargeter closes them for a fist, the G1 mount puts the hand frame where the Inspire's was, and MuJoCo moves them."""
import numpy as np
import pytest

pytest.importorskip('dex_retargeting')

from dex_hand import paths
from dex_hand.retarget import FingerRetargeter, robot_hand_rotation
from dex_hand.synthetic import FACING_CAMERA, make_hand
from dex_hand import frames

MODELS = list(paths.HAND_MODELS)


def retargeter(model, side):
    return FingerRetargeter(paths.URDF_DIR, paths.retarget_config(side, model), side)


def tips(fr, side, q_by_name):
    """Wrist -> fingertip vectors in the hand frame H for joint angles {name: rad} (the rest 0; mimic joints followed)."""
    robot = fr.retargeting.optimizer.robot
    q = np.zeros(robot.dof)
    for name, v in q_by_name.items():
        q[robot.get_joint_index(name)] = v
    robot.compute_forward_kinematics(q)
    p = 'R' if side == 'right' else 'L'
    pos = lambda n: robot.get_link_pose(robot.get_link_index(n))[:3, 3]
    return np.array([pos(f'{p}_{f}_tip') - pos(f'{p}_hand_base_link') for f in ('thumb', 'index', 'middle', 'ring', 'pinky')]) @ fr.r_bh


@pytest.mark.parametrize('side', ['right', 'left'])
@pytest.mark.parametrize('model', MODELS)
def test_the_hand_frame_is_a_proper_rotation_with_fingers_along_x_and_the_thumb_on_the_index_side(model, side):
    fr = retargeter(model, side)
    assert np.linalg.det(fr.r_bh) == pytest.approx(1.0, abs=1e-6)
    open_tips = tips(fr, side, {})
    assert np.all(open_tips[1:, 0] > 0.12)                                     # the four fingers point along H x
    index_side = -1.0 if side == 'right' else 1.0
    assert open_tips[1, 1] * index_side > open_tips[4, 1] * index_side         # index tip is on the index side of the pinky
    assert open_tips[0, 1] * index_side > open_tips[1, 1] * index_side         # the thumb is further out on that side


@pytest.mark.parametrize('side', ['right', 'left'])
@pytest.mark.parametrize('model', MODELS)
def test_fingers_curl_towards_the_same_side_of_the_palm_for_both_hands(model, side):
    fr, p = retargeter(model, side), 'R' if side == 'right' else 'L'
    names = [n for n in fr.joint_names if p + '_index_' in n and ('proximal_joint' in n or '_mcp_' in n)]
    flexed = tips(fr, side, {names[0]: 1.0})[1] - tips(fr, side, {})[1]
    assert flexed[2] > 0.04                                                    # towards +z_H, as the Inspire does


@pytest.mark.parametrize('model', MODELS)
def test_the_retargeter_closes_the_fingers_for_a_fist(model):
    fr = retargeter(model, 'right')
    rot = frames.hand_rotation(make_hand('right', [0.0] * 5, FACING_CAMERA), 'right')

    def angles(curl):
        for _ in range(40):
            joints = fr(frames.tip_vectors(make_hand('right', [curl] * 5, FACING_CAMERA), rot))
        return np.array([joints[n] for n in fr.joint_names])

    assert angles(1.0).sum() > angles(0.0).sum() + 3.0
    assert len(fr.joint_names) == {'inspire': 12, 'dex5': 20}[model]


def test_dex5_has_sixteen_actuators_and_four_passive_joints():
    fr = retargeter('dex5', 'right')
    assert len(fr.retargeting.optimizer.target_joint_names) == 16
    assert {n for n in fr.joint_names} - set(fr.retargeting.optimizer.target_joint_names) == {f'R_{f}_dip_joint' for f in ('index', 'middle', 'ring', 'pinky')}


def test_mounted_on_the_g1_both_hands_have_the_same_hand_frame():
    pytest.importorskip('mujoco')
    from dex_hand.mujoco_arm_scene import ArmScene
    frames_world = []
    for model in MODELS:
        scene = ArmScene(urdf_path=paths.g1_urdf(model))
        r_base = scene.data.xmat[scene.base_id].reshape(3, 3)
        r_bh = robot_hand_rotation(paths.urdf('right', model), 'right')
        frames_world.append((r_base @ r_bh, scene.data.xpos[scene.base_id].copy()))
    assert np.allclose(frames_world[0][0], frames_world[1][0], atol=1e-4)      # same orientation of H
    assert np.linalg.norm(frames_world[0][1] - frames_world[1][1]) < 1e-4      # same wrist point


@pytest.mark.parametrize('model', MODELS)
def test_the_floating_hand_closes_in_mujoco(model):
    pytest.importorskip('mujoco')
    from dex_hand.mujoco_scene import FloatingHandScene
    fr = retargeter(model, 'right')
    scene = FloatingHandScene(paths.urdf('right', model), 'right')
    rot = frames.hand_rotation(make_hand('right', [0.0] * 5, FACING_CAMERA), 'right')
    for _ in range(40):
        joints = fr(frames.tip_vectors(make_hand('right', [1.0] * 5, FACING_CAMERA), rot))
    scene.set_fingers(joints)
    scene.step(600)
    qpos = lambda n: scene.data.qpos[scene.model.jnt_qposadr[scene.model.joint(n).id]]
    name = 'R_index_proximal_joint' if model == 'inspire' else 'R_index_pip_joint'
    assert qpos(name) > 0.8 and qpos(name) == pytest.approx(joints[name], abs=0.15)
    if model == 'dex5':                                                         # the passive distal joint follows the middle one
        assert qpos('R_index_dip_joint') == pytest.approx(0.83 * qpos('R_index_pip_joint'), abs=0.05)
