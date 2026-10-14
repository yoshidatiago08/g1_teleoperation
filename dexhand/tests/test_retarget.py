"""The real dex-retargeting optimiser on the real Inspire hand URDFs (needs dex_retargeting + pinocchio)."""

import numpy as np
import pytest

pytest.importorskip('dex_retargeting')

from dex_hand.retarget import FingerRetargeter

from dex_hand import paths
ACTUATORS = {'right': 'R', 'left': 'L'}


def make(side):
    return FingerRetargeter(paths.URDF_DIR, paths.retarget_config(side), side)


def tips_for(fr, pose):
    """Wrist -> fingertip vectors (hand frame H) the robot hand itself has at actuator angles `pose`."""
    opt = fr.retargeting.optimizer
    q = np.zeros(opt.robot.dof)
    for name, v in pose.items():
        q[opt.robot.get_joint_index(name)] = v
    if opt.adaptor is not None:
        q = opt.adaptor.forward_qpos(q)
    opt.robot.compute_forward_kinematics(q)
    names = opt.task_link_names
    origin = opt.robot.get_link_pose(opt.robot.get_link_index(opt.origin_link_names[0]))[:3, 3]
    vec = np.array([opt.robot.get_link_pose(opt.robot.get_link_index(n))[:3, 3] - origin for n in names])
    return vec @ fr.r_bh                                        # base link B -> hand frame H


@pytest.mark.parametrize('side', ['right', 'left'])
def test_open_robot_hand_points_along_hand_x(side):
    fr = make(side)
    tips = tips_for(fr, {})
    assert np.all(tips[1:, 0] > 0.08)                           # four fingers extend along +x (fingers)
    assert np.all(np.abs(tips[1:, 1]) < 0.08)


@pytest.mark.parametrize('side', ['right', 'left'])
def test_round_trip_recovers_the_actuators(side):
    fr = make(side)
    p = ACTUATORS[side]
    truth = {f'{p}_pinky_proximal_joint': 1.0, f'{p}_ring_proximal_joint': 0.9, f'{p}_middle_proximal_joint': 1.1,
             f'{p}_index_proximal_joint': 0.5, f'{p}_thumb_proximal_pitch_joint': 0.3,
             f'{p}_thumb_proximal_yaw_joint': 0.6}
    tips = tips_for(fr, truth) / fr.retargeting.optimizer.scaling
    for _ in range(40):                                         # the low-pass filter and warm start need a few steps
        out = fr(tips)
    for name, value in truth.items():
        assert out[name] == pytest.approx(value, abs=0.12), name


def test_closing_fingers_increases_the_angles_monotonically():
    fr = make('right')
    last = -1.0
    for amount in (0.0, 0.4, 0.8, 1.2):
        pose = {f'R_{f}_proximal_joint': amount for f in ('pinky', 'ring', 'middle', 'index')}
        tips = tips_for(fr, pose) / fr.retargeting.optimizer.scaling
        for _ in range(30):
            out = fr(tips)
        assert out['R_index_proximal_joint'] > last - 1e-3
        last = out['R_index_proximal_joint']
