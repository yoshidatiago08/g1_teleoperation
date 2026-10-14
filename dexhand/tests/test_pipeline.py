"""Observation -> hand command, end to end on the synthetic hand (real dex-retargeting, no ROS)."""

import numpy as np
import pytest

pytest.importorskip('dex_retargeting')

from dex_hand import frames
from dex_hand.motion import MotionScaler
from dex_hand.pipeline import HandPipeline
from dex_hand.retarget import FingerRetargeter
from dex_hand.sources import Observation
from dex_hand.synthetic import FACING_CAMERA, make_hand

from conftest import PKG
FOCAL, CX, CY = 600.0, 320.0, 240.0


def observe(side, curl, wrist):
    world = make_hand(side, [curl] * 5)
    points = world + wrist
    pixels = np.column_stack([CX + FOCAL * points[:, 0] / points[:, 2], CY + FOCAL * points[:, 1] / points[:, 2]])
    return Observation(world=world, pixels=pixels, depth=None, image=None)


def pipeline(side='right', scale=0.25):
    fr = FingerRetargeter(PKG / 'urdf', PKG / 'config' / f'inspire_hand_{side}.yml', side)
    scaler = MotionScaler([0.35, -0.2, 0.0], np.diag([1.0, -1.0, -1.0]), scale)
    return HandPipeline(side, fr, scaler, FOCAL, CX, CY)


def settle(p, side, curl, wrist, n=30):
    for _ in range(n):
        out = p.step(observe(side, curl, wrist))
    return out


@pytest.mark.parametrize('side', ['right', 'left'])
def test_fingers_close_when_the_synthetic_hand_closes(side):
    p, prefix = pipeline(side), 'R' if side == 'right' else 'L'
    wrist = np.array([0.0, 0.0, 1.5])
    opened = settle(p, side, 0.0, wrist).joints
    closed = settle(p, side, 1.0, wrist).joints
    for finger in ('pinky', 'ring', 'middle', 'index'):
        name = f'{prefix}_{finger}_proximal_joint'
        assert closed[name] > opened[name] + 0.5, finger
    assert opened[f'{prefix}_index_proximal_joint'] < 0.3


def test_depth_comes_from_the_hand_size_and_the_motion_is_scaled():
    p = pipeline(scale=0.25)
    c0 = settle(p, 'right', 0.0, np.array([0.0, 0.0, 1.5]))
    assert c0.depth_from == 'hand size'
    assert np.allclose(c0.position, [0.35, -0.2, 0.0], atol=1e-6)            # first frame anchors: no jump
    c1 = settle(p, 'right', 0.0, np.array([0.0, 0.0, 1.1]))                   # hand moves 0.4 m towards the camera
    assert c1.position[0] - 0.35 == pytest.approx(0.25 * 0.4, abs=0.01)       # operator forward, scaled
    assert c1.ghost[0][0] - 0.35 == pytest.approx(0.4, abs=0.02)              # the unscaled reference
    c2 = settle(p, 'right', 0.0, np.array([0.2, 0.0, 1.1]))                   # 0.2 m to the camera's right = operator left
    assert c2.position[1] - c1.position[1] == pytest.approx(0.25 * 0.2, abs=0.01)


def test_losing_the_hand_holds_the_robot_and_resumes_without_a_jump():
    p = pipeline()
    settle(p, 'right', 0.0, np.array([0.0, 0.0, 1.5]))
    held = settle(p, 'right', 0.0, np.array([0.0, 0.0, 1.3]))
    lost = p.step(Observation(world=None, pixels=None, depth=None, image=None))
    assert not lost.engaged and np.allclose(lost.position, held.position)
    back = p.step(observe('right', 0.0, np.array([0.5, 0.0, 2.5])))           # reappears far away
    assert np.allclose(back.position, held.position, atol=1e-6)


def test_clutch_freezes_the_target_but_the_fingers_keep_moving():
    p = pipeline()
    a = settle(p, 'right', 0.0, np.array([0.0, 0.0, 1.5]))
    b = settle(p, 'right', 1.0, np.array([0.3, 0.0, 1.0]))
    p.step(observe('right', 1.0, np.array([0.3, 0.0, 1.0])), clutch=True)
    for _ in range(30):
        c = p.step(observe('right', 0.0, np.array([-0.3, 0.0, 2.0])), clutch=True)
    assert np.allclose(c.position, b.position, atol=1e-6) and not c.engaged
    assert c.joints['R_index_proximal_joint'] < b.joints['R_index_proximal_joint'] - 0.5
