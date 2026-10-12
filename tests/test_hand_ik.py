import numpy as np
import pytest

from g1tele.config import load_config
from g1tele.hand import HAND_JOINT_RANGE, curl_to_joint, joint_to_curl
from g1tele.retarget.hand_ik import HandIK
from g1tele.retarget.urdf import Urdf


@pytest.fixture(scope='module')
def urdf():
    return Urdf(load_config().robot.urdf)


@pytest.fixture(scope='module', params=['right', 'left'])
def ik(request, urdf):
    return HandIK(urdf, request.param)


def test_open_hand_points_along_the_finger_axis(ik):
    tips = ik.fingertips(np.zeros(6))
    assert (tips[1:, 0] > 0.15).all(), tips                  # four fingers reach out along +x
    assert np.abs(tips[1:, 1:]).max() < 0.06
    assert ik.palm == pytest.approx(0.1377, abs=0.002)


def test_closing_a_finger_brings_its_tip_to_the_palm(ik):
    q = np.zeros(6)
    open_ = ik.fingertips(q)
    q[3] = 1.7                                                # index closed
    closed = ik.fingertips(q)
    assert closed[1, 0] < open_[1, 0] - 0.07                  # the tip comes back toward the wrist
    np.testing.assert_allclose(np.delete(closed, 1, axis=0), np.delete(open_, 1, axis=0), atol=1e-9)


def test_left_hand_is_the_right_hand_mirrored(urdf):
    r, l = HandIK(urdf, 'right'), HandIK(urdf, 'left')
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = curl_to_joint(rng.uniform(0, 1, 6))
        mirrored = l.fingertips(q) * np.array([1.0, -1.0, 1.0])     # the left hand is the right one mirrored across y
        np.testing.assert_allclose(r.fingertips(q), mirrored, atol=0.01)      # the thumb tip offsets come from two different meshes


def test_jacobian_matches_finite_differences(ik):
    rng = np.random.default_rng(1)
    q = curl_to_joint(rng.uniform(0.1, 0.9, 6))
    _, J = ik.fingertips(q, jacobian=True)
    for k in range(6):
        d = np.zeros(6)
        d[k] = 1e-6
        num = (ik.fingertips(q + d) - ik.fingertips(q - d)) / 2e-6
        np.testing.assert_allclose(J[:, :, k], num, atol=1e-5)


def test_the_solver_recovers_the_hand_that_made_the_fingertips(ik):
    rng = np.random.default_rng(2)
    worst_tip, worst_q = 0.0, 0.0
    for _ in range(40):
        q_true = curl_to_joint(rng.uniform(0, 1, 6))
        target = ik.fingertips(q_true) / ik.palm
        q = np.zeros(6)
        for _ in range(6):                                    # a few frames of the same target: it is warm started
            q = ik.solve(target, q)
        worst_tip = max(worst_tip, np.abs(ik.fingertips(q) - ik.fingertips(q_true)).max())
        worst_q = max(worst_q, np.abs(joint_to_curl(q) - joint_to_curl(q_true))[:4].max())   # the thumb has two actuators for one point: several solutions
    assert worst_tip < 0.004, worst_tip
    assert worst_q < 0.1, worst_q


def test_unreachable_targets_give_the_closest_hand_inside_the_limits(ik):
    target = ik.fingertips(np.zeros(6)) / ik.palm * 1.6       # a hand twice as big as the robot's reaches too far
    q = np.zeros(6)
    for _ in range(5):
        q = ik.solve(target, q)
    assert (q >= HAND_JOINT_RANGE[:, 0] - 1e-9).all() and (q <= HAND_JOINT_RANGE[:, 1] + 1e-9).all()
    assert np.abs(q[:4]).max() < 0.2                          # straight fingers stay straight


def test_one_bad_fingertip_does_not_drag_the_others(ik):
    q_true = curl_to_joint(np.array([0.2, 0.2, 0.8, 0.8, 0.3, 0.5]))
    target = ik.fingertips(q_true) / ik.palm
    bad = target.copy()
    bad[2] += np.array([0.0, 0.6, -0.5])                      # the middle fingertip landmark is far off
    q = q_true.copy()
    for _ in range(6):
        q = ik.solve(bad, q)
    c_true, c = joint_to_curl(q_true), joint_to_curl(q)
    assert np.abs(np.delete(c, 2)[[0, 1, 3]] - np.delete(c_true, 2)[[0, 1, 3]]).max() < 0.15
