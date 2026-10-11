import numpy as np
import pytest

from g1tele import protocol
from g1tele.config import load_config
from g1tele.geometry import unit
from g1tele.protocol import ArmObs, BoneLengths, OperatorFrame
from g1tele.retarget.retargeter import Retargeter


@pytest.fixture(scope='module')
def cfg():
    return load_config()


def human_frame(rt, side, q, upper_h=0.30, fore_h=0.27, with_elbow=True, seq=0):
    """The operator frame a human with these bones would produce for robot posture q."""
    kin = rt.kin[side]
    p, r, pe, _, _, _ = kin.fk(q)
    e_h = (pe - kin.shoulder) / kin.upper_arm * upper_h
    w_h = e_h + (p - pe) / kin.forearm * fore_h
    arm = ArmObs(status=protocol.TRACKING, quality=1.0, wrist=w_h, elbow=e_h if with_elbow else None)
    return OperatorFrame(seq=seq, mode=protocol.TRACKING, arms={side: arm},
                         bones={side: BoneLengths(upper_h, fore_h)}), p, pe


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1)))


def test_bone_scaled_follows_bone_directions(cfg):
    """The robot's wrist is reached to the sub-mm and its upper arm points where the operator's does."""
    rt = Retargeter(cfg)
    rng = np.random.default_rng(1)
    kin = rt.kin['right']
    for _ in range(20):
        q = rng.uniform(kin.lower * 0.4, kin.upper * 0.4)
        rt.ik['right'].reset()
        for k in range(150):                   # the arm eases into the pose (engage), about 4 s
            frame, p, pe = human_frame(rt, 'right', q, seq=k)
            out = rt.step(frame, now=k * 0.033)
        assert out.ik_error['right'] < 1.5e-3
        q_out = np.array([out.arm_q[n] for n in kin.names])
        _, _, elbow, _, _, _ = kin.fk(q_out)
        human_upper = frame.arms['right'].elbow
        assert angle_deg(elbow - kin.shoulder, human_upper) < 8.0


def test_uniform_mapping_scales_reach(cfg):
    rt = Retargeter(load_config(overrides={'ik': {'mapping': 'uniform'}}))
    kin = rt.kin['right']
    q = np.array([0.3, -0.3, 0.2, 0.9, 0.0, 0.0, 0.0])
    p, _, _, _, _, _ = kin.fk(q)
    k = (0.30 + 0.27) / (kin.upper_arm + kin.forearm)
    frame = OperatorFrame(mode=protocol.TRACKING, bones={'right': BoneLengths(0.30, 0.27)},
                          arms={'right': ArmObs(status=protocol.TRACKING, wrist=(p - kin.shoulder) * k)})
    for i in range(150):
        frame.seq = i
        out = rt.step(frame, now=i * 0.033)
    assert out.ik_error['right'] < 2e-3
    tip, _, _, _, _, _ = kin.fk(np.array([out.arm_q[n] for n in kin.names]))
    assert np.linalg.norm(tip - p) < 3e-3


def test_lost_arm_holds_then_drifts_slowly_to_rest(cfg):
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    q = np.array([0.5, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    for k in range(150):
        frame, _, _ = human_frame(rt, 'right', q, seq=k)
        out = rt.step(frame, now=k * 0.033)
    held = np.array([out.arm_q[n] for n in kin.names])
    lost = OperatorFrame(seq=99, mode=protocol.TRACKING, arms={'right': ArmObs(status=protocol.LOST)})
    t = 150 * 0.033
    for k in range(int(2.0 / 0.033)):                    # first 2 s: it holds, it does NOT swing to rest
        t += 0.033
        out = rt.step(lost, now=t)
    assert np.allclose([out.arm_q[n] for n in kin.names], held, atol=1e-9)
    for k in range(int(25 / 0.033)):                     # then it drifts slowly
        t += 0.033
        prev = np.array([out.arm_q[n] for n in kin.names])
        out = rt.step(lost, now=t)
        assert np.max(np.abs(np.array([out.arm_q[n] for n in kin.names]) - prev)) <= cfg.robot.rest_speed * 0.033 + 1e-9
    assert np.allclose([out.arm_q[n] for n in kin.names], rt.ik['right'].q_rest, atol=1e-2)


def test_calibration_holds_the_arms_instead_of_sweeping_them(cfg):
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    q = np.array([0.5, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    for k in range(100):
        frame, _, _ = human_frame(rt, 'right', q, seq=k)
        out = rt.step(frame, now=k * 0.033)
    before = np.array([out.arm_q[n] for n in kin.names])
    calibrating = OperatorFrame(seq=200, mode=protocol.CALIBRATING)
    for k in range(60):
        out = rt.step(calibrating, now=(100 + k) * 0.033)
    assert np.allclose([out.arm_q[n] for n in kin.names], before, atol=1e-9)


def test_engaging_never_jumps(cfg):
    """When tracking starts, the arm leaves where it is and the offset to the operator's pose fades out."""
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    q0 = np.array([rt.ik['right'].q[i] for i in range(7)])
    q = np.array([0.9, -0.6, 0.3, 1.3, 0.0, 0.0, 0.0])
    frame, p, _ = human_frame(rt, 'right', q, seq=0)
    out = rt.step(frame, now=0.0)
    out = rt.step(frame, now=0.033)
    first = np.array([out.arm_q[n] for n in kin.names])
    assert np.max(np.abs(first - q0)) < 0.3              # one frame: a small step, not the whole way
    for k in range(2, 150):
        frame.seq = k
        out = rt.step(frame, now=k * 0.033)
    tip = kin.fk(np.array([out.arm_q[n] for n in kin.names]))[0]
    assert out.ik_error['right'] < 3e-3


def test_joint_speed_is_limited(cfg):
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    q0 = rt.ik['right'].q.copy()
    rt.step(None, now=0.0)
    frame, _, _ = human_frame(rt, 'right', np.array([1.0, -1.0, 0.5, 1.5, 0, 0, 0]))
    out = rt.step(frame, now=0.033)
    q = np.array([out.arm_q[n] for n in kin.names])
    assert np.max(np.abs(q - q0)) <= cfg.robot.max_joint_speed * 0.033 + 1e-9
