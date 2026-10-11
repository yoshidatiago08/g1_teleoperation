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
        for k in range(12):
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
    for i in range(15):
        frame.seq = i
        out = rt.step(frame, now=i * 0.033)
    assert out.ik_error['right'] < 2e-3
    tip, _, _, _, _, _ = kin.fk(np.array([out.arm_q[n] for n in kin.names]))
    assert np.linalg.norm(tip - p) < 3e-3


def test_lost_arm_eases_to_rest(cfg):
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    q = np.array([0.5, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    for k in range(10):
        frame, _, _ = human_frame(rt, 'right', q, seq=k)
        rt.step(frame, now=k * 0.033)
    lost = OperatorFrame(seq=99, mode=protocol.TRACKING, arms={'right': ArmObs(status=protocol.LOST)})
    t = 0.4
    for k in range(200):
        t += 0.033
        out = rt.step(lost, now=t)
    assert max(abs(out.arm_q[n]) for n in kin.names) < 1e-2


def test_joint_speed_is_limited(cfg):
    rt = Retargeter(cfg)
    kin = rt.kin['right']
    rt.step(None, now=0.0)
    frame, _, _ = human_frame(rt, 'right', np.array([1.0, -1.0, 0.5, 1.5, 0, 0, 0]))
    out = rt.step(frame, now=0.033)
    q = np.array([out.arm_q[n] for n in kin.names])
    assert np.max(np.abs(q)) <= cfg.robot.max_joint_speed * 0.033 + 1e-9
