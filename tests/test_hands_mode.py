"""Reach / hands-only control: the protocol, the retargeter's clutch, and the perception pipeline."""
import numpy as np
import pytest

from g1tele import bench, protocol
from g1tele.config import load_config
from g1tele.geometry import from_rotvec
from g1tele.protocol import ArmObs, OperatorFrame
from g1tele.retarget.retargeter import Retargeter


def test_control_and_tips_survive_the_wire():
    obs = ArmObs(status=protocol.TRACKING, wrist=np.array([.1, .2, .3]), hand_R=np.eye(3),
                 fingers=np.zeros(6), tips=np.arange(15, dtype=float).reshape(5, 3) / 10)
    f = OperatorFrame(seq=1, mode=protocol.TRACKING, arms={'right': obs}, control=protocol.HANDS)
    g = OperatorFrame.from_bytes(f.to_bytes())
    assert g.control == protocol.HANDS
    np.testing.assert_allclose(g.arms['right'].tips, obs.tips, atol=1e-6)
    assert OperatorFrame.from_bytes(OperatorFrame(mode=protocol.IDLE).to_bytes()).control == protocol.REACH


def _frame(seq, control, wrist, hand_R, sides=('right', 'left')):
    arms = {}
    for s in sides:
        arms[s] = ArmObs(status=protocol.TRACKING, quality=1.0, wrist=np.array(wrist, float) * (1 if s == 'right' else 1),
                         hand_R=hand_R, fingers=np.full(6, 0.3), hand_quality=1.0)
    return OperatorFrame(seq=seq, t_capture=seq / 30.0, mode=protocol.TRACKING, arms=arms, control=control)


@pytest.fixture(scope='module')
def retargeter_after_reach():
    cfg = load_config(overrides={'robot': {'backend': 'kinematic'}})
    rt = Retargeter(cfg)
    t = 0.0
    for i in range(120):                                   # reach: the arm settles at a pose
        rt.step(_frame(i, protocol.REACH, [0.15, -0.05, -0.35], None), now=t)
        t += 1 / 30
    return cfg, rt, t


def test_hands_only_turns_the_wrist_like_the_hand_and_holds_the_arm(retargeter_after_reach):
    cfg, rt, t = retargeter_after_reach
    side = 'right'
    kin, ik = rt.kin[side], rt.ik[side]
    w0, R0 = kin.fk(ik.q)[0], kin.fk(ik.q)[1]
    R_hand = np.eye(3)
    turn = from_rotvec(np.array([0.0, 0.0, np.radians(35.0)]))     # the operator turns the hand 35 deg about the vertical
    n = 0
    for i in range(200, 330):
        R = R_hand if i < 215 else turn @ R_hand
        rt.step(_frame(i, protocol.HANDS, [0.0, 0.0, 0.4], R, sides=('right', 'left')), now=t)
        t += 1 / 30
        n += 1
    w1, R1 = kin.fk(ik.q)[0], kin.fk(ik.q)[1]
    assert np.linalg.norm(w1 - w0) < 0.01                          # the arm held (hands_gain = 0)
    rel = R1 @ R0.T                                                # how the wrist link turned
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert 28 < ang < 42                                           # about the 35 deg the hand turned
    np.testing.assert_allclose(rt.fingers[side], np.full(6, 0.3), atol=0.2)
    assert rt.control == protocol.HANDS


def test_back_to_reach_does_not_jump(retargeter_after_reach):
    cfg, rt, t = retargeter_after_reach
    side = 'right'
    kin, ik = rt.kin[side], rt.ik[side]
    before = ik.q.copy()
    rt.step(_frame(999, protocol.REACH, [0.15, -0.05, -0.35], None), now=t)
    assert rt.control == protocol.REACH
    assert np.abs(ik.q - before).max() < 0.2                       # one frame later the arm has not leapt anywhere


@pytest.mark.slow
def test_pipeline_hands_only_mode_on_the_synthetic_operator():
    cfg = load_config()
    rig, results = bench.run_synthetic(cfg, seconds=40.0, rig_kwargs=dict(tilt_deg=10.0))
    pipe = rig.pipe
    assert pipe.set_control(protocol.HANDS)
    errs, n_tracked = [], 0
    for _ in range(150):
        out = pipe.process(rig.read())
        assert out.control == protocol.HANDS and out.mode == protocol.TRACKING
        tr = rig.truth
        for s, a in out.arms.items():
            if a.status == protocol.TRACKING and a.hand_R is not None:
                n_tracked += 1
                errs.append(np.degrees(np.arccos(np.clip((np.trace(a.hand_R.T @ tr.hand_R[s]) - 1) / 2, -1, 1))))
                assert a.fingers is not None and a.tips is not None and a.wrist is not None
    assert n_tracked > 100
    assert np.percentile(errs, 90) < 25
    assert pipe.set_control(protocol.REACH)
    assert pipe.process(rig.read()).control == protocol.REACH
