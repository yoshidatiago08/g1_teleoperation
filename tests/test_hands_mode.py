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


def test_hold_keeps_the_whole_robot_still_for_as_long_as_it_lasts(retargeter_after_reach):
    cfg, rt, t = retargeter_after_reach
    q = {s: rt.ik[s].q.copy() for s in rt.sides}
    hand = {s: rt.fingers[s].copy() for s in rt.sides}
    for i in range(int(12 * 30)):                                   # 12 s: longer than robot.lost_rest_after
        frame = OperatorFrame(seq=2000 + i, t_capture=t, mode=protocol.TRACKING, arms={}, control=protocol.HOLD)
        rt.step(frame, now=t)
        t += 1 / 30
    for s in rt.sides:
        np.testing.assert_allclose(rt.ik[s].q, q[s], atol=1e-9)     # no drift towards the rest pose
        np.testing.assert_allclose(rt.fingers[s], hand[s])


class HeadPose:
    """The synthetic pose plus a nose and two ears, with a scripted nod."""

    def __init__(self, inner, rig, state):
        self.inner, self.rig, self.state = inner, rig, state

    def process(self, rgb):
        res = self.inner.process(rgb)
        t = self.rig.index * self.rig.dt
        dy = 0.0
        for start in self.state['nods']:
            if start <= t < start + 0.5:
                dy = 0.25 * np.sin(np.pi * (t - start) / 0.5)
        res.uv[0] = [300.0, 200.0 + 40.0 * dy]
        res.uv[7], res.uv[8] = [280.0, 200.0], [320.0, 200.0]
        res.vis[[0, 7, 8]] = 0.99
        return res


@pytest.mark.slow
def test_two_nods_run_the_whole_cycle_reach_calibrate_hands_match_reach(monkeypatch):
    from g1tele.perception import hands_calib
    monkeypatch.setattr(hands_calib, 'STILL_M', 10.0)              # the scripted operator keeps moving its arms
    cfg = load_config(overrides={'perception': {'hands_calib_range': [0.3, 3.0], 'hands_hold_s': 0.4,
                                                'match_tol_m': 5.0, 'match_dwell_s': 0.3, 'nod_cooldown_s': 1.0}})
    rig, _ = bench.run_synthetic(cfg, seconds=40.0, rig_kwargs=dict(tilt_deg=10.0))
    pipe = rig.pipe
    state = {'c': 0.0, 'nods': []}
    rig.op.curls = lambda t: np.full(6, state['c'])
    pipe.pose = HeadPose(rig.pose, rig, state)

    def run(seconds, drive=True):
        outs = []
        for _ in range(int(seconds / rig.dt)):
            if drive and pipe.enter_cal is not None:
                state['c'] = 1.0 if pipe.enter_cal.name == 'fist' else 0.0
            outs.append(pipe.process(rig.read()))
        return outs

    outs = run(1.0)
    assert pipe.stage == 'reach' and all(o.control == protocol.REACH for o in outs)
    now = rig.index * rig.dt
    state['nods'] = [now + 0.2, now + 1.0]                          # two nods
    outs = run(3.0)
    assert pipe.stage == 'enter' and any(o.control == protocol.HOLD for o in outs)
    assert set(pipe.entry_wrist) == {'right', 'left'}               # where the wrists were when the robot froze
    outs = run(20.0)
    assert pipe.stage == 'hands', pipe.enter_cal.name
    assert any(o.control == protocol.HOLD and o.prompt for o in outs)       # the calibration told the operator what to do
    ht = pipe.hand_tracks['right']
    assert ht.open[0] < 1.3 and ht.closed[0] - ht.open[0] > 1.5      # this person's close-range finger range was learnt
    outs = run(2.0)
    assert all(o.control == protocol.HANDS for o in outs)
    assert any(a.hand_R is not None and a.fingers is not None for o in outs for a in o.arms.values())
    now = rig.index * rig.dt
    state['nods'] = [now + 0.3, now + 1.1]                          # two nods again: leave
    outs = run(3.0)
    assert pipe.stage in ('exit', 'reach')
    assert any(o.control == protocol.HOLD for o in outs)
    outs = run(8.0)
    assert pipe.stage == 'reach' and outs[-1].control == protocol.REACH


@pytest.mark.slow
def test_a_calibration_that_cannot_finish_gives_up_and_the_robot_stays_put(monkeypatch):
    cfg = load_config(overrides={'perception': {'hands_calib_range': [0.3, 0.9], 'hands_calib_timeout_s': 5.0}})
    rig, _ = bench.run_synthetic(cfg, seconds=40.0, rig_kwargs=dict(tilt_deg=10.0))
    pipe = rig.pipe                                                 # the hands are 1.8 m away: "move closer" never succeeds
    pipe.request_toggle()
    outs = [pipe.process(rig.read()) for _ in range(int(9 / rig.dt))]
    assert outs[0].control == protocol.HOLD and 'closer' in outs[3].prompt
    assert pipe.stage in ('exit', 'reach')                          # it gave up and went to the match stage
    assert all(o.control in (protocol.HOLD, protocol.REACH) for o in outs)
