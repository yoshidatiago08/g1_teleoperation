import numpy as np

from g1tele import protocol
from g1tele.config import load_config
from g1tele.protocol import ArmObs, BoneLengths, OperatorFrame
from g1tele.retarget.retargeter import Retargeter


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def run(elbow_dir, fore_dir, quality=1.0, frames=150):
    cfg = load_config(overrides={'robot': {'backend': 'kinematic'}})
    rt = Retargeter(cfg)
    u, f = unit(elbow_dir), unit(fore_dir)
    elbow = 0.30 * u
    wrist = elbow + 0.27 * f
    arms = {s: ArmObs(status=protocol.TRACKING, quality=1.0, wrist=wrist.copy(), elbow=elbow.copy(), elbow_quality=quality)
            for s in ('right', 'left')}
    arms['left'].wrist, arms['left'].elbow = wrist * np.array([1, -1, 1]), elbow * np.array([1, -1, 1])
    t = 0.0
    out = None
    for i in range(frames):
        out = rt.step(OperatorFrame(seq=i, t_capture=t, mode=protocol.TRACKING, arms=arms,
                                    bones={s: BoneLengths(0.30, 0.27) for s in arms}), now=t)
        t += 1 / 30
    return out


def test_the_robot_copies_your_arm_directions_and_says_how_well():
    out = run([0.2, -0.3, -0.9], [0.8, -0.1, -0.5])
    assert set(out.dir_error) == {'right', 'left'}
    for upper, fore in out.dir_error.values():
        assert upper < 8.0 and fore < 8.0, out.dir_error


def test_no_number_when_the_elbow_is_not_trusted():
    out = run([0.2, -0.3, -0.9], [0.8, -0.1, -0.5], quality=0.2)
    assert out.dir_error == {}
