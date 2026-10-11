"""A bad calibration must be caught: before it is used, and while tracking."""
import numpy as np
import pytest

from g1tele import protocol
from g1tele.config import load_config
from g1tele.perception.calibration import Profile, validate_profile
from g1tele.perception.pipeline import PerceptionPipeline
from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig
from g1tele.protocol import BoneLengths


def good_profile():
    p = Profile(bones={'right': BoneLengths(0.30, 0.27), 'left': BoneLengths(0.30, 0.27)}, shoulder_width=0.38)
    p.frame = np.column_stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])
    p.up = np.array([0, -1.0, 0])
    return p


def test_validation_accepts_a_sane_calibration():
    assert validate_profile(good_profile(), ['right', 'left']) == []


@pytest.mark.parametrize('mutate, word', [
    (lambda p: p.bones.update(right=BoneLengths(0.55, 0.27)), 'upper arm'),
    (lambda p: p.bones.update(left=BoneLengths(0.30, 0.08)), 'forearm'),
    (lambda p: p.report.update(asymmetry_cm=6.0), 'differ'),
    (lambda p: p.report.update(bone_spread_cm=9.0), 'changed between poses'),
    (lambda p: setattr(p, 'shoulder_width', 0.9), 'shoulder width'),
    (lambda p: setattr(p, 'frame', np.eye(3)), 'facing'),
    (lambda p: setattr(p, 'up', np.array([1.0, 0, 0])), 'up'),
    (lambda p: setattr(p, 'frame', None), 'shoulders'),
])
def test_validation_rejects_a_bad_calibration(mutate, word):
    p = good_profile()
    mutate(p)
    problems = validate_profile(p, ['right', 'left'])
    assert problems and any(word in x for x in problems), problems


def test_an_implausible_saved_profile_is_not_reused(tmp_path):
    cfg = load_config()
    rig = SyntheticRig(SyntheticOperator(), tilt_deg=10.0)
    bad = Profile(bones={'right': BoneLengths(0.70, 0.60), 'left': BoneLengths(0.70, 0.60)}, shoulder_width=0.38)
    pipe = PerceptionPipeline(cfg, rig.pose, rig.hands, bad)
    assert pipe.calibrator.full and 'ignored' in pipe.notice       # does the whole calibration, not a quick one


@pytest.mark.slow
def test_calibration_that_stops_matching_is_redone_automatically(tmp_path):
    """The operator's real arm lengths change after calibration (wrong calibration, or another person
    stepping in): the monitor must notice and start a new calibration by itself."""
    cfg = load_config(overrides={'calibration': {'profile_dir': str(tmp_path)}})
    op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=cfg.calibration.hand_poses)
    rig = SyntheticRig(op, tilt_deg=10.0)
    pipe = PerceptionPipeline(cfg, rig.pose, rig.hands)
    changed_at = None
    for _ in range(int(80 / rig.dt)):
        frame = rig.read()
        if changed_at is None and pipe.profile is not None and not pipe.calibrating and rig.truth.t > 40:
            op.upper, op.fore, changed_at = 0.40, 0.36, rig.truth.t
        out = pipe.process(frame)
        if pipe.drift_events:
            break
    assert pipe.drift_events and pipe.drift_events[0][1] == 'bones'
    assert pipe.drift_events[0][0] - changed_at < 25
    assert out.mode == protocol.CALIBRATING and 'Recalibrating' in out.prompt
