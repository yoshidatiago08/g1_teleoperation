import numpy as np

from g1tele import protocol
from g1tele.config import load_config
from g1tele.perception.calibration import Profile
from g1tele.perception.camera import RGBDRecorder, ReplayCamera
from g1tele.perception.pipeline import PerceptionPipeline
from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig
from g1tele.protocol import BoneLengths


def test_camera_session_roundtrip(tmp_path):
    rig = SyntheticRig(SyntheticOperator(), tilt_deg=10.0)
    prof = Profile(bones={'right': BoneLengths(0.3, 0.27), 'left': BoneLengths(0.3, 0.27)}, shoulder_width=0.38)
    prof.frame, prof.up, prof.up_source = np.eye(3), np.array([0, -1.0, 0]), 'floor'
    rec = RGBDRecorder(str(tmp_path / 's'), {'some': 'config'})
    frames = []
    for _ in range(6):
        f = rig.read()
        frames.append(f)
        rec.set_start_profile(prof.to_full())
        rec.write(f, b'live-bytes')
    rec.close(prof.to_full())
    cam = ReplayCamera(str(tmp_path / 's'), realtime=False)
    assert len(cam) == 6 and cam.meta['config'] == {'some': 'config'}
    for f in frames:
        g = cam.read()
        valid = f.depth > 0
        assert np.max(np.abs(g.depth[valid] - f.depth[valid])) < 0.0011           # 2 mm steps
        assert g.t_capture == f.t_capture and g.color.shape == f.color.shape
    p = Profile.from_full(cam.meta['profile_start'])
    assert p.frame is not None and abs(p.bones['right'].upper - 0.3) < 1e-9 and p.up_source == 'floor'


def test_a_recording_that_began_after_calibration_replays_without_recalibrating():
    cfg = load_config()
    rig = SyntheticRig(SyntheticOperator(), tilt_deg=10.0)
    prof = Profile(bones={'right': BoneLengths(0.3, 0.27), 'left': BoneLengths(0.3, 0.27)}, shoulder_width=0.38)
    prof.frame = np.column_stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])
    prof.up = np.array([0, -1.0, 0])
    pipe = PerceptionPipeline(cfg, rig.pose, rig.hands, prof, calibrated=True)
    assert not pipe.calibrating and pipe.calibrator is None
    out = pipe.process(rig.read())
    assert out.mode == protocol.TRACKING
