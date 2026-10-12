"""End to end on a synthetic operator: calibration, tracking accuracy, the closed loop."""
import numpy as np
import pytest

from g1tele import bench, protocol
from g1tele.config import load_config
from g1tele.perception.synthetic import SyntheticOperator

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def run():
    cfg = load_config(overrides={'perception': {'reach_hands': True}})      # the hand model is measured here too
    # MediaPipe-like noise: 4 px landmark jitter and noisier depth than a clean RealSense
    rig, results = bench.run_synthetic(cfg, seconds=60.0, rig_kwargs=dict(tilt_deg=10.0, pixel_noise=4.0, depth_noise=0.004))
    return cfg, rig, results


def test_calibration_recovers_bones_and_up(run):
    cfg, rig, results = run
    p = rig.pipe.profile
    for side in ('right', 'left'):
        assert abs(p.bones[side].upper - 0.30) < 0.012 and abs(p.bones[side].fore - 0.27) < 0.012
    assert p.up_source == 'floor'
    assert np.degrees(np.arccos(np.clip(p.up @ rig.up_cam, -1, 1))) < 1.5
    assert np.degrees(np.arccos(np.clip((np.trace(p.frame.T @ rig.R_cb) - 1) / 2, -1, 1))) < 1.5
    assert abs(p.shoulder_width - 0.38) < 0.02
    assert rig.pipe.drift_events == [] and rig.pipe.rejections == 0     # a healthy run triggers no recalibration
    assert p.hand_open != list(__import__('g1tele.hand', fromlist=['x']).DEFAULT_OPEN)   # learnt this person's range


def test_tracking_accuracy(run):
    cfg, rig, results = run
    m = bench.metrics_synthetic(rig, results)
    assert m['wrist_p95_mm'] < 50 and m['elbow_p95_mm'] < 100     # the elbow is a swivel hint when it is hidden
    assert m['wrist_rms_mm'] < 28
    assert m['still_jitter_mm'] < 12, 'a still arm must not shake'
    assert m['finger_mae'] < 0.12
    assert m['tracking_pct'] > 85
    assert m['frame_ms_p95'] < 25                   # leaves room for MediaPipe at 30-60 fps


def test_calibration_is_saved_and_reused_quickly(run, tmp_path):
    cfg, rig, results = run
    from g1tele.perception.calibration import Profile
    from g1tele.perception.pipeline import PerceptionPipeline
    from g1tele.perception.synthetic import SyntheticRig
    prof = rig.pipe.profile
    prof.save(str(tmp_path), 'tester')
    loaded = Profile.load(str(tmp_path), 'tester')
    assert abs(loaded.bones['right'].upper - prof.bones['right'].upper) < 1e-9
    rig2 = SyntheticRig(SyntheticOperator(poses=('rest',), hand_poses=False), tilt_deg=10.0)
    pipe2 = PerceptionPipeline(cfg, rig2.pose, rig2.hands, loaded)
    assert pipe2.calibrator.stages == ['rest']      # only the rest pose is repeated
    n = 0
    while pipe2.calibrating and n < 600:
        pipe2.process(rig2.read())
        n += 1
    assert not pipe2.calibrating and n < 300        # a few seconds


def test_closed_loop_follows_the_operator():
    cfg = load_config()
    m = bench.run_closed_loop(cfg, seconds=42.0)
    assert m['robot_wrist_vs_truth_p95_mm'] < 40       # vs the bone-scaled truth; the robot uses a wrist-only mapping when the elbow is hidden
    assert m['end_to_end_lag_ms'] <= 130
    assert m['ik_error_mm_mean'] < 3
    assert m['tracking_pct'] > 80


def test_episode_can_be_re_retargeted_offline(run, tmp_path):
    """Record a tracked stretch, replay it through the retargeter with another setting."""
    from g1tele.cli import replay
    from g1tele.record import EpisodeRecorder
    from g1tele.retarget.retargeter import Retargeter
    cfg, rig, results = run
    cfg = load_config(overrides={'robot': {'backend': 'kinematic'}})
    rt = Retargeter(cfg)
    rec = EpisodeRecorder(str(tmp_path), task='demo')
    rec.start()
    tracked = [o for o, _ in results if o.mode == protocol.TRACKING][:90]
    for f in tracked:
        rec.add(f.to_bytes(), rt.step(f, now=f.t_capture), {}, None)
    ep = rec.stop()
    replay.main([ep, '--set', 'robot.backend=kinematic', '--set', 'ik.elbow_weight=0.1', '--out', str(tmp_path / 'v2')])
    import json, os
    out = [d for d in os.listdir(tmp_path / 'v2')]
    assert out and os.path.exists(tmp_path / 'v2' / out[0] / 'episode_0000' / 'data.json')
    data = json.load(open(tmp_path / 'v2' / out[0] / 'episode_0000' / 'data.json'))
    assert len(data['data']) == len(tracked)


def test_arms_pointing_at_the_camera_do_not_glitch():
    """The hardest case for one camera: both forearms reach toward it (hands at table height), landmarks of the
    hidden elbow are erratic. The robot's wrist must stay steady: no shaking, no jumps."""
    import tempfile
    from g1tele.retarget.retargeter import Retargeter
    from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig
    from g1tele.perception.pipeline import PerceptionPipeline
    cfg = load_config(overrides={'calibration': {'profile_dir': tempfile.mkdtemp()}, 'robot': {'backend': 'kinematic'}})
    op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=True, scenario='table')
    rig = SyntheticRig(op, tilt_deg=10, pixel_noise=4, depth_noise=0.004, erratic=True, seed=1)
    pipe, rt = PerceptionPipeline(cfg, rig.pose, rig.hands), Retargeter(cfg)
    pos = {'right': [], 'left': []}
    for _ in range(int(55 / rig.dt)):
        f = rig.read()
        out = pipe.process(f)
        tg = rt.step(out, now=f.t_capture)
        if out.mode == protocol.TRACKING and rig.truth.t > op.free_start + 4:
            for s in pos:
                if tg.status[s] == protocol.TRACKING:
                    k = rt.kin[s]
                    pos[s].append(k.fk(np.array([tg.arm_q[n] for n in k.names]))[0])
    for s, p in pos.items():
        a = np.array(p)
        assert len(a) > 500
        noise = np.sqrt(np.mean(np.sum(np.diff(a, n=3, axis=0) ** 2, axis=1)) / 20)
        assert noise < 0.006, f'{s} wrist shakes: {1000 * noise:.1f} mm'              # was 12 mm before the end-on handling
        assert np.max(np.linalg.norm(np.diff(a, axis=0), axis=1)) < 0.04, f'{s} wrist jumps'   # was 98 mm


def test_hand_orientation_survives_the_other_hand_and_flipped_palms():
    """Hands on the table converge, so one crop holds both and the hand model may return the other one or a flipped
    palm. The robot's wrist orientation must not follow those readings."""
    import sys, tempfile
    from g1tele.geometry import rotation_angle
    from g1tele.hand import robot_hand_in_link
    from g1tele.retarget.retargeter import Retargeter
    from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig
    from g1tele.perception.pipeline import PerceptionPipeline
    cfg = load_config(overrides={'calibration': {'profile_dir': tempfile.mkdtemp()}, 'robot': {'backend': 'kinematic'},
                                       'perception': {'reach_hands': True}})      # this test is about the orientation path, which is optional now
    op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=True, scenario='table', converge=True)
    rig = SyntheticRig(op, tilt_deg=10, pixel_noise=4, depth_noise=0.004, erratic=True, hand_glitch=True, seed=3)
    pipe, rt = PerceptionPipeline(cfg, rig.pose, rig.hands), Retargeter(cfg)
    err = []
    for _ in range(int(55 / rig.dt)):
        f = rig.read()
        out = pipe.process(f)
        tg = rt.step(out, now=f.t_capture)
        if out.mode == protocol.TRACKING and rig.truth.t > op.free_start + 4:
            for s in ('right', 'left'):
                if tg.status[s] == protocol.TRACKING:
                    k = rt.kin[s]
                    r = k.fk(np.array([tg.arm_q[n] for n in k.names]))[1] @ robot_hand_in_link(s)
                    err.append(np.degrees(rotation_angle(r, rig.truth.hand_R[s])))
    err = np.array(err)
    assert np.percentile(err, 95) < 25, 'the wrist follows wrong-hand / flipped readings'
    assert np.mean(err > 60) < 0.005
