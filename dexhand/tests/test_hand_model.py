"""The hand model (hand_model.py): a metric hand fitted to MediaPipe's 2D landmarks, against the raw 3D landmarks.

The "camera" here is a pinhole; MediaPipe is imitated by 2D noise on every landmark and a world-landmark depth that
is good for landmarks you can see and bad for the ones hidden behind other parts of the hand."""
import threading
import time

import numpy as np
import pytest

pytest.importorskip('scipy')          # the VM image has it; the notebook image does not need the hand model

from dex_hand import frames
from dex_hand import hand_model as hm
from dex_hand.frames import rotation_from_vec
from dex_hand.synthetic import FACING_CAMERA, make_hand

F, CX, CY = 600.0, 320.0, 240.0
WRIST = np.array([0.05, -0.03, 0.7])
HAND_LENGTH = 0.185


def project(points):
    return np.column_stack([CX + F * points[:, 0] / points[:, 2], CY + F * points[:, 1] / points[:, 2]])


def observe(rng, true_world, wrist=WRIST, px=1.5, z_visible=0.01, z_hidden=0.05):
    """(MediaPipe world landmarks, pixels): 2D noise everywhere, depth noise 1 cm on visible landmarks and 5 cm on hidden ones."""
    points = true_world + wrist
    hidden = ~hm.visibility(points, F)
    sigma = np.where(hidden, z_hidden, z_visible)[:, None] * np.array([0.3, 0.3, 1.0])
    return (true_world - true_world[0]) + rng.normal(0, 1, (21, 3)) * sigma, project(points) + rng.normal(0, px, (21, 2))


def endon(tilt=(0.0, 0.0)):
    """Fingers pointing at the camera, palm up in the image."""
    f, n = np.array([0, 0, -1.0]), np.array([0, -1.0, 0])
    return rotation_from_vec([tilt[0], tilt[1], 0.0]) @ np.column_stack([f, np.cross(n, f), n])


@pytest.fixture(scope='module')
def calibration():
    """A shape measured from a noisy stretched palm-to-camera hand, and each estimator's frame offset from the true hand frame
    (the calibration absorbs that constant, so poses are compared relative to it)."""
    rng = np.random.default_rng(1)
    flat = make_hand('right', [0.0] * 5)
    samples = [observe(rng, flat, np.array([0.0, 0.0, 0.7])) for _ in range(20)]
    shape = hm.HandShape.from_samples('right', [s[0] for s in samples], hand_length=HAND_LENGTH)
    fit = hm.HandFit(shape)
    for w, p in samples:
        last = fit.update(w, p, None, F, CX, CY)
    true_frame = frames.hand_rotation(flat, 'right')
    offsets = {'model': last.rotation.T @ true_frame,
               'raw': frames.mean_rotation([frames.hand_rotation(w, 'right') for w, _ in samples]).T @ true_frame}
    return shape, offsets


def track(shape, rng, true_world, frames_n=8, wrist=WRIST, depth=None):
    fit, last = hm.HandFit(shape), None
    for _ in range(frames_n):
        w, p = observe(rng, true_world, wrist)
        last = fit.update(w, p, depth, F, CX, CY)
    return last, w, p


# -- the shape ---------------------------------------------------------------------------------------------------------
def test_the_shape_is_your_stretched_hand(calibration):
    shape, _ = calibration
    assert shape.scale_source == 'hand length' and shape.hand_length == pytest.approx(HAND_LENGTH, abs=1e-6)
    assert np.degrees(shape.bend()) < 20                                   # a stretched hand
    flat = make_hand('right', [0.0] * 5)
    truth = np.linalg.norm((flat - flat[0])[list(hm.TIPS)], axis=1)
    assert shape.reach == pytest.approx(truth, abs=0.012)                  # every finger within 12 mm


def test_the_shape_takes_its_metric_scale_from_a_depth_sensor_when_there_is_one():
    rng = np.random.default_rng(2)
    flat = make_hand('right', [0.0] * 5)
    obs = [observe(rng, flat, np.array([0.0, 0.0, 0.8]), z_visible=0.0) for _ in range(10)]
    shape = hm.HandShape.from_samples('right', [o[0] * 0.9 for o in obs], [o[1] for o in obs], [0.8] * 10, F, CX, CY)
    assert shape.scale_source == 'depth sensor'
    assert shape.hand_length == pytest.approx(HAND_LENGTH, abs=0.006)        # MediaPipe's 10% too small hand is corrected
    assert hm.HandShape.from_samples('right', [o[0] for o in obs]).scale_source == 'mediapipe'


def test_a_bent_hand_is_not_a_calibration_pose():
    rng = np.random.default_rng(3)
    fist = make_hand('right', [0.5, 1.0, 1.0, 1.0, 1.0])
    shape = hm.HandShape.from_samples('right', [observe(rng, fist)[0] for _ in range(10)])
    assert np.degrees(shape.bend()) > 40


# -- visibility --------------------------------------------------------------------------------------------------------
def test_nothing_is_hidden_when_the_open_palm_faces_the_camera():
    assert hm.visibility(make_hand('right', [0.0] * 5) + WRIST, F).all()


def test_fingers_pointing_at_the_camera_hide_the_knuckles_and_the_wrist():
    visible = hm.visibility(make_hand('right', [0.0] * 5, endon()) + WRIST, F)
    assert visible[list(hm.TIPS)].all()
    assert not visible[[0, 5, 9, 13, 17]].any()


def test_a_fist_seen_from_the_back_hides_the_fingertips_behind_the_palm():
    back_of_hand = rotation_from_vec([np.pi, 0, 0]) @ FACING_CAMERA
    visible = hm.visibility(make_hand('right', [0.5, 1.0, 1.0, 1.0, 1.0], back_of_hand) + WRIST, F)
    assert not visible[[8, 12, 16]].any()                      # (the pinky curls past the edge of the palm)


# -- the fit -----------------------------------------------------------------------------------------------------------
def test_a_pose_of_the_model_itself_is_recovered_exactly(calibration):
    shape, _ = calibration
    x = np.zeros(hm.N_PARAMS)
    x[hm.I_ABD:hm.I_ABD + 4], x[hm.I_F0:hm.I_F0 + 4], x[hm.I_F1:hm.I_F1 + 4], x[hm.I_F2:hm.I_F2 + 4] = 0.1, 0.8, 0.7, 0.5
    x[hm.I_THUMB:hm.I_THUMB + 5] = [0.2, -0.3, 0.4, 0.3, 0.3]
    points = hm.HandModel(shape).hand_points(x[None])[0]
    r = rotation_from_vec([0.3, 0.2, 0.1]) @ FACING_CAMERA
    camera = points @ r.T + WRIST
    fit = hm.HandFit(shape)
    for _ in range(12):
        out = fit.update(camera - camera[0], project(camera), None, F, CX, CY)
    assert out.ok and out.error_px < 0.2
    assert np.degrees(frames.rotation_angle(r, out.rotation)) < 2.0
    assert out.wrist == pytest.approx(WRIST, abs=0.01)
    assert fit.q[:16] == pytest.approx(x[6:22], abs=0.08)                    # finger joints
    assert np.abs(out.hand_landmarks - points)[[8, 12, 16, 20]].max() < 0.008        # the fingertips
    assert np.abs(out.hand_landmarks - points).max() < 0.02                         # and every landmark (the thumb's joints trade off against each other)


@pytest.mark.parametrize('name,rotation,curls', [
    ('end-on open', endon(), [0.0] * 5),
    ('end-on tilted, half closed', endon((0.2, 0.15)), [0.5, 0.6, 0.6, 0.6, 0.6]),
    ('end-on, closed', endon(), [0.6, 1.0, 1.0, 1.0, 1.0]),
    ('facing, half closed', FACING_CAMERA, [0.5, 0.6, 0.6, 0.6, 0.6]),
    ('facing, closed', FACING_CAMERA, [0.6, 1.0, 1.0, 1.0, 1.0]),
    ('side, closed', rotation_from_vec([0, 0, 1.2]) @ FACING_CAMERA, [0.6, 1.0, 1.0, 1.0, 1.0]),
])
def test_the_model_finds_depth_orientation_and_fingertips_where_the_raw_landmarks_fail(calibration, name, rotation, curls):
    """Averaged over noise draws. The raw method is MediaPipe's 3D landmarks (palm rotation from them, distance from the hand's apparent size)."""
    shape, offsets = calibration
    true = make_hand('right', curls, rotation)
    r_true = frames.hand_rotation(true, 'right')
    tips_true = frames.tip_vectors(true, r_true)
    errors = {'model': [], 'raw': []}
    for seed in range(5):
        fit, w, p = track(shape, np.random.default_rng(100 + seed), true)
        assert fit.ok, name
        raw_r = frames.hand_rotation(w, 'right')
        errors['model'].append((np.degrees(frames.rotation_angle(r_true, fit.rotation @ offsets['model'])),
                                abs(fit.wrist[2] - WRIST[2]) * 100,
                                np.linalg.norm(fit.tip_vectors - tips_true, axis=1).mean() * 1000))
        errors['raw'].append((np.degrees(frames.rotation_angle(r_true, raw_r @ offsets['raw'])),
                              abs(frames.depth_from_scale(w, p, F) - WRIST[2]) * 100,
                              np.linalg.norm(frames.tip_vectors(w, raw_r) - tips_true, axis=1).mean() * 1000))
    model, raw = np.mean(errors['model'], axis=0), np.mean(errors['raw'], axis=0)
    print(f'{name}: model rot {model[0]:.1f} deg depth {model[1]:.1f} cm tips {model[2]:.1f} mm | raw {raw[0]:.1f} deg {raw[1]:.1f} cm {raw[2]:.1f} mm')
    assert model[0] < 15 and model[1] < 3.5 and model[2] < 22
    assert model[1] < raw[1] + 0.5 and model[2] < raw[2] + 3


def test_the_hidden_landmarks_depth_does_not_matter(calibration):
    """Garbage MediaPipe depth (10 cm) on the landmarks that are hidden: the model hardly changes, the raw rotation does."""
    shape, offsets = calibration
    true = make_hand('right', [0.0] * 5, endon())
    r_true = frames.hand_rotation(true, 'right')
    hidden = ~hm.visibility(true + WRIST, F)
    assert hidden.sum() >= 5
    results = {}
    for label, z_hidden in (('good', 0.0), ('garbage', 0.10)):
        rng = np.random.default_rng(7)
        model, raw = [], []
        for seed in range(4):
            fit = hm.HandFit(shape)
            for _ in range(8):
                w, p = observe(rng, true, z_visible=0.005, z_hidden=z_hidden)
                out = fit.update(w, p, None, F, CX, CY)
            model.append(np.degrees(frames.rotation_angle(r_true, out.rotation @ offsets['model'])))
            raw.append(np.degrees(frames.rotation_angle(r_true, frames.hand_rotation(w, 'right') @ offsets['raw'])))
        results[label] = np.mean(model), np.mean(raw)
    assert results['garbage'][0] < results['good'][0] + 4
    assert results['garbage'][1] > results['good'][1] + 5                    # the raw palm rotation does suffer


def test_the_first_frame_is_not_fooled_by_the_mirror_image(calibration):
    """Fingers pointing at the camera, tilted: the 2D landmarks fit a hand that points away just as well. The depth of the
    landmarks that are visible in each pose tells them apart."""
    shape, offsets = calibration
    true = make_hand('right', [0.0] * 5, endon((0.2, 0.15)))
    r_true = frames.hand_rotation(true, 'right')
    worst = 0.0
    for seed in range(8):
        w, p = observe(np.random.default_rng(300 + seed), true)
        out = hm.HandFit(shape).update(w, p, None, F, CX, CY)
        worst = max(worst, np.degrees(frames.rotation_angle(r_true, out.rotation @ offsets['model'])))
        assert out.z_error < 0.03
    assert worst < 25


def test_a_depth_sensor_reading_of_the_background_behind_a_hidden_wrist_is_ignored(calibration):
    shape, _ = calibration
    true = make_hand('right', [0.0] * 5, endon())
    fit, _, _ = track(shape, np.random.default_rng(5), true, depth=2.5)      # the sensor reads the wall behind a hidden wrist
    assert fit.ok and abs(fit.wrist[2] - WRIST[2]) < 0.04
    good, _, _ = track(shape, np.random.default_rng(5), true, depth=WRIST[2])
    assert abs(good.wrist[2] - WRIST[2]) < 0.03


def test_a_hand_that_does_not_match_the_image_is_reported_unreliable(calibration):
    shape, _ = calibration
    rng = np.random.default_rng(6)
    fit = hm.HandFit(shape)
    true = make_hand('right', [0.0] * 5)
    w, p = observe(rng, true)
    assert fit.update(w, p, None, F, CX, CY).ok
    out = fit.update(w, p[rng.permutation(21)], None, F, CX, CY)             # landmarks scrambled
    assert not out.ok and out.error_px > hm.BAD_PX


def test_tracking_a_frame_is_fast_enough(calibration):
    shape, _ = calibration
    rng = np.random.default_rng(8)
    true = make_hand('right', [0.3, 0.5, 0.5, 0.5, 0.5], FACING_CAMERA)
    fit = hm.HandFit(shape)
    fit.update(*observe(rng, true), None, F, CX, CY)
    frames_ = [observe(rng, true, WRIST + 0.01 * np.array([np.sin(k), np.cos(k), 0])) for k in range(20)]
    start = time.perf_counter()
    for w, p in frames_:
        fit.update(w, p, None, F, CX, CY)
    assert (time.perf_counter() - start) / 20 < 0.06


# -- inside the pipeline and the VM -------------------------------------------------------------------------------------
def test_the_pipeline_follows_the_model_and_holds_when_it_is_unreliable(calibration):
    pytest.importorskip('dex_retargeting')
    from dex_hand import paths
    from dex_hand.motion import MotionScaler
    from dex_hand.pipeline import HandPipeline
    from dex_hand.retarget import FingerRetargeter
    from dex_hand.sources import Observation
    shape, _ = calibration
    pipe = HandPipeline('right', FingerRetargeter(paths.URDF_DIR, paths.retarget_config('right'), 'right'),
                        MotionScaler([0.35, -0.2, 0.0], np.diag([1.0, -1.0, -1.0]), 0.25), F, CX, CY)
    pipe.set_shape(shape)
    rng = np.random.default_rng(9)
    true = make_hand('right', [0.0] * 5, endon((0.2, 0.15)))
    for _ in range(8):
        cmd = pipe.step(Observation(*observe(rng, true), None, None))
    assert cmd.depth_from == 'hand model' and cmd.engaged and cmd.fit_error_px < 6
    held = cmd.position.copy()
    w, p = observe(rng, true)
    cmd = pipe.step(Observation(w, p[rng.permutation(21)], None, None))
    assert pipe.unreliable and not cmd.engaged and cmd.position == pytest.approx(held)
    assert not pipe.step(Observation(w, p, None, None)).engaged or not pipe.unreliable      # and it recovers


def test_the_vm_measures_your_hand_at_the_calibration_and_sets_the_finger_scale():
    pytest.importorskip('zmq')
    pytest.importorskip('dex_retargeting')
    pytest.importorskip('mujoco')
    from dex_hand import link
    from dex_hand.vm import VmApp
    port = 5791
    app = VmApp('right', 'synthetic', port=port, render=False, finger_scale='auto')
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', port)
    try:
        time.sleep(0.5)
        events_out.send({'cmd': 'calibrate'})
        state, end = None, time.time() + 6.0
        while time.time() < end:
            message = state_in.recv(100)
            if message:
                state = message['state']
                if state['calibration']['state'] in ('done', 'failed'):
                    break
        assert state['calibration']['state'] == 'done', state['calibration']
        assert 'depth sensor' in state['calibration']['hand'] and 'finger scale' in state['calibration']['hand']
        assert app.shape is not None and app.shape.hand_length == pytest.approx(0.185, abs=0.01)
        assert 1.0 < app.retargeter.retargeting.optimizer.scaling < 1.4
        time.sleep(1.0)
        end = time.time() + 2.0
        while time.time() < end:
            message = state_in.recv(100)
            state = message['state'] if message else state
        assert state['imitating'] and state['depth_from'] == 'hand model' and state['status'] == 'IMITATING'
    finally:
        stop.set()
        thread.join(3)
        state_in.close()
        events_out.close()


def test_the_left_hand_works_the_same_way():
    rng = np.random.default_rng(11)
    flat = make_hand('left', [0.0] * 5)
    cal = [observe(rng, flat, np.array([0.0, 0.0, 0.7])) for _ in range(15)]
    shape = hm.HandShape.from_samples('left', [c[0] for c in cal], hand_length=HAND_LENGTH)
    fit = hm.HandFit(shape)
    for w, p in cal:
        last = fit.update(w, p, None, F, CX, CY)
    offset = last.rotation.T @ frames.hand_rotation(flat, 'left')
    for rotation, curls in ((FACING_CAMERA, [0.5, 0.6, 0.6, 0.6, 0.6]), (endon((0.2, 0.15)), [0.0] * 5)):
        true = make_hand('left', curls, rotation)
        out, _, _ = track(shape, np.random.default_rng(12), true)
        assert out.ok and abs(out.wrist[2] - WRIST[2]) < 0.04
        assert np.degrees(frames.rotation_angle(frames.hand_rotation(true, 'left'), out.rotation @ offset)) < 20      # (the calibration's own tilt error, about 10 degrees for a flat palm, is in the offset)
