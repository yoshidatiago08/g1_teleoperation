"""The notebook <-> VM split: the link, frame encoding, and the whole VM app driven like the notebook drives it."""
import threading
import time

import numpy as np
import pytest

pytest.importorskip('zmq')
pytest.importorskip('msgpack')
pytest.importorskip('dex_retargeting')
pytest.importorskip('mujoco')

from dex_hand import capture, link
from dex_hand.sources import wrist_depth

PORT = 5731


def test_channels_deliver_and_keep_only_the_newest():
    frames_in, state_out, events_in = link.vm_side(PORT)
    frames_out, state_in, events_out = link.notebook_side('127.0.0.1', PORT)
    try:
        for _ in range(50):                                   # wait for the PUB/SUB connection like a stream would
            frames_out.send({'i': -1})
            state_out.send({'x': 1})
            if frames_in.recv(100) is not None:
                break
        for i in range(5):                                    # a camera-like stream: the receiver reads late
            frames_out.send({'i': i})
            time.sleep(0.02)
        events_out.send({'cmd': 'clutch'})
        time.sleep(0.2)
        assert frames_in.recv(500)['i'] == 4                  # conflated: the newest frame only
        assert events_in.recv(500) == {'cmd': 'clutch'}
        assert state_in.recv(500) == {'x': 1}
        assert frames_in.recv(50) is None
    finally:
        for c in (frames_in, state_out, events_in, frames_out, state_in, events_out):
            c.close()


def test_frame_encoding_roundtrip_with_depth():
    bgr = (np.random.RandomState(0).rand(48, 64, 3) * 255).astype(np.uint8)
    depth = np.full((48, 64), 1500, np.uint16)                # 1.5 m at 1 mm per unit
    frame = capture.Frame(bgr, depth, 600.0, 32.0, 24.0, 0.001)
    message = capture.encode(frame, t=1.0)
    out_bgr, out_depth = capture.decode(message)
    assert out_bgr.shape == bgr.shape and out_depth.shape == (24, 32)         # depth sent at half size
    assert wrist_depth(out_depth, (30.0, 20.0), message['depth_scale'], message['depth_downscale']) == pytest.approx(1.5)
    assert capture.decode(capture.encode(capture.Frame(bgr, None, 600.0, 32.0, 24.0), 1.0))[1] is None
    with_up = capture.encode(capture.Frame(bgr, None, 600.0, 32.0, 24.0, up=np.array([0.0, -1.0, 0.0])), 1.0)
    assert with_up['up'] == [0.0, -1.0, 0.0] and 'up' not in message


def test_wrist_depth_ignores_invalid_pixels():
    depth = np.zeros((20, 20), np.uint16)
    assert wrist_depth(depth, (10, 10), 0.001) is None


def start_vm(port, calibrated):
    from dex_hand.vm import VmApp
    app = VmApp('right', 'synthetic', port=port, render=False)
    if calibrated:
        app._install_calibration(np.eye(3), 0.0)
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', port)
    time.sleep(0.5)
    return app, state_in, events_out, stop, thread


@pytest.fixture
def vm():
    """A VM that has not been calibrated yet."""
    app, state_in, events_out, stop, thread = start_vm(PORT + 10, calibrated=False)
    yield app, state_in, events_out
    stop.set()
    thread.join(3)
    state_in.close()
    events_out.close()


@pytest.fixture
def calibrated_vm():
    app, state_in, events_out, stop, thread = start_vm(PORT + 20, calibrated=True)
    yield app, state_in, events_out
    stop.set()
    thread.join(3)
    state_in.close()
    events_out.close()


def latest(state_in, wait=0.5):
    end, message = time.time() + wait, None
    while time.time() < end:
        message = state_in.recv(50) or message
    return message['state']


def test_vm_streams_state_with_scaled_goal_and_fingers(calibrated_vm):
    app, state_in, _ = calibrated_vm
    state = latest(state_in, 1.5)
    assert state['tracking'] and state['engaged']
    assert len(state['joints']) == 12 and len(state['skeleton']) == 21
    assert state['scene']['to_target_cm'] > 0
    goal, ghost = np.array(state['goal_pos']), np.array(state['ghost']['pos'])
    from dex_hand.vm import HOME
    home = np.array(HOME)
    assert np.linalg.norm(goal - home) <= np.linalg.norm(ghost - home) * 0.3 + 1e-6    # scaled 0.3 of the unscaled motion


def test_pause_freezes_the_wrist_and_the_fingers_and_resume_continues(calibrated_vm):
    app, state_in, events_out = calibrated_vm
    latest(state_in, 0.5)
    events_out.send({'cmd': 'clutch'})
    time.sleep(0.4)
    a = latest(state_in, 0.3)
    time.sleep(1.0)
    b = latest(state_in, 0.3)
    assert a['status'] == 'PAUSED' and not b['imitating']
    assert a['goal_pos'] == b['goal_pos'] and a['joints'] == b['joints']      # nothing moves, fingers included
    events_out.send({'cmd': 'clutch'})                                         # resume
    time.sleep(0.4)
    c = latest(state_in, 1.0)
    assert c['status'] == 'IMITATING' and c['imitating']
    time.sleep(0.8)
    d = latest(state_in, 0.3)
    assert d['joints'] != b['joints']                                          # it follows the hand again


def test_nothing_moves_before_the_calibration(vm):
    from dex_hand.vm import HOME
    app, state_in, _ = vm
    a = latest(state_in, 0.5)
    time.sleep(1.5)
    b = latest(state_in, 0.3)
    assert b['status'] == 'PRESS C TO CALIBRATE' and not b['imitating']
    assert np.allclose(b['goal_pos'], HOME) and a['joints'] == b['joints']
    assert all(v == 0.0 for v in b['joints'].values())                         # open fingers, at home


def test_finishing_the_calibration_starts_the_imitation(vm):
    app, state_in, _ = vm
    assert not latest(state_in, 0.3)['imitating']
    app._install_calibration(np.eye(3), 0.05)
    time.sleep(0.6)
    state = latest(state_in, 0.8)
    assert state['calibration']['state'] == 'done' and state['imitating'] and state['engaged']


def test_scale_event_changes_the_gain(vm):
    app, state_in, events_out = vm
    before = latest(state_in, 0.3)['scale']
    events_out.send({'cmd': 'scale', 'factor': 0.5})
    time.sleep(0.3)
    assert latest(state_in, 0.3)['scale'] == pytest.approx(before * 0.5)


def test_simulator_renders_when_a_gl_backend_exists():
    from dex_hand import paths
    from dex_hand.mujoco_scene import FloatingHandScene
    scene = FloatingHandScene(paths.urdf('right'), 'right')
    try:
        image = scene.render(160, 120)
    except Exception as error:                                # no EGL/OSMesa on this machine
        pytest.skip(f'no offscreen GL here: {error}')
    assert image.shape == (120, 160, 3) and image.std() > 1.0


def test_the_cylinder_is_left_alone_when_the_hand_first_appears(calibrated_vm):
    """Regression: a first target that put curled fingers inside the cylinder launched it (and it fell out of the world)."""
    app, state_in, _ = calibrated_vm
    end = time.time() + 5.0
    while time.time() < end:
        scene = latest(state_in, 0.5)['scene']
        assert scene['cyl_pos'][2] > 0.85 and abs(scene['to_target_cm'] - 18.0) < 1.0, scene


def test_calibrate_event_reports_progress_and_ends_in_done_or_failed(vm):
    app, state_in, events_out = vm
    events_out.send({'cmd': 'calibrate'})
    seen = set()
    end = time.time() + 9.0                                  # 1.5 s of measuring, then the hand model is fitted to the hand
    while time.time() < end:
        message = state_in.recv(100)
        if message:
            seen.add(message['state']['calibration']['state'])
    assert 'collecting' in seen and seen & {'done', 'failed'}   # the synthetic hand rotates: either is a valid outcome


def test_set_scale_event_sets_both_gains(vm):
    app, state_in, events_out = vm
    events_out.send({'cmd': 'set_scale', 'motion': 0.12, 'rotation': 0.5})
    time.sleep(0.4)
    state = latest(state_in, 0.3)
    assert state['scale'] == pytest.approx(0.12) and state['rotation_scale'] == pytest.approx(0.5)
