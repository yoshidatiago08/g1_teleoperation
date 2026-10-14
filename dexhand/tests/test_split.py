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


def test_wrist_depth_ignores_invalid_pixels():
    depth = np.zeros((20, 20), np.uint16)
    assert wrist_depth(depth, (10, 10), 0.001) is None


@pytest.fixture
def vm():
    from dex_hand.vm import VmApp
    app = VmApp('right', 'synthetic', port=PORT + 10, render=False)
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', PORT + 10)
    time.sleep(0.5)
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


def test_vm_streams_state_with_scaled_goal_and_fingers(vm):
    app, state_in, _ = vm
    state = latest(state_in, 1.5)
    assert state['tracking'] and state['engaged']
    assert len(state['joints']) == 12 and len(state['skeleton']) == 21
    assert state['scene']['to_target_cm'] > 0
    goal, ghost = np.array(state['goal_pos']), np.array(state['ghost']['pos'])
    from dex_hand.vm import HOME
    home = np.array(HOME)
    assert np.linalg.norm(goal - home) <= np.linalg.norm(ghost - home) * 0.3 + 1e-6    # scaled 0.3 of the unscaled motion


def test_clutch_event_freezes_the_goal_but_not_the_fingers(vm):
    app, state_in, events_out = vm
    latest(state_in, 0.5)
    events_out.send({'cmd': 'clutch'})
    time.sleep(0.4)
    a = latest(state_in, 0.3)
    time.sleep(1.0)
    b = latest(state_in, 0.3)
    assert a['clutch'] and not b['engaged']
    assert np.allclose(a['goal_pos'], b['goal_pos'])
    assert a['joints'] != b['joints']                         # the synthetic fingers keep opening and closing


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


def test_the_cylinder_is_left_alone_when_the_hand_first_appears(vm):
    """Regression: a first target that put curled fingers inside the cylinder launched it (and it fell out of the world)."""
    app, state_in, _ = vm
    end = time.time() + 5.0
    while time.time() < end:
        scene = latest(state_in, 0.5)['scene']
        assert scene['cyl_pos'][2] > 0.85 and abs(scene['to_target_cm'] - 18.0) < 1.0, scene
