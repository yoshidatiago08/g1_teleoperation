"""Recording what MediaPipe sees, replaying it through the VM, and the offline analysis."""
import threading
import time

import numpy as np
import pytest

pytest.importorskip('scipy')          # the VM image has it; the notebook image does not need the hand model

from dex_hand import analysis, record
from dex_hand.sources import Observation
from dex_hand.synthetic import FACING_CAMERA, make_hand

from test_hand_model import F, CX, CY, WRIST, endon, observe


def make_recording(path, rng):
    rec = record.Recorder(str(path))
    t = 1000.0
    flat = make_hand('right', [0.0] * 5)
    for _ in range(48):                                              # 1.6 s of the calibration pose
        w, p = observe(rng, flat, np.array([0.0, 0.0, 0.7]))
        rec.add(Observation(w, p, 0.7, None), F, CX, CY, None, True, t)
        t += 1 / 30
    hand = make_hand('right', [0.0] * 5, endon((0.2, 0.15)))
    for k in range(60):                                              # then the hand points at the camera
        w, p = observe(rng, hand, WRIST + 0.02 * np.array([np.sin(k / 5), 0, 0]))
        rec.add(Observation(w, p, None, None), F, CX, CY, None, False, t)
        t += 1 / 30
    rec.add(Observation(None, None, None, None), F, CX, CY, None, False, t)        # a frame without a hand
    rec.save()


def test_a_recording_is_analysed_into_a_report(tmp_path):
    path = tmp_path / 'take.npz'
    make_recording(path, np.random.default_rng(1))
    data = record.load(path)
    assert data['world'].shape == (109, 21, 3) and data['seen'].sum() == 108 and data['calibrating'].sum() == 48
    result = analysis.analyse(data, 'right', hand_length=0.185)
    assert result['ok'].mean() > 0.95
    moving = result['t'] > 1000.0 + 49 / 30
    assert abs(np.median(result['model_z'][moving]) - WRIST[2]) < 0.04       # end-on, no sensor: the model finds the distance
    text = analysis.report(result)
    assert 'hand: 18.' in text and 'palm rotation, model vs raw' in text
    analysis.plot(result, str(tmp_path / 'take.png'))
    assert (tmp_path / 'take.png').stat().st_size > 1000


def test_the_vm_replays_a_recording_including_its_calibration(tmp_path):
    pytest.importorskip('zmq')
    pytest.importorskip('dex_retargeting')
    pytest.importorskip('mujoco')
    from dex_hand import link
    from dex_hand.vm import VmApp
    path = tmp_path / 'take.npz'
    make_recording(path, np.random.default_rng(2))
    port = 5793
    app = VmApp('right', 'replay', port=port, render=False, replay=str(path), hand_length=0.185)
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', port)
    try:
        seen, end = set(), time.time() + 14.0
        while time.time() < end:
            message = state_in.recv(100)
            if message:
                seen.add(message['state']['calibration']['state'])
                if message['state'].get('depth_from') == 'hand model':
                    break
        assert 'collecting' in seen and 'done' in seen
        assert app.shape is not None and app.shape.hand_length == pytest.approx(0.185, abs=0.01)
    finally:
        stop.set()
        thread.join(3)
        state_in.close()
        events_out.close()
