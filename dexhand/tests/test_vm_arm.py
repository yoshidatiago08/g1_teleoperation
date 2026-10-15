"""The whole arm mode: the VM app with CuRobo IK and the G1 arm in MuJoCo, driven by the synthetic hand (needs a CUDA GPU: the VM image)."""
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('curobo')
pytest.importorskip('zmq')
pytest.importorskip('dex_retargeting')
pytest.importorskip('mujoco')
if not torch.cuda.is_available():
    pytest.skip('no CUDA GPU', allow_module_level=True)

from dex_hand import link
from dex_hand.vm import HOME_ARM, VmApp

PORT = 5861


@pytest.fixture(scope='module')
def arm_vm():
    app = VmApp('right', 'synthetic', port=PORT, render=False, mode='arm')
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', PORT)
    time.sleep(0.5)
    yield app, state_in, events_out
    stop.set()
    thread.join(5)
    state_in.close()
    events_out.close()


def latest(state_in, wait=0.5):
    end, message = time.time() + wait, None
    while time.time() < end:
        message = state_in.recv(50) or message
    return message['state']


def test_before_the_calibration_the_arm_is_at_home_and_does_not_move(arm_vm):
    app, state_in, _ = arm_vm
    a = latest(state_in, 0.5)
    time.sleep(1.0)
    b = latest(state_in, 0.5)
    assert b['mode'] == 'arm' and b['status'] == 'PRESS C TO CALIBRATE'
    assert np.allclose(b['scene']['hand_pos'], HOME_ARM, atol=0.01)         # the arm starts with the hand at home
    assert a['scene']['hand_pos'] == pytest.approx(b['scene']['hand_pos'], abs=1e-3)


def test_after_calibrating_the_hand_follows_the_scaled_target_through_the_arm(arm_vm):
    app, state_in, events_out = arm_vm
    app._install_calibration(np.eye(3), 0.0)
    time.sleep(1.0)
    gaps, moved, reach_limits = [], [], 0
    end = time.time() + 6.0
    while time.time() < end:
        state = latest(state_in, 0.25)
        if state['status'].startswith('OUT OF REACH'):
            reach_limits += 1
            continue
        gaps.append(np.linalg.norm(np.array(state['scene']['hand_pos']) - np.array(state['goal_pos'])))
        moved.append(np.linalg.norm(np.array(state['scene']['hand_pos']) - HOME_ARM))
    assert len(gaps) > 8
    assert np.median(gaps) < 0.03                                          # the simulated hand is where the IK was asked to put it
    assert max(moved) > 0.03                                               # and it really moved (the synthetic hand wanders)
    assert state['arm']['error_mm'] < 50


def test_pausing_freezes_the_arm_and_resuming_continues(arm_vm):
    app, state_in, events_out = arm_vm
    app._install_calibration(np.eye(3), 0.0)
    events_out.send({'cmd': 'clutch'})
    time.sleep(0.8)
    a = latest(state_in, 0.3)
    time.sleep(1.0)
    b = latest(state_in, 0.3)
    assert a['status'] == 'PAUSED'
    assert np.allclose(a['scene']['hand_pos'], b['scene']['hand_pos'], atol=2e-3)
    events_out.send({'cmd': 'clutch'})
    assert latest(state_in, 1.0)['status'] in ('IMITATING', 'OUT OF REACH (arm holding)')


def test_a_target_beyond_the_reach_says_so_and_the_arm_holds(arm_vm):
    app, state_in, events_out = arm_vm
    app._install_calibration(np.eye(3), 0.0)
    events_out.send({'cmd': 'set_scale', 'motion': 4.0, 'rotation': 1.0})          # the synthetic hand's wandering becomes metres
    seen, end = set(), time.time() + 25.0
    while time.time() < end and 'OUT OF REACH (arm holding)' not in seen:
        seen.add(latest(state_in, 0.3)['status'])
    events_out.send({'cmd': 'set_scale', 'motion': 0.3, 'rotation': 1.0})
    assert 'OUT OF REACH (arm holding)' in seen
    assert latest(state_in, 0.5)['scene']['hand_pos'][2] > 0.82                     # nothing went through the table
