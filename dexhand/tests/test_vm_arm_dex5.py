"""The arm mode with the Dex5-1 hand (a separate module from test_vm_arm: two CUDA-graph IK solvers must not run in one process at once)."""
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
from dex_hand.vm import VmApp

PORT = 5862


def latest(state_in, wait=0.5):
    end, message = time.time() + wait, None
    while time.time() < end:
        message = state_in.recv(50) or message
    return message['state']


def test_the_dex5_hand_runs_the_whole_arm_mode_too():
    app = VmApp('right', 'synthetic', port=PORT, render=False, mode='arm', hand_model='dex5')
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', PORT)
    try:
        time.sleep(0.5)
        app._install_calibration(np.eye(3), 0.0)
        time.sleep(1.0)
        seen = [latest(state_in, 0.3) for _ in range(10)]
        assert all(s['hand_model'] == 'dex5' for s in seen)
        assert seen[-1]['status'] in ('IMITATING', 'OUT OF REACH (arm at its limit)')
        assert any('R_index_pip_joint' in s['joints'] for s in seen)                # 16-joint Dex5 fingers, not the Inspire's 6
        fingers = [s['scene']['robot_joints']['R_index_pip_joint'] for s in seen]
        assert max(fingers) >= min(fingers)
    finally:
        stop.set()
        thread.join(5)
        state_in.close()
        events_out.close()
