import random
import time

import numpy as np

from g1tele import protocol
from g1tele.config import LinkCfg
from g1tele.link import OperatorLink, RobotLink
from g1tele.protocol import ArmObs, BoneLengths, OperatorFrame


def make_frame(seq=1):
    return OperatorFrame(
        seq=seq, t_capture=time.time(), mode=protocol.TRACKING, prompt='hi',
        arms={'right': ArmObs(protocol.TRACKING, 0.9, np.array([0.1, -0.2, -0.3]), np.array([0.0, -0.1, -0.2]),
                              np.eye(3), np.linspace(0, 1, 6), 1.0),
              'left': ArmObs(status=protocol.LOST)},
        bones={'right': BoneLengths(0.3, 0.27)}, waist=np.array([0.1, 0.0, -0.05]),
        timing_ms={'total': 5.0})


def test_roundtrip():
    f = make_frame()
    g = OperatorFrame.from_bytes(f.to_bytes())
    assert g.seq == f.seq and g.mode == f.mode and g.prompt == 'hi'
    np.testing.assert_allclose(g.arms['right'].wrist, f.arms['right'].wrist, atol=1e-6)
    np.testing.assert_allclose(g.arms['right'].hand_R, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(g.arms['right'].fingers, np.linspace(0, 1, 6), atol=1e-6)
    assert g.arms['left'].wrist is None and g.arms['left'].status == protocol.LOST
    assert g.bones['right'].upper == 0.3 and g.timing_ms['total'] == 5.0
    np.testing.assert_allclose(g.waist, f.waist, atol=1e-6)


def test_frame_is_small():
    assert len(make_frame().to_bytes()) < 700     # no images cross the network


def test_link_latest_frame_wins_and_events_are_reliable():
    port = random.randint(41000, 60000)
    cfg = LinkCfg(host='127.0.0.1', stream_port=port, control_port=port + 1, feedback_port=port + 2)
    robot = RobotLink(cfg, bind='127.0.0.1')
    op = OperatorLink(cfg)
    for _ in range(100):                              # PUB/SUB needs a moment to connect
        op.send_frame(make_frame(seq=-1))
        frame, _ = robot.poll(30)
        if frame is not None:
            break
    assert frame is not None, 'link never connected'
    for i in range(20):
        op.send_frame(make_frame(seq=i))
        time.sleep(0.002)                             # a camera does not send 20 frames in 0 ms
    op.send_event(cmd='record_start', task='t')
    op.send_event(cmd='record_stop')
    time.sleep(0.2)
    frame, events = robot.poll(200)
    assert frame is not None and frame.seq == 19      # conflated: only the newest frame is kept
    assert [e['cmd'] for e in events] == ['record_start', 'record_stop']
    robot.send_feedback({'mode': 'TRACKING'}, b'jpegbytes')
    time.sleep(0.2)
    fb = op.poll_feedback()
    assert fb['status']['mode'] == 'TRACKING' and fb['jpeg'] == b'jpegbytes'
    op.close(), robot.close()
