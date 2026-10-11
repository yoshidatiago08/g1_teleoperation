import json
import os

import numpy as np

from g1tele import protocol
from g1tele.record import EpisodeRecorder, read_operator_stream
from g1tele.retarget.retargeter import RobotTargets
from g1tele.retarget.urdf import arm_joint_names


def test_episode_has_the_unitree_schema(tmp_path):
    rec = EpisodeRecorder(str(tmp_path), task='pick', goal='pick the block')
    rec.start()
    targets = RobotTargets(arm_q={n: 0.1 for s in ('left', 'right') for n in arm_joint_names(s)},
                           waist_q={'waist_yaw_joint': 0.0, 'waist_roll_joint': 0.0, 'waist_pitch_joint': 0.0},
                           hand={'left': np.zeros(6), 'right': np.ones(6)},
                           status={'left': protocol.TRACKING, 'right': protocol.HELD},
                           ik_error={'left': 0.001, 'right': 0.002}, seq=3)
    for _ in range(3):
        rec.add(b'raw-frame', targets, {n: 0.09 for n in targets.arm_q}, image=np.zeros((8, 8, 3), np.uint8))
    path = rec.stop()
    data = json.load(open(os.path.join(path, 'data.json')))
    assert set(data) == {'info', 'text', 'data'} and len(data['data']) == 3
    item = data['data'][0]
    assert set(item) >= {'idx', 'colors', 'depths', 'states', 'actions', 'sim_state'}
    assert set(item['actions']) == {'left_arm', 'right_arm', 'left_ee', 'right_ee', 'body'}
    assert len(item['actions']['right_arm']['qpos']) == 7 and len(item['actions']['right_ee']['qpos']) == 6
    assert item['actions']['right_ee']['qpos'] == [0.0] * 6      # closed hand = 0 in the DDS convention
    assert item['colors']['color_0'].startswith('colors/')
    quality = json.load(open(os.path.join(path, 'quality.json')))
    assert quality[0]['status']['right'] == protocol.HELD
    assert [b for b in read_operator_stream(os.path.join(path, 'operator.msgpack'))] == [b'raw-frame'] * 3
    # episodes are numbered, not overwritten
    rec2 = EpisodeRecorder(str(tmp_path), task='pick')
    rec2.start()
    rec2.add(None, targets, {})
    assert rec2.stop().endswith('episode_0001')
