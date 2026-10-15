"""The notebook's ROS bridge (runs only where ROS 2 is installed: the notebook image)."""
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('msgpack')

import msgpack

from dex_hand.ros_bridge import RosBridge


def through_the_wire(state):
    return msgpack.unpackb(msgpack.packb(state, use_bin_type=True), raw=False)


def test_publish_accepts_a_state_like_the_vm_sends_it(monkeypatch):
    """Regression: an int where ROS wants a float (e.g. a quaternion (1, 0, 0, 0)) aborts the whole process."""
    monkeypatch.setattr(RosBridge, '_start', lambda self, side, rviz: [])
    bridge = RosBridge('right', rviz=False)
    try:
        state = {'joints': {'R_index_proximal_joint': 0.5}, 'goal_pos': [0.2, 0.0, 1.0], 'goal_quat': [1, 0, 0, 0],
                 'ghost': {'pos': [0.3, 0.0, 1.0], 'quat': [1, 0, 0, 0]}, 'skeleton': [[0.0, 0.0, 0.0]] * 21,
                 'scene': {'hand_pos': [0.2, 0, 1], 'hand_quat': [1, 0, 0, 0], 'cyl_pos': [0.38, -0.02, 0.88],
                           'cyl_quat': [1, 0, 0, 0], 'to_target_cm': 18.0, 'lifted_cm': 0.0, 'on_target': False}}
        bridge.publish(through_the_wire(state))
        bridge.publish({})                                     # nothing received yet: must not raise
    finally:
        bridge.close()


def test_arm_mode_publishes_the_whole_robot_and_the_pelvis_frame(monkeypatch):
    monkeypatch.setattr(RosBridge, '_start', lambda self, side, rviz: [])
    bridge = RosBridge('right', rviz=False, mode='arm')
    try:
        state = {'goal_pos': [0.2, 0.0, 1.0], 'goal_quat': [1, 0, 0, 0],
                 'scene': {'hand_pos': [0.2, 0, 1], 'hand_quat': [1, 0, 0, 0], 'cyl_pos': [0.38, -0.02, 0.88], 'cyl_quat': [1, 0, 0, 0],
                           'robot_joints': {'right_elbow_joint': 0, 'R_index_proximal_joint': 0.5}}}
        bridge.publish(through_the_wire(state))
    finally:
        bridge.close()
