import numpy as np
import pytest

from g1tele import protocol
from g1tele.config import load_config
from g1tele.retarget.retargeter import RobotTargets
from g1tele.retarget.urdf import arm_joint_names, hand_joint_names
from g1tele.robot.base import HAND_JOINT_RANGE, curl_to_dds, curl_to_joint

mujoco = pytest.importorskip('mujoco')


def settle(backend, seconds=1.0):
    for _ in range(int(seconds * backend.rate_hz)):
        backend.step()


def test_mujoco_arms_reach_their_targets_and_mimic_joints_follow():
    from g1tele.robot.mujoco_backend import MujocoBackend
    b = MujocoBackend(load_config())
    targets = RobotTargets(
        arm_q={n: v for n, v in zip(arm_joint_names('right'), [-0.8, -0.3, 0.2, 1.1, 0.4, -0.2, 0.3])},
        hand={'right': np.array([1, 0.5, 0, 1, 0.3, 0.8]), 'left': np.zeros(6)})
    b.apply(targets)
    settle(b)
    obs = b.observe()
    for n, v in targets.arm_q.items():
        assert abs(obs[n] - v) < 5e-3, n
    want = curl_to_joint(targets.hand['right'])
    for n, v in zip(hand_joint_names('right'), want):
        assert abs(obs[n] - v) < 0.03, n
    # mimic: the intermediate index joint follows the proximal one 1:1
    assert abs(obs['R_index_intermediate_joint'] - obs['R_index_proximal_joint']) < 0.03
    b.close()


def test_mujoco_servo_tracks_a_moving_target_with_small_lag_and_no_overshoot():
    from g1tele.robot.mujoco_backend import MujocoBackend
    b = MujocoBackend(load_config())
    # a step: no overshoot
    b.apply(RobotTargets(arm_q={'right_shoulder_pitch_joint': -1.0, 'right_elbow_joint': 1.0}))
    peak = {'right_shoulder_pitch_joint': 0.0, 'right_elbow_joint': 0.0}
    for _ in range(300):
        b.step()
        obs = b.observe()
        peak['right_shoulder_pitch_joint'] = max(peak['right_shoulder_pitch_joint'], -obs['right_shoulder_pitch_joint'])
        peak['right_elbow_joint'] = max(peak['right_elbow_joint'], obs['right_elbow_joint'])
    assert all(v < 1.25 for v in peak.values()), peak     # a 1 rad jump of two joints at once
    # a ramp (a human arm moving at 1 rad/s): the joint lags the target by under 50 ms
    lags = []
    for k in range(250):
        t = k / b.rate_hz
        b.apply(RobotTargets(arm_q={'right_elbow_joint': 0.3 * np.sin(2 * np.pi * 0.5 * t)}))
        b.step()
        if k > 100:
            q = b.observe()['right_elbow_joint']
            lags.append(0.3 * np.sin(2 * np.pi * 0.5 * t) - q)
    peak_speed = 0.3 * 2 * np.pi * 0.5
    assert max(abs(l) for l in lags) / peak_speed < 0.05
    b.close()


def test_hand_conventions():
    assert HAND_JOINT_RANGE.shape == (6, 2)
    np.testing.assert_allclose(curl_to_joint(np.zeros(6)), HAND_JOINT_RANGE[:, 0])
    np.testing.assert_allclose(curl_to_joint(np.ones(6)), HAND_JOINT_RANGE[:, 1])
    np.testing.assert_allclose(curl_to_dds(np.array([0, 1, 0.25, 0.5, 1, 0])), [1, 0, 0.75, 0.5, 0, 1])


# ---------------------------------------------------------------- DDS backend with a stub SDK
class StubPub:
    def __init__(self, topic, typ):
        self.topic, self.written = topic, []
    def Init(self, *a):
        pass
    def Write(self, msg, *a):
        self.written.append(msg)


class StubSub:
    def __init__(self, topic, typ):
        self.topic = topic
    def Init(self, *a):
        pass
    def Read(self, *a):
        if self.topic == 'rt/lowstate':
            m = type('S', (), {})()
            m.motor_state = [type('M', (), {'q': 0.1 * i})() for i in range(35)]
            m.mode_machine = 5
            return m
        return None


class StubCmd:
    def __init__(self):
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.mode = 0


class StubLowCmd:
    def __init__(self):
        self.mode_pr, self.mode_machine, self.crc = 0, 0, 0
        self.motor_cmd = [StubCmd() for _ in range(35)]


class StubCRC:
    def Crc(self, msg):
        return 1234


def stub_sdk(calls):
    return dict(ChannelFactoryInitialize=lambda *a: calls.append(a), ChannelPublisher=StubPub,
                ChannelSubscriber=StubSub, MotorCmds_=lambda: type('H', (), {'cmds': []})(),
                MotorStates_=object, LowCmd_=object, LowState_=object, CRC=StubCRC,
                motor_cmd=StubCmd, low_cmd=StubLowCmd)


def fake_clock(rate_hz):
    t = [0.0]
    def clock():
        t[0] += 1.0 / rate_hz
        return t[0]
    return clock


def test_dds_backend_maps_joints_gains_and_hands():
    from g1tele.robot.unitree_dds import JOINT_INDEX, UnitreeDDSBackend
    calls = []
    cfg = load_config(overrides={'robot': {'backend': 'dds', 'dds_domain': 1, 'dds_interface': 'lo'}})
    b = UnitreeDDSBackend(cfg, sdk=stub_sdk(calls), wait_state_s=2.0)
    b._clock = fake_clock(b.rate_hz)
    assert calls == [(1, 'lo')]
    assert JOINT_INDEX['left_shoulder_pitch_joint'] == 15 and JOINT_INDEX['right_wrist_yaw_joint'] == 28
    assert JOINT_INDEX['waist_yaw_joint'] == 12
    assert b.msg.mode_machine == 5
    # starts at the robot's current pose, with PD gains set
    assert abs(b.msg.motor_cmd[16].q - 1.6) < 1e-9 and b.msg.motor_cmd[16].kp == 80.0
    assert b.msg.motor_cmd[26].kp == 40.0                # wrist
    b.apply(RobotTargets(arm_q={'right_elbow_joint': 0.7},
                         waist_q={'waist_yaw_joint': 0.2},
                         hand={'right': np.array([1, 1, 1, 1, 1, 1.]), 'left': np.zeros(6)}))
    for _ in range(200):
        b.step()
    cmd = b.cmd_pub.written[-1]
    assert abs(cmd.motor_cmd[25].q - 0.7) < 1e-6 and abs(cmd.motor_cmd[12].q - 0.2) < 1e-6
    assert cmd.crc == 1234
    hand = b.hand_pub.written[-1]
    assert [round(m.q, 3) for m in hand.cmds[:6]] == [0.0] * 6      # right hand closed -> 0
    assert [round(m.q, 3) for m in hand.cmds[6:]] == [1.0] * 6      # left hand open -> 1
    b.close()


def test_dds_real_robot_uses_arm_sdk_and_ramps_speed():
    from g1tele.robot.unitree_dds import UnitreeDDSBackend
    cfg = load_config(overrides={'robot': {'backend': 'dds', 'target': 'real', 'dds_domain': 0}})
    b = UnitreeDDSBackend(cfg, sdk=stub_sdk([]), wait_state_s=2.0)
    b._clock = fake_clock(b.rate_hz)
    b.t_start = b.t_last = b._clock()
    assert b.cmd_pub.topic == 'rt/arm_sdk' and b.msg.motor_cmd[29].q == 1.0
    b.apply(RobotTargets(arm_q={'right_elbow_joint': 1.5}))
    b.step()
    assert abs(b.cmd_pub.written[-1].motor_cmd[25].q - 2.5) < 0.2       # first steps move very little
    b.close()
