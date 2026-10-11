"""Unitree DDS backend: drives unitree_sim_isaaclab (Isaac Sim) and, with the same code, a real G1.

It speaks the robot's own protocol, which is the whole point: whatever works here works on the
robot, and every Unitree simulator (Isaac Lab, unitree_mujoco) already listens on it.

    arms + waist   rt/lowcmd (or rt/arm_sdk), hg LowCmd_ with CRC, 250 Hz, joint order of the G1 29-DoF
    Inspire hands  rt/inspire/cmd, MotorCmds_, 12 values in [0, 1] (1 = open), right hand first
    state          rt/lowstate, rt/inspire/state

Needs `unitree_sdk2py` (+ cyclonedds). The conventions below were taken from unitree's
xr_teleoperate and unitree_sim_isaaclab sources; the logic is unit-tested with a stub SDK, but run
it against the simulator before trusting it with anything else.

Safety: the real robot listens on DDS domain 0, the simulators on domain 1. `robot.target` and the
domain are separate settings on purpose, and `real` needs --confirm-real on the command line.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import numpy as np

from g1tele.config import Config
from g1tele.retarget.retargeter import RobotTargets
from g1tele.retarget.urdf import hand_joint_names
from g1tele.robot.base import RobotBackend, curl_to_dds, curl_to_joint

log = logging.getLogger('g1tele.dds')

TOPIC_LOWCMD, TOPIC_ARM_SDK, TOPIC_LOWSTATE = 'rt/lowcmd', 'rt/arm_sdk', 'rt/lowstate'
TOPIC_INSPIRE_CMD, TOPIC_INSPIRE_STATE = 'rt/inspire/cmd', 'rt/inspire/state'

# G1 29-DoF motor order in LowCmd_/LowState_ (unitree_hg IDL)
G1_JOINTS = (
    ['left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
     'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
     'waist_yaw', 'waist_roll', 'waist_pitch']
    + [f'left_{n}' for n in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll',
                              'wrist_pitch', 'wrist_yaw')]
    + [f'right_{n}' for n in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll',
                               'wrist_pitch', 'wrist_yaw')])
JOINT_INDEX: Dict[str, int] = {f'{n}_joint': i for i, n in enumerate(G1_JOINTS)}
NUM_MOTORS = 35                 # the IDL message has 35 slots, the G1 29-DoF uses the first 29
ARM_SDK_ENABLE_SLOT = 29        # rt/arm_sdk: motor 29's q is the weight of this controller (1 = on)
INSPIRE_RIGHT_BASE, INSPIRE_LEFT_BASE = 0, 6     # slots in the 12-motor Inspire messages

# PD gains (kp, kd) as in xr_teleoperate
GAINS = {'arm': (80.0, 3.0), 'wrist': (40.0, 1.5), 'strong': (300.0, 3.0), 'weak': (80.0, 3.0)}


def gains_for(name: str):
    if 'wrist' in name:
        return GAINS['wrist']
    if 'shoulder' in name or 'elbow' in name:
        return GAINS['arm']
    if 'ankle' in name:
        return GAINS['weak']
    return GAINS['strong']


def _load_sdk():
    """Import the SDK pieces, with a helpful error."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as e:
        raise ImportError(
            f'unitree_sdk2py is not installed ({e}). Install it from '
            'https://github.com/unitreerobotics/unitree_sdk2_python (pip install cyclonedds, then '
            'pip install -e that repo) or use robot.backend=mujoco.') from e
    return dict(ChannelFactoryInitialize=ChannelFactoryInitialize, ChannelPublisher=ChannelPublisher,
                ChannelSubscriber=ChannelSubscriber, MotorCmds_=MotorCmds_, MotorStates_=MotorStates_,
                LowCmd_=LowCmd_, LowState_=LowState_, CRC=CRC,
                motor_cmd=unitree_go_msg_dds__MotorCmd_, low_cmd=unitree_hg_msg_dds__LowCmd_)


class UnitreeDDSBackend(RobotBackend):
    def __init__(self, cfg: Config, sdk: Optional[dict] = None, wait_state_s: float = 15.0):
        self.cfg = cfg
        self.sdk = sdk or _load_sdk()
        self.rate_hz = cfg.robot.rate_hz
        self.real = cfg.robot.target == 'real'
        self.arm_sdk = self.real           # on the real robot only the arms/waist are ours: use arm_sdk
        self.sides = list(cfg.perception.arms)
        if cfg.robot.dds_interface:
            self.sdk['ChannelFactoryInitialize'](cfg.robot.dds_domain, cfg.robot.dds_interface)
        else:
            self.sdk['ChannelFactoryInitialize'](cfg.robot.dds_domain)
        self.cmd_pub = self.sdk['ChannelPublisher'](TOPIC_ARM_SDK if self.arm_sdk else TOPIC_LOWCMD,
                                                    self.sdk['LowCmd_'])
        self.cmd_pub.Init()
        self.state_sub = self.sdk['ChannelSubscriber'](TOPIC_LOWSTATE, self.sdk['LowState_'])
        self.state_sub.Init()
        self.hand_pub = self.sdk['ChannelPublisher'](TOPIC_INSPIRE_CMD, self.sdk['MotorCmds_'])
        self.hand_pub.Init()
        self.hand_sub = self.sdk['ChannelSubscriber'](TOPIC_INSPIRE_STATE, self.sdk['MotorStates_'])
        self.hand_sub.Init()
        self.crc = self.sdk['CRC']()

        self.lock = threading.Lock()
        self.state_q = np.zeros(NUM_MOTORS)
        self.hand_state = np.full(12, np.nan)
        self.mode_machine = 0
        self.have_state = False
        self._stop = False
        self._reader = threading.Thread(target=self._read_state, daemon=True)
        self._reader.start()
        self._wait_for_state(wait_state_s)

        self.msg = self.sdk['low_cmd']()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.mode_machine
        self.q_cmd = self.state_q.copy()            # what we publish; starts where the robot is
        self.q_goal = self.state_q.copy()
        self.commanded = set()                      # joints our targets move
        for name, idx in JOINT_INDEX.items():
            kp, kd = gains_for(name)
            m = self.msg.motor_cmd[idx]
            m.mode, m.kp, m.kd, m.q, m.dq, m.tau = 1, kp, kd, float(self.q_cmd[idx]), 0.0, 0.0
        if self.arm_sdk:
            self.msg.motor_cmd[ARM_SDK_ENABLE_SLOT].q = 1.0
        self.hand_msg = self.sdk['MotorCmds_']()
        self.hand_msg.cmds = [self.sdk['motor_cmd']() for _ in range(12)]
        for m in self.hand_msg.cmds:
            m.q = 1.0                                # open
        self.hand_goal = np.ones(12)
        self._clock = time.monotonic             # replaceable in tests
        self.t_start = self._clock()
        self.t_last = self.t_start
        log.info('DDS backend up: domain %d, %s, target=%s', cfg.robot.dds_domain,
                 TOPIC_ARM_SDK if self.arm_sdk else TOPIC_LOWCMD, cfg.robot.target)

    # ------------------------------------------------------------------ state
    def _read_state(self):
        while not self._stop:
            msg = self.state_sub.Read()
            if msg is not None:
                with self.lock:
                    for i in range(min(NUM_MOTORS, len(msg.motor_state))):
                        self.state_q[i] = msg.motor_state[i].q
                    self.mode_machine = msg.mode_machine
                    self.have_state = True
            hm = self.hand_sub.Read()
            if hm is not None:
                with self.lock:
                    for i in range(min(12, len(hm.states))):
                        self.hand_state[i] = hm.states[i].q
            time.sleep(0.002)

    def _wait_for_state(self, timeout: float):
        t0 = time.time()
        while not self.have_state:
            if time.time() - t0 > timeout:
                raise TimeoutError(f'no rt/lowstate within {timeout:.0f} s on DDS domain '
                                   f'{self.cfg.robot.dds_domain}. Is the simulator running, and are both '
                                   'sides on the same domain/interface?')
            time.sleep(0.02)

    # ------------------------------------------------------------------ control
    def apply(self, targets: RobotTargets) -> None:
        with self.lock:
            for name, q in {**targets.arm_q, **targets.waist_q}.items():
                idx = JOINT_INDEX.get(name)
                if idx is not None:
                    self.q_goal[idx] = q
                    self.commanded.add(idx)
            for side, curls in targets.hand.items():
                base = INSPIRE_RIGHT_BASE if side == 'right' else INSPIRE_LEFT_BASE
                self.hand_goal[base:base + 6] = curl_to_dds(curls)

    def step(self) -> None:
        now = self._clock()
        dt = min(max(now - self.t_last, 1e-4), 0.05)
        self.t_last = now
        with self.lock:
            goal = self.q_goal.copy()
            hands = self.hand_goal.copy()
        limit = self.cfg.robot.max_joint_speed
        if self.real:
            # soft start on a physical robot: the speed limit ramps up over the first seconds
            limit *= min(1.0, 0.1 + (now - self.t_start) / 4.0)
        for idx in self.commanded:
            step = np.clip(goal[idx] - self.q_cmd[idx], -limit * dt, limit * dt)
            self.q_cmd[idx] += step
            self.msg.motor_cmd[idx].q = float(self.q_cmd[idx])
        self.msg.crc = self.crc.Crc(self.msg)
        self.cmd_pub.Write(self.msg)
        for i, v in enumerate(hands):
            self.hand_msg.cmds[i].q = float(v)
        self.hand_pub.Write(self.hand_msg)

    def observe(self) -> Dict[str, float]:
        with self.lock:
            out = {n: float(self.state_q[i]) for n, i in JOINT_INDEX.items()}
            for side in self.sides:
                base = INSPIRE_RIGHT_BASE if side == 'right' else INSPIRE_LEFT_BASE
                dds = self.hand_state[base:base + 6]
                if not np.any(np.isnan(dds)):
                    # state is normalised like the command (1 = open): back to curls, then radians
                    for n, q in zip(hand_joint_names(side), curl_to_joint(1.0 - dds)):
                        out[n] = float(q)
        return out

    def close(self) -> None:
        self._stop = True
