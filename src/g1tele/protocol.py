"""What crosses the network. Small, explicit and independent of any simulator or middleware.

The perception side (where the camera is) sends one `OperatorFrame` per camera frame: the
operator's arms in a gravity-aligned body frame, hand orientations and finger curls. It contains
no images and no robot-specific numbers, so it is about 1 kB and is the same for every simulator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import msgpack
import numpy as np

# Order of the six Inspire hand actuators (and of ArmObs.fingers). Values: 0 = open ... 1 = closed.
FINGERS = ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']

# ArmObs.status
TRACKING = 'TRACKING'        # joints measured this frame
HELD = 'HELD'                # briefly not measured: last value (bone-constrained) is repeated
LOST = 'LOST'                # not measured for a while: the robot should ease back to rest
CALIBRATING = 'CALIBRATING'
IDLE = 'IDLE'

# OperatorFrame.control: what the operator is steering with
REACH = 'reach'              # body pose: the arms follow your arms, the wrist keeps its orientation
HANDS = 'hands'              # close to the camera, only your hands: the wrist rotates and the fingers move, the arm holds
HOLD = 'hold'                # the whole robot stays exactly where it is (a calibration step is going on)


def _arr(a):
    return None if a is None else np.asarray(a, dtype=float)


@dataclass
class ArmObs:
    status: str = LOST
    quality: float = 0.0                       # 0..1, how much the measurement can be trusted
    wrist: Optional[np.ndarray] = None         # m, relative to this arm's shoulder, body frame
    elbow: Optional[np.ndarray] = None         # same; None when the elbow is not visible
    hand_R: Optional[np.ndarray] = None        # 3x3 hand frame (x fingers, z palm normal), body frame
    fingers: Optional[np.ndarray] = None       # 6 curls in FINGERS order
    hand_quality: float = 0.0
    elbow_quality: float = 1.0                 # 0..1: how much the elbow can be trusted (0 when the forearm points at the camera)
    tips: Optional[np.ndarray] = None          # 5x3 wrist -> fingertip vectors in the hand frame, in palm lengths (thumb, index, middle, ring, pinky)

    def to_wire(self):
        return {'s': self.status, 'q': round(float(self.quality), 3),
                'w': None if self.wrist is None else self.wrist.astype(np.float32).tolist(),
                'e': None if self.elbow is None else self.elbow.astype(np.float32).tolist(),
                'R': None if self.hand_R is None else self.hand_R.astype(np.float32).ravel().tolist(),
                'f': None if self.fingers is None else self.fingers.astype(np.float32).tolist(),
                't': None if self.tips is None else self.tips.astype(np.float32).ravel().tolist(),
                'hq': round(float(self.hand_quality), 3), 'eqy': round(float(self.elbow_quality), 3)}

    @classmethod
    def from_wire(cls, d):
        return cls(status=d['s'], quality=d['q'], wrist=_arr(d['w']), elbow=_arr(d['e']),
                   hand_R=None if d['R'] is None else np.array(d['R'], dtype=float).reshape(3, 3),
                   fingers=_arr(d['f']), tips=None if d.get('t') is None else _arr(d['t']).reshape(5, 3),
                   hand_quality=d.get('hq', 0.0),
                   elbow_quality=d.get('eqy', 1.0))


@dataclass
class BoneLengths:
    """The operator's calibrated bone lengths, in metres."""
    upper: float = 0.0   # shoulder -> elbow
    fore: float = 0.0    # elbow -> wrist

    def valid(self):
        return self.upper > 0.0 and self.fore > 0.0


@dataclass
class OperatorFrame:
    seq: int = 0
    t_capture: float = 0.0                     # wall clock (s) when the camera frame was captured
    t_send: float = 0.0
    mode: str = IDLE                           # IDLE | CALIBRATING | TRACKING
    prompt: str = ''                           # what the operator should do (calibration)
    arms: Dict[str, ArmObs] = field(default_factory=dict)
    bones: Dict[str, BoneLengths] = field(default_factory=dict)
    waist: Optional[np.ndarray] = None         # yaw, roll, pitch (rad) relative to the calibration
    timing_ms: Dict[str, float] = field(default_factory=dict)   # perception stage timings
    control: str = REACH                       # REACH | HANDS

    def to_bytes(self) -> bytes:
        self.t_send = time.time()
        return msgpack.packb({
            'v': 1, 'n': self.seq, 'tc': self.t_capture, 'ts': self.t_send, 'm': self.mode,
            'p': self.prompt,
            'a': {side: obs.to_wire() for side, obs in self.arms.items()},
            'b': {side: [b.upper, b.fore] for side, b in self.bones.items()},
            'wa': None if self.waist is None else [float(x) for x in self.waist],
            'tm': self.timing_ms, 'cm': self.control}, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'OperatorFrame':
        d = msgpack.unpackb(data, raw=False)
        if d.get('v') != 1:
            raise ValueError(f'unsupported OperatorFrame version {d.get("v")}')
        return cls(seq=d['n'], t_capture=d['tc'], t_send=d['ts'], mode=d['m'], prompt=d['p'],
                   arms={s: ArmObs.from_wire(o) for s, o in d['a'].items()},
                   bones={s: BoneLengths(*v) for s, v in d['b'].items()},
                   waist=_arr(d['wa']), timing_ms=d.get('tm', {}), control=d.get('cm', REACH))
