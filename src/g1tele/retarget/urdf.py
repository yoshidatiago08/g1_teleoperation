"""Read the kinematics we need out of a G1 URDF (arm chains, waist, Inspire hand joints).

Plain ElementTree: no Pinocchio, no ROS, nothing to install on a teammate's machine.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4, parent link -> joint frame
    axis: np.ndarray              # unit, in the joint frame
    lower: float
    upper: float
    mimic: Optional[Tuple[str, float, float]] = None   # (master joint, multiplier, offset)


class Urdf:
    def __init__(self, path: str):
        self.path = path
        root = ET.parse(path).getroot()
        self.joints: Dict[str, Joint] = {}
        for j in root.findall('joint'):
            origin = np.eye(4)
            o = j.find('origin')
            if o is not None:
                origin[:3, :3] = rpy_matrix(np.fromstring(o.get('rpy', '0 0 0'), sep=' '))
                origin[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
            a = j.find('axis')
            axis = np.fromstring(a.get('xyz', '1 0 0'), sep=' ') if a is not None else np.array([1., 0, 0])
            axis = axis / (np.linalg.norm(axis) or 1.0)
            lim = j.find('limit')
            typ = j.get('type')
            if typ == 'continuous':
                lo, hi = -math.pi, math.pi
            elif lim is not None and 'lower' in lim.attrib:
                lo, hi = float(lim.get('lower')), float(lim.get('upper'))
            else:
                lo = hi = 0.0
            m = j.find('mimic')
            mimic = None if m is None else (m.get('joint'), float(m.get('multiplier', 1.0)),
                                            float(m.get('offset', 0.0)))
            self.joints[j.get('name')] = Joint(j.get('name'), typ, j.find('parent').get('link'),
                                               j.find('child').get('link'), origin, axis, lo, hi, mimic)
        self.by_child = {j.child: j for j in self.joints.values()}

    def chain(self, base: str, tip: str) -> List[Joint]:
        """Joints from `base` link to `tip` link, in order (fixed joints included)."""
        out, link = [], tip
        while link != base:
            j = self.by_child.get(link)
            if j is None:
                raise RuntimeError(f'{self.path}: no chain from {base} to {tip} (stopped at {link})')
            out.append(j)
            link = j.parent
        return out[::-1]

    def movable(self, joints: List[Joint]) -> List[Joint]:
        return [j for j in joints if j.type in ('revolute', 'continuous', 'prismatic')]


def arm_joint_names(side: str) -> List[str]:
    return [f'{side}_{n}_joint' for n in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                                         'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]


WAIST_JOINTS = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']

# Inspire hand: the joint each of the six actuators drives (the rest follow through <mimic>)
HAND_PREFIX = {'right': 'R_', 'left': 'L_'}
HAND_ACTUATOR_JOINT = ['pinky_proximal_joint', 'ring_proximal_joint', 'middle_proximal_joint',
                       'index_proximal_joint', 'thumb_proximal_pitch_joint', 'thumb_proximal_yaw_joint']


def hand_joint_names(side: str) -> List[str]:
    return [HAND_PREFIX[side] + n for n in HAND_ACTUATOR_JOINT]
