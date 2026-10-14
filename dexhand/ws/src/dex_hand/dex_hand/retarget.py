"""Fingers: wrist -> fingertip vectors -> Inspire joint angles, through dex-retargeting.

dex-retargeting (https://github.com/dexsuite/dex-retargeting) does the optimisation: it finds the robot
joint angles whose wrist -> fingertip vectors best match the human's (vector retargeting), with joint
limits, the URDF's coupled (mimic) joints and a low-pass filter.

The one change from its examples: they rotate the human landmarks into a MANO convention that the
robot URDF's base link has to share. Our URDF is the G1's own hand, so instead the human vectors are
expressed in the robot hand's base link, using the same hand frame definition (frames.hand_rotation)
on the URDF's knuckle positions. Same result, no assumption about the URDF's axes.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dex_hand.frames import hand_rotation


def robot_hand_rotation(urdf_path, side):
    """R_BH: the hand frame H's axes (columns) in the URDF's hand base link B, from the knuckles."""
    prefix = 'R' if side == 'right' else 'L'
    knuckle = {}
    for joint in ET.parse(urdf_path).getroot().findall('joint'):
        for finger in ('index', 'middle', 'pinky'):
            if joint.get('name') == f'{prefix}_{finger}_proximal_joint':
                knuckle[finger] = np.array([float(v) for v in joint.find('origin').get('xyz').split()])
    points = np.zeros((21, 3))
    points[5], points[9], points[17] = knuckle['index'], knuckle['middle'], knuckle['pinky']
    return hand_rotation(points, side)


class FingerRetargeter:
    def __init__(self, urdf_dir, config_path, side, scaling_factor=None):
        from dex_retargeting.retargeting_config import RetargetingConfig
        RetargetingConfig.set_default_urdf_dir(str(urdf_dir))
        config = RetargetingConfig.load_from_file(str(config_path))
        if scaling_factor is not None:
            config.scaling_factor = float(scaling_factor)
        self.retargeting = config.build()
        self.joint_names = list(self.retargeting.joint_names)
        self.r_bh = robot_hand_rotation(Path(urdf_dir) / Path(config.urdf_path).name, side)
        indices = np.asarray(self.retargeting.optimizer.target_link_human_indices)
        self.origin_idx, self.task_idx = indices[0], indices[1]

    def __call__(self, tips_in_hand_frame):
        """`tips_in_hand_frame`: 5x3 wrist -> thumb, index, middle, ring, pinky tip (metres, hand frame H).
        Returns {joint name: angle (rad)} for every joint of the hand (the coupled ones included)."""
        ref = np.asarray(tips_in_hand_frame, dtype=float) @ self.r_bh.T       # H -> hand base link B
        qpos = self.retargeting.retarget(ref)
        return dict(zip(self.joint_names, (float(q) for q in qpos)))
