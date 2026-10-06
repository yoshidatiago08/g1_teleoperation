"""Small helpers that read facts about the arm out of the G1 URDF."""
import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(rpy):
    """Rotation matrix for URDF roll-pitch-yaw angles."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def joint_origin(joint):
    """4x4 transform of a URDF joint's <origin> (parent link -> joint frame)."""
    origin = joint.find('origin')
    t = np.eye(4)
    if origin is not None:
        t[:3, :3] = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
        t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return t


def arm_geometry(urdf_file, side='right', base_link='torso_link'):
    """Return (shoulder_xyz, reach) for one arm ('right' or 'left').

    shoulder_xyz: position of the shoulder joint relative to base_link.
    reach: summed link lengths from the shoulder to the wrist, used to scale the human arm.
    """
    tip_link = f'{side}_wrist_yaw_link'
    root = ET.parse(urdf_file).getroot()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    shoulder = next((j for j in root.findall('joint')
                     if j.get('name') == f'{side}_shoulder_pitch_joint'), None)
    if shoulder is None:
        raise RuntimeError(f'{urdf_file} has no {side}_shoulder_pitch_joint')

    def xyz(joint):
        return [float(v) for v in joint.find('origin').get('xyz', '0 0 0').split()]

    reach, link = 0.0, tip_link
    while link != base_link:
        joint = by_child.get(link)
        if joint is None:
            raise RuntimeError(f'{urdf_file}: no chain from {base_link} to {tip_link}')
        if joint is not shoulder:
            reach += sum(v * v for v in xyz(joint)) ** 0.5
        link = joint.find('parent').get('link')
    if reach <= 0.0:
        raise RuntimeError('Could not derive a nonzero arm reach from the URDF')
    return xyz(shoulder), reach


def rest_tip_position(urdf_file, side='right', base_link='torso_link'):
    """Position of the wrist link, in base_link, with every joint at 0 (the arm hanging at rest)."""
    tip_link = f'{side}_wrist_yaw_link'
    root = ET.parse(urdf_file).getroot()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    transform, link = np.eye(4), tip_link
    while link != base_link:
        joint = by_child.get(link)
        if joint is None:
            raise RuntimeError(f'{urdf_file}: no chain from {base_link} to {tip_link}')
        transform = joint_origin(joint) @ transform
        link = joint.find('parent').get('link')
    return [float(v) for v in transform[:3, 3]]

