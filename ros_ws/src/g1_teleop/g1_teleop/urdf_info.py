"""Small helpers that read facts about the arm out of the G1 URDF."""
import xml.etree.ElementTree as ET


def right_arm_geometry(urdf_file, base_link='torso_link', tip_link='right_wrist_yaw_link'):
    """Return (shoulder_xyz, reach) for the right arm.

    shoulder_xyz: position of the right shoulder joint relative to base_link.
    reach: summed link lengths from the shoulder to the wrist, used to scale the human arm.
    """
    root = ET.parse(urdf_file).getroot()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    shoulder = next((j for j in root.findall('joint')
                     if j.get('name') == 'right_shoulder_pitch_joint'), None)
    if shoulder is None:
        raise RuntimeError(f'{urdf_file} has no right_shoulder_pitch_joint')

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
