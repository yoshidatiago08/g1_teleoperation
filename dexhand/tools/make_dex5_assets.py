"""Build the Unitree Dex5-1 assets: standalone hand URDFs, the G1 with Dex5-1 hands, and the dex-retargeting configs.

    git clone --depth 1 https://github.com/unitreerobotics/unitree_ros /tmp/unitree_ros
    python3 dexhand/tools/make_dex5_assets.py --src /tmp/unitree_ros/robots/dexterous_hand_description/dex5_1

(needs numpy, scipy and trimesh: the VM image has them; PYTHONPATH=dexhand/src for dex_hand.) Run it again only if the source
URDFs change; the outputs are committed. Then run tools/make_curobo_config.py --hand-model dex5 (VM image) for the CuRobo config.

What it does to Unitree's Dex5-URDF-{L,R}:
  * renames links and joints to the Inspire convention the rest of dexhand relies on (R_hand_base_link, R_index_mcp_joint, ...:
    everything of the right hand starts with R_, of the left with L_), and prefixes the meshes with dex5_
  * the fourth joint of each long finger (distal) is passive in the real hand (16 of the 20 joints are active): it follows the
    middle one with DIP_RATIO (the ratio of their joint ranges; an assumption, Unitree does not publish the linkage here), as a <mimic>
  * adds a fingertip frame per finger (the end of the last link's mesh), the frames dex-retargeting matches
  * attaches the hands to the G1's wrist in place of the Inspire ones, turned so that the HAND FRAME (frames.hand_rotation) is where
    the Inspire's was: the same operator pose gives the same hand pose, whatever hand is mounted. The wrist point is unchanged.
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from dex_hand.retarget import robot_hand_rotation

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
FINGERS = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'}
DIP_RATIO = 0.83                      # dip = DIP_RATIO * pip (limits 1.396 / 1.684)
THUMB_JOINTS = {1: 'yaw', 2: 'roll', 3: 'mcp', 4: 'ip'}
FINGER_JOINTS = {1: 'abduction', 2: 'mcp', 3: 'pip', 4: 'dip'}


def vec(text):
    return np.array([float(v) for v in text.split()])


def fmt(v):
    return ' '.join(f'{x:.9g}' for x in v)


def build_hand(src, prefix):
    """The renamed hand of one side as a list of XML elements (links, joints) and the meshes to copy."""
    letter = prefix
    root = ET.parse(next(Path(src).glob(f'Dex5-URDF-{letter}/*.urdf'))).getroot()
    mesh_dir = Path(src) / f'Dex5-URDF-{letter}' / 'meshes'
    link_names = {}
    joint_names = {}
    for link in root.findall('link'):
        n = link.get('name')
        if n.startswith('base_link'):
            link_names[n] = f'{prefix}_hand_base_link'
        elif n.startswith('Link_'):
            f, k = int(n[5]), int(n[6])
            link_names[n] = f'{prefix}_{FINGERS[f]}_{k}'
    for joint in root.findall('joint'):
        n = joint.get('name')                                 # Yaw_11R, Roll_21L, Pitch_23R ...
        f, k = int(n.split('_')[1][0]), int(n.split('_')[1][1])
        role = (THUMB_JOINTS if f == 1 else FINGER_JOINTS)[k]
        joint_names[n] = f'{prefix}_{FINGERS[f]}_{role}_joint'
    elements, meshes = [], {}
    for link in root.findall('link'):
        link.set('name', link_names[link.get('name')])
        for mesh in link.iter('mesh'):
            old = mesh.get('filename').split('/')[-1]
            new = f'dex5_{prefix}_{old}'
            meshes[new] = mesh_dir / old
            mesh.set('filename', f'package://dex_hand/urdf/meshes/{new}')
        elements.append(link)
    for joint in root.findall('joint'):
        old = joint.get('name')
        joint.set('name', joint_names[old])
        joint.find('parent').set('link', link_names[joint.find('parent').get('link')])
        joint.find('child').set('link', link_names[joint.find('child').get('link')])
        if old.startswith('Pitch_') and old[7] == '4' and old[6] != '1':        # distal joint of a long finger: passive
            pip = joint_names[old.replace('_24', '_23').replace('_34', '_33').replace('_44', '_43').replace('_54', '_53')]
            ET.SubElement(joint, 'mimic', joint=pip, multiplier=str(DIP_RATIO), offset='0')
        elements.append(joint)
    # fingertip frames: the end of each last link's mesh
    for f, name in FINGERS.items():
        last = f'{prefix}_{name}_4'
        old_link = next(k for k, v in link_names.items() if v == last)
        mesh = trimesh.load(mesh_dir / f'{old_link}.STL', force='mesh')
        far = np.linalg.norm(mesh.vertices, axis=1)
        tip = mesh.vertices[far > far.max() - 0.002].mean(axis=0)
        tip_link = ET.Element('link', name=f'{prefix}_{name}_tip')
        joint = ET.Element('joint', name=f'{prefix}_{name}_tip_joint', type='fixed')
        ET.SubElement(joint, 'origin', xyz=fmt(tip), rpy='0 0 0')
        ET.SubElement(joint, 'parent', link=last)
        ET.SubElement(joint, 'child', link=tip_link.get('name'))
        elements += [tip_link, joint]
    return elements, meshes


def hand_robot(src, prefix, name):
    robot = ET.Element('robot', name=name)
    elements, meshes = build_hand(src, prefix)
    for e in elements:
        robot.append(e)
    return robot, meshes


def retarget_config(side, prefix):
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    active = [f'{prefix}_thumb_{r}_joint' for r in THUMB_JOINTS.values()]
    for f in fingers[1:]:
        active += [f'{prefix}_{f}_{r}_joint' for r in ('abduction', 'mcp', 'pip')]
    return f"""# dex-retargeting config for the Unitree Dex5-1 hand ({side}), vector retargeting: the wrist -> fingertip vectors of your hand are
# matched by the robot hand's wrist -> fingertip vectors. Generated by tools/make_dex5_assets.py.
# 16 actuated joints (thumb 4, and abduction + 2 flexions for each long finger); the distal joints follow the middle ones (<mimic>).
# Both the human vectors and the URDF are expressed in the robot hand's base link (dex_hand.retarget), so no MANO rotation is needed.
retargeting:
  type: vector
  urdf_path: dex5_hand_{side}.urdf

  target_joint_names: [ {', '.join(repr(n) for n in active)} ]
  target_origin_link_names: [ '{prefix}_hand_base_link', '{prefix}_hand_base_link', '{prefix}_hand_base_link', '{prefix}_hand_base_link', '{prefix}_hand_base_link' ]
  target_task_link_names: [ '{prefix}_thumb_tip', '{prefix}_index_tip', '{prefix}_middle_tip', '{prefix}_ring_tip', '{prefix}_pinky_tip' ]
  # Robot fingers vs yours: 1.0 = same length. This is NOT the hand motion scale. 'auto' (the default) measures it at the calibration.
  scaling_factor: 1.0

  # (origin, task) MediaPipe landmark per vector: wrist -> each fingertip
  target_link_human_indices: [ [ 0, 0, 0, 0, 0 ], [ 4, 8, 12, 16, 20 ] ]

  low_pass_alpha: 0.2
"""


def g1_with_dex5(hands):
    """The G1 upper-body URDF with the Inspire hands replaced by the Dex5-1 ones (hands: {'R': (elements, ...), 'L': ...})."""
    tree = ET.parse(ASSETS / 'g1' / 'g1_upper_inspire.urdf')
    root = tree.getroot()
    r_inspire = {s: robot_hand_rotation(ASSETS / 'urdf' / f'inspire_hand_{side}.urdf', side)
                 for s, side in (('R', 'right'), ('L', 'left'))}
    r_dex5 = {s: robot_hand_rotation(ASSETS / 'urdf' / f'dex5_hand_{side}.urdf', side)
              for s, side in (('R', 'right'), ('L', 'left'))}
    for prefix in ('R', 'L'):
        mount = next(j for j in root.findall('joint') if j.find('child').get('link') == f'{prefix}_hand_base_link')
        origin = mount.find('origin')
        r_old = Rotation.from_euler('xyz', vec(origin.get('rpy'))).as_matrix()
        r_new = r_old @ r_inspire[prefix] @ r_dex5[prefix].T
        origin.set('rpy', fmt(Rotation.from_matrix(r_new).as_euler('xyz')))
        # drop every Inspire hand link and joint of this side except the mount joint, then add the Dex5-1 ones
        for el in list(root):
            if el.tag == 'link' and el.get('name').startswith(f'{prefix}_'):
                root.remove(el)
            elif el.tag == 'joint' and el.get('name').startswith(f'{prefix}_') and el is not mount:
                root.remove(el)
        root.remove(mount)
        root.append(mount)
        elements, _ = hands[prefix]
        for el in elements:
            root.append(el)
    root.set('name', 'g1_upper_dex5')
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='unitree_ros/robots/dexterous_hand_description/dex5_1')
    src = parser.parse_args().src
    (ASSETS / 'urdf' / 'meshes').mkdir(parents=True, exist_ok=True)
    (ASSETS / 'g1' / 'meshes').mkdir(parents=True, exist_ok=True)
    built = {}
    for prefix, side in (('R', 'right'), ('L', 'left')):
        robot, meshes = hand_robot(src, prefix, f'dex5_hand_{side}')
        ET.indent(robot)
        ET.ElementTree(robot).write(ASSETS / 'urdf' / f'dex5_hand_{side}.urdf', xml_declaration=True, encoding='utf-8')
        for new, path in meshes.items():
            shutil.copy(path, ASSETS / 'urdf' / 'meshes' / new)
        (ASSETS / 'config' / f'dex5_hand_{side}.yml').write_text(retarget_config(side, prefix))
        built[prefix] = build_hand(src, prefix)                       # fresh elements for the G1 (the first set belongs to the hand file)
        print(side, len(meshes), 'meshes')
    tree = g1_with_dex5(built)
    root = tree.getroot()
    for mesh in root.iter('mesh'):                                     # the G1's hand meshes live next to the hand ones in g1/meshes
        name = mesh.get('filename').split('/')[-1]
        if name.startswith('dex5_'):
            mesh.set('filename', f'package://dex_hand/g1/meshes/{name}')
            shutil.copy(ASSETS / 'urdf' / 'meshes' / name, ASSETS / 'g1' / 'meshes' / name)
    ET.indent(tree)
    tree.write(ASSETS / 'g1' / 'g1_upper_dex5.urdf', xml_declaration=True, encoding='utf-8')
    print('g1_upper_dex5.urdf written')


if __name__ == '__main__':
    main()
