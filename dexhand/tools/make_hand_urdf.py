"""Cut the Inspire hands out of the G1 URDF into standalone hand URDFs, with fingertip frames.

    python3 dexhand/tools/make_hand_urdf.py

Writes ws/src/dex_hand/urdf/inspire_hand_{right,left}.urdf and copies their meshes. Run it again only
if the G1 URDF changes; the outputs are committed. The fingertip offsets (the extreme point of each
last link's mesh) are the ones the main pipeline already uses (src/g1tele/retarget/hand_ik.py).
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets' / 'g1' / 'g1_29dof_inspire_dfq.urdf'
OUT = ROOT / 'dexhand' / 'ws' / 'src' / 'dex_hand' / 'urdf'

LAST_LINK = {'thumb': 'thumb_distal', 'index': 'index_intermediate', 'middle': 'middle_intermediate',
             'ring': 'ring_intermediate', 'pinky': 'pinky_intermediate'}
TIP_OFFSET = {
    'R': {'thumb': (0.0223, 0.0104, -0.0060), 'index': (0.0025, 0.0452, -0.0050),
          'middle': (0.0024, 0.0487, -0.0050), 'ring': (0.0025, 0.0453, -0.0050),
          'pinky': (0.0024, 0.0375, -0.0050)},
    'L': {'thumb': (0.0223, -0.0104, -0.0060), 'index': (0.0027, -0.0453, -0.0050),
          'middle': (0.0024, -0.0487, -0.0050), 'ring': (0.0027, -0.0453, -0.0050),
          'pinky': (0.0031, -0.0375, -0.0050)},
}


def extract(tree, prefix):
    root = tree.getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    joints = root.findall('joint')
    base = f'{prefix}_hand_base_link'
    keep, frontier = {base}, [base]
    kept_joints = []
    while frontier:
        parent = frontier.pop()
        for j in joints:
            if j.find('parent').get('link') == parent:
                child = j.find('child').get('link')
                kept_joints.append(j)
                keep.add(child)
                frontier.append(child)
    new = ET.Element('robot', name=f'inspire_hand_{"right" if prefix == "R" else "left"}')
    used = set()
    for name in sorted(keep):
        for m in links[name].iter('material'):
            if m.get('name'):
                used.add(m.get('name'))
        new.append(links[name])
    for m in root.findall('material'):
        if m.get('name') in used and len(m):
            new.insert(0, m)
    for j in kept_joints:
        new.append(j)
    for finger, offset in TIP_OFFSET[prefix].items():
        tip = f'{prefix}_{finger}_tip'
        ET.SubElement(new, 'link', name=tip)
        joint = ET.SubElement(new, 'joint', name=f'{tip}_joint', type='fixed')
        ET.SubElement(joint, 'origin', xyz=' '.join(f'{v:.4f}' for v in offset), rpy='0 0 0')
        ET.SubElement(joint, 'parent', link=f'{prefix}_{LAST_LINK[finger]}')
        ET.SubElement(joint, 'child', link=tip)
    meshes = set()
    for mesh in new.iter('mesh'):
        fname = mesh.get('filename').split('/')[-1]
        meshes.add(fname)
        mesh.set('filename', f'package://dex_hand/urdf/meshes/{fname}')
    ET.indent(new)
    return new, meshes


def main():
    tree = ET.parse(SRC)
    (OUT / 'meshes').mkdir(parents=True, exist_ok=True)
    for prefix, side in (('R', 'right'), ('L', 'left')):
        robot, meshes = extract(ET.parse(SRC), prefix)
        ET.ElementTree(robot).write(OUT / f'inspire_hand_{side}.urdf', xml_declaration=True, encoding='utf-8')
        for m in meshes:
            shutil.copy(SRC.parent / 'meshes' / m, OUT / 'meshes' / m)
        print(side, len(meshes), 'meshes')


if __name__ == '__main__':
    main()
