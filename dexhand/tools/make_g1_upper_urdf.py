"""Cut the G1 without its legs (the lower body is static) out of the main project's URDF, with the meshes it uses.

    python3 dexhand/tools/make_g1_upper_urdf.py

Writes assets/g1/g1_upper_inspire.urdf and assets/g1/meshes/. The joint and link names are unchanged, so MuJoCo, CuRobo and
RViz all use the same names as the main project. Run it again only if the main URDF changes; the outputs are committed.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets' / 'g1' / 'g1_29dof_inspire_dfq.urdf'
OUT = ROOT / 'dexhand' / 'assets' / 'g1'
LEG_MARKERS = ('hip', 'knee', 'ankle')


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    drop = {l.get('name') for l in root.findall('link') if any(m in l.get('name') for m in LEG_MARKERS)}
    for link in list(root.findall('link')):
        if link.get('name') in drop:
            root.remove(link)
    for joint in list(root.findall('joint')):
        if joint.find('child').get('link') in drop or joint.find('parent').get('link') in drop:
            root.remove(joint)
    meshes = set()
    for mesh in root.iter('mesh'):
        name = mesh.get('filename').split('/')[-1]
        meshes.add(name)
        mesh.set('filename', f'package://dex_hand/g1/meshes/{name}')
    root.set('name', 'g1_upper_inspire')
    (OUT / 'meshes').mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(OUT / 'g1_upper_inspire.urdf', xml_declaration=True, encoding='utf-8')
    for name in meshes:
        shutil.copy(SRC.parent / 'meshes' / name, OUT / 'meshes' / name)
    size = sum(f.stat().st_size for f in (OUT / 'meshes').iterdir()) / 1e6
    print(f'{len(drop)} leg links dropped, {len(meshes)} meshes copied ({size:.1f} MB)')


if __name__ == '__main__':
    main()
