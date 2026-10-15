"""Generate the CuRobo robot configuration for the right arm of our G1 (Inspire hands, no legs).

    python dexhand/tools/make_curobo_config.py        # needs curobo installed (the VM image)

Writes assets/curobo/g1_right_arm.yml. The IK moves only the 7 right-arm joints; the 3 waist joints are locked (the left arm and the fingers are
not on the kinematic chain). Collision spheres for the pelvis, waist, torso, head and the right arm come from CuRobo's own G1 configuration (the link names
are the same); the Inspire hand is not in it, so spheres are fitted to its meshes here (k-means on the mesh vertices, open hand). The tool
frame is R_hand_base_link, the frame dex-retargeting and the simulation use for the hand. Run it again only if the URDF changes.
"""
from pathlib import Path

import numpy as np
import trimesh
import yaml
import yourdfpy
from scipy.cluster.vq import kmeans2

from curobo._src.util_file import get_robot_configs_path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'assets' / 'g1' / 'g1_upper_inspire.urdf'
OUT = ROOT / 'assets' / 'curobo' / 'g1_right_arm.yml'
G1_REFERENCE = Path(get_robot_configs_path()) / 'unitree_g1_29dof_retarget.yml'

ARM = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
       'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
ARM_HOME = [-0.2094, -0.3316, 0.0, 0.5236, 0.0, 0.0, 0.0]          # the same relaxed pose as CuRobo's G1 default
BODY_LINKS = ['pelvis', 'pelvis_contour_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link',
              'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
              'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
HAND = 'R_hand_base_link'
HAND_SPHERES = 14


def hand_vertices():
    """The open right hand's mesh vertices (all hand links) in the hand base link frame."""
    def handler(fname):
        return str(ROOT / 'assets' / 'g1' / 'meshes' / fname.split('/')[-1])
    robot = yourdfpy.URDF.load(str(URDF), load_meshes=True, build_scene_graph=True, filename_handler=handler,
                               mesh_dir=str(ROOT / 'assets' / 'g1' / 'meshes'))
    points = []
    for link in robot.link_map:
        if not link.startswith('R_') or 'tip' in link:
            continue
        geoms = [g for g in (robot.link_map[link].collisions or []) if g.geometry.mesh is not None]
        for g in geoms:
            mesh = trimesh.load(handler(g.geometry.mesh.filename), force='mesh')
            if g.origin is not None:
                mesh.apply_transform(g.origin)
            mesh.apply_transform(robot.get_transform(frame_to=link, frame_from=HAND))
            points.append(mesh.vertices)
    return np.vstack(points)


def fit_spheres(points, k=HAND_SPHERES):
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    centers, labels = kmeans2(sample, k, minit='++', seed=0)
    spheres = []
    for i in range(k):
        cluster = sample[labels == i]
        if len(cluster) == 0:
            continue
        d = np.linalg.norm(cluster - centers[i], axis=1)
        spheres.append({'center': [round(float(v), 5) for v in centers[i]], 'radius': round(float(np.percentile(d, 95)) + 0.003, 5)})
    return spheres


def main():
    ref = yaml.safe_load(open(G1_REFERENCE))['kinematics']
    robot = yourdfpy.URDF.load(str(URDF), load_meshes=False, build_scene_graph=False)
    # Only the joints between the pelvis and the hand are part of CuRobo's kinematic chain (waist 3 + right arm 7); the left arm and the
    # fingers are not on it, so they are neither locked nor listed.
    parent = {j.child: j for j in robot.robot.joints}
    chain, link = [], HAND
    while link in parent:
        joint = parent[link]
        if joint.type in ('revolute', 'continuous', 'prismatic'):
            chain.append(joint.name)
        link = joint.parent
    names = chain[::-1]
    default = {n: 0.0 for n in names}
    default.update(dict(zip(ARM, ARM_HOME)))

    spheres = {link: ref['collision_spheres'][link] for link in BODY_LINKS if link in ref['collision_spheres']}
    spheres[HAND] = fit_spheres(hand_vertices())
    links = list(spheres)
    ignore = {k: [x for x in v if x in links] for k, v in ref['self_collision_ignore'].items() if k in links}
    ignore.setdefault('right_wrist_yaw_link', []).append(HAND)
    for link in ('right_wrist_roll_link', 'right_wrist_pitch_link'):
        ignore.setdefault(link, []).append(HAND)
    ignore[HAND] = ['right_wrist_yaw_link', 'right_wrist_pitch_link', 'right_wrist_roll_link']
    # CuRobo's reference ignores torso vs the LEFT shoulder-roll link (adjacent on the robot) but not the right one, which looks like an
    # omission: their spheres overlap at ordinary reaching poses and the arm could not move forward.
    ignore.setdefault('torso_link', []).append('right_shoulder_roll_link')

    config = {'robot_cfg': {'kinematics': {
        'format_version': 2.0,
        'base_link': 'pelvis',
        'tool_frames': [HAND],
        'collision_link_names': links,
        'collision_spheres': spheres,
        'collision_sphere_buffer': 0.0,
        'self_collision_ignore': ignore,
        'self_collision_buffer': {link: 0.0 for link in links},
        'mesh_link_names': links,
        'load_meshes': False,
        'lock_joints': {n: float(default[n]) for n in names if n not in ARM},
        'cspace': {
            'joint_names': names,
            'default_joint_position': [float(default[n]) for n in names],
            'max_acceleration': 15.0,
            'max_jerk': 500.0,
            'cspace_distance_weight': [1.0] * len(names),
            'null_space_weight': [1.0] * len(names),
        },
    }}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('# Generated by tools/make_curobo_config.py. Right arm only (7 joints); the waist locked.\n')
        yaml.safe_dump(config, f, sort_keys=False)
    print(f'{len(names)} joints, {len(names) - len(ARM)} locked, {len(links)} collision links, {len(spheres[HAND])} hand spheres -> {OUT}')


if __name__ == '__main__':
    main()
