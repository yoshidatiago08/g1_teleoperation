"""The floating hand in MuJoCo (needs mujoco): it follows the target, collides, and its fingers obey the servos."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

from dex_hand import paths
from dex_hand import frames
from dex_hand.mujoco_scene import CYL_START, TABLE_TOP, FloatingHandScene
from dex_hand.retarget import robot_hand_rotation

URDF = paths.urdf('right')


def wxyz(rotation):
    x, y, z, w = frames.quaternion(rotation)
    return np.array([w, x, y, z])


def palm_down_forward():
    """Base-link orientation for fingers forward, palm down: world <- H (diag 1,-1,-1) times H <- B."""
    return wxyz(np.diag([1.0, -1.0, -1.0]) @ robot_hand_rotation(URDF, 'right').T)


@pytest.fixture
def scene():
    return FloatingHandScene(URDF, 'right', home_position=(0.15, -0.02, TABLE_TOP + 0.20), home_quat=palm_down_forward())


def move(scene, goal, seconds=0.6):
    start = scene.goal_pos.copy()
    steps = int(seconds / scene.model.opt.timestep)
    for i in range(steps):
        scene.set_target(start + (np.asarray(goal) - start) * (i + 1) / steps, scene.data.mocap_quat[0])
        scene.step()
    scene.step(300)


def test_hand_follows_a_moving_target(scene):
    move(scene, (0.25, 0.08, TABLE_TOP + 0.25))
    assert np.linalg.norm(scene.hand_position() - scene.goal_pos) < 0.01


def test_table_stops_the_hand(scene):
    move(scene, (0.45, -0.02, TABLE_TOP - 0.10))               # over the table, target below its surface
    assert scene.hand_position()[2] > TABLE_TOP - 0.01              # the wrist origin is at the palm surface
    assert scene.data.contact.dist[:scene.data.ncon].min() > -0.01       # touching, not sunk in


def test_fingers_follow_the_servos_and_coupled_joints_follow_them(scene):
    scene.set_fingers({'R_index_proximal_joint': 1.2})
    scene.step(600)
    m, d = scene.model, scene.data
    proximal = d.qpos[m.jnt_qposadr[m.joint('R_index_proximal_joint').id]]
    intermediate = d.qpos[m.jnt_qposadr[m.joint('R_index_intermediate_joint').id]]
    assert proximal == pytest.approx(1.2, abs=0.1)
    assert intermediate == pytest.approx(proximal, abs=0.1)     # <mimic multiplier=1>


def test_the_hand_pushes_the_cylinder(scene):
    scene.set_target((0.10, CYL_START[1], TABLE_TOP + 0.045), scene.data.mocap_quat[0])
    scene.snap_to_target()
    scene.step(300)
    before = scene.data.xpos[scene.cyl_id].copy()
    move(scene, (0.34, CYL_START[1], TABLE_TOP + 0.045), seconds=1.0)
    after = scene.data.xpos[scene.cyl_id]
    assert after[0] > before[0] + 0.03


def test_status_and_reset(scene):
    scene.step(300)                                             # the cylinder starts 2 mm above the table and settles
    assert scene.status()['lifted_cm'] < 0.1
    scene.data.qpos[scene.cyl_qadr:scene.cyl_qadr + 3] = [0.5, 0.3, TABLE_TOP + 0.1]
    scene.reset_scene()
    assert np.allclose(scene.data.xpos[scene.cyl_id][:2], CYL_START)
