"""The blue cube and the camera views of the simulator scene (needs mujoco)."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

from dex_hand import paths
from dex_hand.mujoco_scene import CUBE_SIDE, CUBE_START, CYL_START, TABLE_TOP, FloatingHandScene


@pytest.fixture(scope='module')
def scene():
    return FloatingHandScene(paths.urdf('right', 'dex5'), 'right')


def test_the_cube_sits_on_the_table_apart_from_the_cylinder(scene):
    scene.step(500)
    p = scene.data.xpos[scene.cube_id]
    assert p[2] == pytest.approx(TABLE_TOP + CUBE_SIDE / 2, abs=0.004)
    assert np.allclose(p[:2], CUBE_START, atol=0.003)
    gap = np.linalg.norm(np.array(CUBE_START) - np.array(CYL_START)) - CUBE_SIDE / 2 - 0.032
    assert gap > 0.04                                                         # room to grasp either one


def test_the_state_reports_the_cube_and_reset_puts_both_objects_back(scene):
    state = scene.scene_state()
    assert state['cube_pos'] and state['cube_lifted_cm'] == pytest.approx(0.0, abs=0.5)
    adr = scene.cube_qadr
    scene.data.qpos[adr:adr + 3] = [0.5, 0.3, 1.2]
    scene.data.qpos[scene.cyl_qadr:scene.cyl_qadr + 3] = [0.5, -0.3, 1.2]
    scene.reset_scene()
    assert np.allclose(scene.data.xpos[scene.cube_id][:2], CUBE_START) and np.allclose(scene.data.xpos[scene.cyl_id][:2], CYL_START)


def test_each_camera_view_renders_and_the_close_ones_follow_the_nearest_object(scene):
    images = {name: scene.render(160, 120, camera=name) for name in ('front', 'close', 'top')}
    assert all(img.shape == (120, 160, 3) and img.std() > 5 for img in images.values())
    assert not np.array_equal(images['front'], images['close'])
    far = np.linalg.norm(scene.focus_point() - scene.data.xpos[scene.cyl_id])
    scene.data.qpos[scene.cube_qadr:scene.cube_qadr + 2] = [0.45, 0.2]       # move the cube away: the focus stays between hand and nearest object
    scene.mj.mj_forward(scene.model, scene.data)
    for _ in range(60):
        focus = scene.focus_point()
    nearest = min((scene.data.xpos[scene.cyl_id], scene.data.xpos[scene.cube_id]), key=lambda p: np.linalg.norm(p - scene.data.xpos[scene.base_id]))
    assert np.linalg.norm(focus - nearest) < 0.5 * np.linalg.norm(focus - scene.data.xpos[scene.base_id]) + 0.05


def test_the_arm_scene_has_the_cube_too():
    from dex_hand.mujoco_arm_scene import ArmScene
    arm = ArmScene(urdf_path=paths.g1_urdf('inspire'))
    arm.step(300)
    assert 'cube_pos' in arm.scene_state() and 'cube' not in ''.join(arm.robot_joints())
    assert arm.render(160, 120, camera='close').shape == (120, 160, 3)


def test_the_vm_sends_the_overview_and_the_close_up_to_the_notebook():
    import threading
    import time
    pytest.importorskip('zmq')
    pytest.importorskip('dex_retargeting')
    from dex_hand import link
    from dex_hand.vm import VmApp
    port = 5864
    app = VmApp('right', 'synthetic', port=port, render=True, render_size=(160, 120), mode='floating', views=('front', 'close', 'top'))
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,), daemon=True)
    thread.start()
    _, state_in, events_out = link.notebook_side('127.0.0.1', port)
    try:
        message, end = None, time.time() + 15.0
        while time.time() < end:
            message = state_in.recv(100)
            if message and message.get('sim') and message.get('views', {}).get('top'):
                break
        assert message and message['sim'] and set(message['views']) == {'close', 'top'}
        assert 'cube_pos' in message['state']['scene']
    finally:
        stop.set()
        thread.join(3)
        state_in.close()
        events_out.close()
