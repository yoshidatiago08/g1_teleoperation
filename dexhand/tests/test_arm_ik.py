"""CuRobo right-arm IK on the real G1 model (needs curobo and a CUDA GPU: the VM image)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('curobo')
if not torch.cuda.is_available():
    pytest.skip('no CUDA GPU', allow_module_level=True)

from dex_hand import frames, paths
from dex_hand.arm_ik import ARM_JOINTS, ArmIK
from dex_hand.mujoco_scene import TABLE_TOP
from dex_hand.retarget import robot_hand_rotation

HOME = np.array([0.22, -0.15, 0.98])         # wrist (hand base link) in the world frame: reachable, clear of the cylinder


def wxyz(rotation):
    x, y, z, w = frames.quaternion(rotation)
    return np.array([w, x, y, z])


PALM_DOWN = wxyz(np.diag([1.0, -1.0, -1.0]) @ robot_hand_rotation(paths.urdf('right'), 'right').T)


@pytest.fixture(scope='module')
def ik():
    solver = ArmIK(avoid_table=True)
    assert solver.start(HOME, PALM_DOWN) is not None
    return solver


@pytest.fixture
def at_home(ik):
    ik.start(HOME, PALM_DOWN)
    return ik


def test_start_puts_the_hand_on_the_home_pose(at_home):
    position, quat = at_home.forward(at_home._joints())
    assert np.allclose(position, HOME, atol=0.003)
    assert abs(float(np.dot(quat, PALM_DOWN))) > 0.999                    # same orientation (q and -q are the same rotation)


def test_only_the_right_arm_moves_the_waist_stays_locked(at_home):
    joints = at_home.track(HOME + [0.03, 0.02, 0.0], PALM_DOWN)
    assert all(abs(joints[n]) < 1e-6 for n in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'))
    assert set(ARM_JOINTS) <= set(joints)


def test_tracks_a_moving_target_accurately_and_fast(at_home):
    import time
    errors, times = [], []
    for i in range(90):
        a = 2 * np.pi * i / 90
        target = HOME + np.array([0.04 * np.sin(a), 0.04 * np.sin(a), 0.03 * (np.cos(a) - 1)])
        t = time.perf_counter()
        joints = at_home.track(target, PALM_DOWN)
        times.append(time.perf_counter() - t)
        errors.append(np.linalg.norm(at_home.forward(joints)[0] - target) * 1000)
    assert np.median(errors) < 2.0 and np.percentile(errors, 95) < 8.0
    assert np.median(times[10:]) < 0.015                                    # fast enough for 30 Hz with room to spare


def test_joint_speed_is_limited(at_home):
    previous = np.array([at_home._joints()[n] for n in ARM_JOINTS])
    worst = 0.0
    for i in range(40):
        joints = at_home.track(HOME + [0.0, 0.0, 0.0] + np.array([0.12 * (i > 5), 0.0, 0.0]), PALM_DOWN)   # a 12 cm jump of the target
        current = np.array([joints[n] for n in ARM_JOINTS])
        worst = max(worst, np.abs(current - previous).max() * 30)
        previous = current
    assert worst <= 3.0 + 1e-3                                              # rad/s at 30 Hz


def test_an_unreachable_target_holds_the_arm(at_home):
    before = at_home._joints()
    after = at_home.track(np.array([0.95, -0.15, 0.98]), PALM_DOWN)         # far beyond the arm's reach
    assert all(abs(before[n] - after[n]) < 0.2 for n in ARM_JOINTS)         # it did not fly anywhere
    assert at_home.failures >= 1


def test_an_unreachable_target_gives_the_nearest_reachable_pose_and_the_arm_follows_back_at_once(at_home):
    far = np.array([0.95, -0.15, 0.98])
    for _ in range(60):
        joints = at_home.track(far, PALM_DOWN)
    position = at_home.forward(joints)[0]
    assert np.allclose(position, at_home.achieved[0], atol=0.01)            # it went as far as it can, not frozen at home
    assert position[0] > HOME[0] + 0.05 and np.linalg.norm(position - far) > 0.2
    for _ in range(30):                                                     # the target comes back to a reachable point
        joints = at_home.track(HOME + [0.05, 0.0, 0.0], PALM_DOWN)
    assert at_home.failures == 0 and np.allclose(at_home.forward(joints)[0], HOME + [0.05, 0.0, 0.0], atol=0.01)


def test_the_table_is_avoided_but_without_it_the_hand_goes_through(at_home):
    def lowest(solver):
        solver.start(HOME, PALM_DOWN)
        zs = []
        for s in np.linspace(0, 1, 100):
            joints = solver.track(HOME - [0.0, 0.0, 0.30 * s], PALM_DOWN)
            zs.append(solver.forward(joints)[0][2])
        return min(zs)
    assert lowest(at_home) > TABLE_TOP + 0.03                               # stays above the table
    assert lowest(ArmIK(avoid_table=False)) < TABLE_TOP                     # the same motion without the obstacle goes below its top
