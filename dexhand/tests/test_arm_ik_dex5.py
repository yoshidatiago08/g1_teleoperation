"""CuRobo right-arm IK with the Dex5-1 mounted (needs curobo and a CUDA GPU: the VM image)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('curobo')
if not torch.cuda.is_available():
    pytest.skip('no CUDA GPU', allow_module_level=True)

from dex_hand import frames, paths
from dex_hand.arm_ik import ArmIK
from dex_hand.retarget import robot_hand_rotation

HOME = np.array([0.22, -0.15, 0.98])


def wxyz(rotation):
    x, y, z, w = frames.quaternion(rotation)
    return np.array([w, x, y, z])


def test_the_arm_reaches_the_home_pose_and_tracks_with_the_dex5_hand():
    palm_down = wxyz(np.diag([1.0, -1.0, -1.0]) @ robot_hand_rotation(paths.urdf('right', 'dex5'), 'right').T)
    ik = ArmIK(hand_model='dex5')
    assert ik.start(HOME, palm_down) is not None
    for i in range(30):
        joints = ik.track(HOME + [0.003 * i, 0.0, 0.0], palm_down)
    assert np.linalg.norm(ik.forward(joints)[0] - (HOME + [0.087, 0.0, 0.0])) < 0.01
    assert ik.failures == 0
