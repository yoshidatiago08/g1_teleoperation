"""Interface every robot backend implements, and the hand actuator conventions they share."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from g1tele.retarget.retargeter import RobotTargets

# Inspire hand actuator ranges as the hardware (and unitree_sim_isaaclab) defines them, in radians
# of the driven joint, in protocol.FINGERS order: pinky, ring, middle, index, thumb bend, thumb yaw.
# A curl of 0 is the first number, a curl of 1 the second.
HAND_JOINT_RANGE = np.array([[0.0, 1.7], [0.0, 1.7], [0.0, 1.7], [0.0, 1.7], [0.0, 0.5], [-0.1, 1.3]])


def curl_to_joint(curls) -> np.ndarray:
    """Curls (0 open ... 1 closed) -> Inspire joint angles in radians."""
    c = np.clip(np.asarray(curls, dtype=float), 0.0, 1.0)
    return HAND_JOINT_RANGE[:, 0] + c * (HAND_JOINT_RANGE[:, 1] - HAND_JOINT_RANGE[:, 0])


def curl_to_dds(curls) -> np.ndarray:
    """Curls -> the [0, 1] command of the Inspire DDS interface (1 = open, 0 = closed)."""
    return 1.0 - np.clip(np.asarray(curls, dtype=float), 0.0, 1.0)


class RobotBackend(ABC):
    """Applies RobotTargets to something: a kinematic model, a MuJoCo scene, an Isaac sim over DDS."""

    #: rate (Hz) at which `step` should be called
    rate_hz: float = 100.0

    @abstractmethod
    def apply(self, targets: RobotTargets) -> None:
        """Set the newest targets. Cheap; the motion happens in `step`."""

    @abstractmethod
    def step(self) -> None:
        """Advance one control period."""

    def observe(self) -> Dict[str, float]:
        """Current joint positions by URDF joint name (empty if the backend cannot tell)."""
        return {}

    def reset_scene(self) -> None:
        """Put the objects of the scene back (the operator pressed 'o'). Optional."""

    def scene_status(self) -> Dict[str, float]:
        """Numbers about the scene for the operator's overlay (cylinder to target distance...). Optional."""
        return {}

    def render(self) -> Optional[np.ndarray]:
        """An RGB image of the robot, or None."""
        return None

    def close(self) -> None:
        pass
