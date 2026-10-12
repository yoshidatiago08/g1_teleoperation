"""Interface every robot backend implements, and the hand actuator conventions they share."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from g1tele.retarget.retargeter import RobotTargets

from g1tele.hand import HAND_JOINT_RANGE, curl_to_joint  # noqa: F401  (shared with the retargeting)


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

    def initialize(self, targets: RobotTargets) -> None:
        """Place the robot at `targets` before anything moves (so it does not sweep there at start)."""
        self.apply(targets)

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
