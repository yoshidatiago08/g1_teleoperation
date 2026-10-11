"""A backend with no physics: the joints are exactly the targets. For tests and dry runs."""
from __future__ import annotations

from typing import Dict

from g1tele.config import Config
from g1tele.retarget.retargeter import RobotTargets
from g1tele.robot.base import RobotBackend


class KinematicBackend(RobotBackend):
    def __init__(self, cfg: Config):
        self.rate_hz = cfg.robot.rate_hz
        self.joints: Dict[str, float] = {}
        self.targets = RobotTargets()
        self.steps = 0

    def apply(self, targets: RobotTargets) -> None:
        self.targets = targets
        self.joints.update(targets.arm_q)
        self.joints.update(targets.waist_q)

    def step(self) -> None:
        self.steps += 1

    def observe(self) -> Dict[str, float]:
        return dict(self.joints)
