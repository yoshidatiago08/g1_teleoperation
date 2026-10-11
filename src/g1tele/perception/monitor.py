"""Keeps checking, while tracking, that the calibration still holds.

A calibration can be wrong from the start (a bad moment during the poses) or stop being right (the camera was
bumped, someone else stepped in). Without a check the only cure was restarting. Two checks run all the time:

  * bone lengths: the arm lengths measured over the last seconds against the calibrated ones
    (arms pointing at the camera are skipped: their depth is unreliable)
  * "up": from the IMU every frame, or the floor plane every second or so, against the calibrated "up"
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from g1tele.config import CalibrationCfg
from g1tele.geometry import unit
from g1tele.perception.gravity import estimate_up_from_floor


class CalibrationMonitor:
    def __init__(self, cfg: CalibrationCfg, sides):
        self.cfg, self.sides = cfg, list(sides)
        self.reset()

    def reset(self):
        self.bones: Dict[str, Deque] = {s: deque(maxlen=240) for s in self.sides}
        self.bone_bad_since: Optional[float] = None
        self.up_bad_since: Optional[float] = None
        self.last_up_check = -1e9
        self.status = 'ok'
        self.rng = np.random.default_rng(1)

    def update(self, t, profile, shoulders: Dict[str, Optional[np.ndarray]], meas, frame) -> Optional[str]:
        """Returns 'bones' or 'camera' when the calibration should be redone, else None."""
        cfg = self.cfg
        # ---- bone lengths
        for side in self.sides:
            m, s = meas[side], shoulders.get(side)
            if s is None or m['E'] is None or m['W'] is None or m['E'].P is None or m['W'].P is None:
                continue
            d = unit(m['W'].P - s)
            ray = unit(s)
            if d is None or ray is None or abs(float(d @ ray)) > 0.8:
                continue
            self.bones[side].append((np.linalg.norm(m['E'].P - s), np.linalg.norm(m['W'].P - m['E'].P)))
        bad = False
        enough = False
        for side in self.sides:
            b = profile.bones.get(side)
            if b is None or not b.valid() or len(self.bones[side]) < 100:
                continue
            enough = True
            med = np.median(np.array(self.bones[side]), axis=0)
            tol = cfg.drift_bone_pct / 100.0
            if abs(med[0] / b.upper - 1) > tol or abs(med[1] / b.fore - 1) > tol:
                bad = True
        if bad:
            self.bone_bad_since = self.bone_bad_since if self.bone_bad_since is not None else t
            if t - self.bone_bad_since > cfg.drift_seconds:
                return 'bones'
        elif enough:
            self.bone_bad_since = None
        # ---- "up"
        up_now = None
        if frame.gravity is not None:
            up_now = -np.asarray(frame.gravity)
        elif profile.up_source == 'floor' and t - self.last_up_check > 1.0:
            self.last_up_check = t
            res = estimate_up_from_floor(frame.depth, frame.intrinsics, profile.up, self.rng)
            up_now = None if res is None else res[0]
        if up_now is not None and profile.up is not None and profile.up_source in ('imu', 'floor'):
            ang = float(np.degrees(np.arccos(np.clip(up_now @ profile.up, -1, 1))))
            if ang > cfg.drift_up_deg:
                self.up_bad_since = self.up_bad_since if self.up_bad_since is not None else t
                if t - self.up_bad_since > 3.0:
                    return 'camera'
            else:
                self.up_bad_since = None
        self.status = 'CHECK' if (bad or self.up_bad_since is not None) else 'ok'
        return None
