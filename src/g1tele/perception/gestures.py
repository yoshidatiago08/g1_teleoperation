"""Head gestures from the Pose model's nose, to switch modes without touching anything.

A nod: the nose goes DOWN by `amplitude` head widths relative to where it has been resting, and comes back,
with little sideways movement (a head shake or a turn is not a nod). `count` nods within `window_s` trigger.
The resting position follows slow movements (leaning, a tilted head) with a time constant of `base_tau`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE = 0, 7, 8, 2, 5


def head_width(uv, vis, min_vis: float = 0.5) -> Optional[float]:
    """Pixels between the ears (or twice-ish the eyes' distance); None if the head cannot be measured."""
    if min(vis[LEFT_EAR], vis[RIGHT_EAR]) >= min_vis:
        return float(np.linalg.norm(uv[LEFT_EAR] - uv[RIGHT_EAR]))
    if min(vis[LEFT_EYE], vis[RIGHT_EYE]) >= min_vis:
        return float(2.2 * np.linalg.norm(uv[LEFT_EYE] - uv[RIGHT_EYE]))
    return None


class NodDetector:
    def __init__(self, amplitude: float = 0.15, count: int = 2, window_s: float = 1.6, cooldown_s: float = 2.5,
                 base_tau: float = 1.2):
        self.amplitude, self.count, self.window_s, self.cooldown_s, self.base_tau = \
            amplitude, count, window_s, cooldown_s, base_tau
        self.reset()

    def reset(self):
        self.base: Optional[np.ndarray] = None
        self.t_last: Optional[float] = None
        self.down = False                  # the nose is below the threshold, a nod is under way
        self.cycle_dx = 0.0                # sideways travel during this nod
        self.peak_dy = 0.0                 # how far down it went
        self.nods: List[float] = []        # times of the completed nods
        self.blocked_until = 0.0

    def update(self, t: float, nose, width: Optional[float]) -> bool:
        """Feed the nose position (pixels) and the head width (pixels). True at the moment the last nod completes."""
        if width is None or width < 4.0:
            self.t_last = None
            return False
        nose = np.asarray(nose, dtype=float)
        if self.base is None or self.t_last is None:
            self.base, self.t_last = nose.copy(), t
            return False
        dt = max(t - self.t_last, 1e-3)
        self.t_last = t
        dx, dy = (nose - self.base) / width
        if not self.down:                  # the resting position only follows while no nod is under way
            self.base += (nose - self.base) * (1.0 - np.exp(-dt / self.base_tau))
        self.nods = [n for n in self.nods if t - n < self.window_s]
        if t < self.blocked_until:
            self.down = False
            return False
        if not self.down:
            if dy > self.amplitude:
                self.down, self.cycle_dx, self.peak_dy = True, abs(dx), dy
        else:
            self.cycle_dx = max(self.cycle_dx, abs(dx))
            self.peak_dy = max(self.peak_dy, dy)
            if dy < 0.4 * self.amplitude:  # came back up
                self.down = False
                if self.cycle_dx < 0.6 * self.peak_dy:                         # not a shake or a turn
                    self.nods.append(t)
                if len(self.nods) >= self.count:
                    self.nods.clear()
                    self.blocked_until = t + self.cooldown_s
                    return True
        return False
