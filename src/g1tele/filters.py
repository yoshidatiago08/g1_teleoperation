"""Low-latency noise filters.

An exponential moving average has one knob: smooth OR responsive. For precise manipulation we
need both, so the signals go through a One Euro filter (Casiez et al. 2012): heavy smoothing while
the signal is nearly still (kills jitter), little smoothing while it moves fast (no lag).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One Euro filter for a scalar or a vector. Call with (value, time in seconds)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.5):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None       # filtered value
        self.dx = None      # filtered derivative
        self.t = None

    @property
    def velocity(self):
        return self.dx

    def __call__(self, value, t: float):
        value = np.asarray(value, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = value.copy(), np.zeros_like(value), t
            return self.x.copy()
        dt = max(t - self.t, 1e-4)
        self.t = t
        raw_dx = (value - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * raw_dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = _alpha(dt, cutoff)
        self.x = a * value + (1 - a) * self.x
        return self.x.copy()


class OneEuroRotation:
    """One Euro filter for rotation matrices (the speed is the angular speed in rad/s)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.5):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.r = None
        self.speed = 0.0
        self.t = None

    def __call__(self, r: np.ndarray, t: float):
        if self.r is None:
            self.r, self.t = r.copy(), t
            return self.r.copy()
        dt = max(t - self.t, 1e-4)
        self.t = t
        delta = Rotation.from_matrix(self.r.T @ r).as_rotvec()
        raw_speed = float(np.linalg.norm(delta)) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.speed = a_d * raw_speed + (1 - a_d) * self.speed
        a = _alpha(dt, self.min_cutoff + self.beta * self.speed)
        self.r = self.r @ Rotation.from_rotvec(a * delta).as_matrix()
        return self.r.copy()


class RingStat:
    """Fixed-size window of samples with robust statistics."""

    def __init__(self, size: int):
        self.size = size
        self.data = []

    def add(self, x):
        self.data.append(np.asarray(x, dtype=float))
        if len(self.data) > self.size:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data = []

    def median(self):
        return np.median(np.array(self.data), axis=0)

    def spread(self):
        """Distance of the 90th-percentile sample from the median (a robust wobble measure)."""
        arr = np.array(self.data)
        return float(np.percentile(np.linalg.norm(arr - np.median(arr, axis=0), axis=-1), 90))
