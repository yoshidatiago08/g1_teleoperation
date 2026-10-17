"""Smoothing of the tracked hand pose that knows which directions the image does not determine.

The hand fit (hand_model.HandFit) returns, with every pose, how uncertain it is: a covariance for the palm rotation and for the wrist
position, in camera axes. A flat palm facing the camera has a huge uncertainty about the two axes in its plane (its tilt hardly changes the
picture); the distance to the camera is always the weak position axis. That is where MediaPipe's noise turns into jitter.

`PoseSmoother` is a pair of Kalman filters with a constant-velocity motion model, one for the rotation and one for the position, whose
measurement noise is that covariance per frame. So in a direction the image fixes well the estimate follows the measurement almost at once
(no lag), and in a weakly observed direction it follows the motion model and the measurement only nudges it. A measurement that disagrees with
the prediction far more than its covariance explains (a mirror-image flip of the fit, a MediaPipe glitch) is down-weighted; if it keeps
disagreeing the hand really did move and the filter restarts there.
"""
import numpy as np

from dex_hand.frames import rotation_from_vec, rotvec

ROT_ACCEL = (0.4, 30.0)     # rad/s^2, (steady, moving): how hard the palm may accelerate (process noise, one sigma). The filter moves between
POS_ACCEL = (0.12, 8.0)      # m/s^2                       the two by how well the measurements agree with plain noise around its estimate
MOVING_FROM, MOVING_SPAN = 3.0, 5.0   # the averaged innovation, in units of what noise alone would leave, where it starts to count as motion, and the range to full
INNOVATION_MEMORY = 0.5     # exponential average over frames
MEASUREMENT_FLOOR_DEG = 0.7 # a measurement is never trusted better than this (deg) / the next one (mm)
POSITION_FLOOR = 0.001
GATE = 16.0                 # squared Mahalanobis distance above which a measurement is treated as an outlier
RESTART_AFTER = 6           # consecutive outliers: the hand really moved, restart from the measurement


class _Cv:
    """Constant-velocity Kalman filter on a 3-vector (or a rotation vector): state (value, rate), P is 6x6."""

    def __init__(self, accel, floor):
        self.accel, self.floor = accel, floor ** 2
        self.moving = 0.0                                              # 0 = steady, 1 = moving
        self.q = accel[0] ** 2

    def start(self, value, cov):
        self.rate = np.zeros(3)
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = cov
        self.P[3:, 3:] = np.eye(3) * 100.0
        self.outliers = 0
        self.moving, self.mean_innovation = 0.0, np.zeros(3)

    def predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        Q = np.zeros((6, 6))
        Q[:3, :3] = dt ** 3 / 3 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = dt ** 2 / 2 * np.eye(3)
        Q[3:, 3:] = dt * np.eye(3)
        self.q = (self.accel[0] + self.moving * (self.accel[1] - self.accel[0])) ** 2
        self.P = F @ self.P @ F.T + self.q * Q

    def correct(self, innovation, cov):
        """Returns the correction (value, rate) for the innovation (measurement - prediction)."""
        R = cov + self.floor * np.eye(3)
        S = self.P[:3, :3] + R
        m2 = float(innovation @ np.linalg.solve(S, innovation))
        if m2 > GATE:
            self.outliers += 1
            if self.outliers >= RESTART_AFTER:
                return None
            R = R * (m2 / GATE)                                        # an outlier is trusted less, in proportion to how far out it is
            S = self.P[:3, :3] + R
        else:
            self.outliers = 0
            # Noise around a steady hand gives innovations of changing sign that average out; a hand that is really moving away from the estimate
            # gives innovations of one sign, which add up. (Not the size of the innovation: a larger process noise makes it larger, which
            # would keep the filter 'moving' for ever.)
            a = INNOVATION_MEMORY
            self.mean_innovation = self.mean_innovation + a * (innovation - self.mean_innovation)
            level = float(self.mean_innovation @ np.linalg.solve(R, self.mean_innovation)) / (3.0 * a / (2.0 - a))
            self.moving = float(np.clip((level - MOVING_FROM) / MOVING_SPAN, 0.0, 1.0))
        K = self.P[:, :3] @ np.linalg.inv(S)
        self.P = self.P - K @ self.P[:3]
        correction = K @ innovation
        return correction[:3], correction[3:]


class PoseSmoother:
    def __init__(self, rot_accel=ROT_ACCEL, pos_accel=POS_ACCEL):
        self._rot = _Cv(rot_accel, np.radians(MEASUREMENT_FLOOR_DEG))
        self._pos = _Cv(pos_accel, POSITION_FLOOR)
        self.r = self.p = None

    def reset(self):
        self.r = self.p = None

    def update(self, r_meas, p_meas, cov_rot, cov_pos, dt):
        """The smoothed (rotation, position) for this frame's measurement and its covariances (camera axes). `dt`: seconds since the last frame."""
        r_meas, p_meas = np.asarray(r_meas, dtype=float), np.asarray(p_meas, dtype=float)
        if cov_rot is None or cov_pos is None:                  # no uncertainty known: trust the measurement (no smoothing)
            self.reset()
            return r_meas, p_meas
        dt = float(np.clip(dt if dt else 1 / 30, 1 / 120, 0.2))
        if self.r is None:
            self.r, self.p = r_meas.copy(), p_meas.copy()
            self._rot.start(None, cov_rot)
            self._pos.start(None, cov_pos)
            return self.r.copy(), self.p.copy()
        # predict
        self.r = rotation_from_vec(self._rot.rate * dt) @ self.r
        self.p = self.p + self._pos.rate * dt
        self._rot.predict(dt)
        self._pos.predict(dt)
        # correct
        out = self._rot.correct(rotvec(r_meas @ self.r.T), cov_rot)
        restart_r = out is None
        if not restart_r:
            self.r = rotation_from_vec(out[0]) @ self.r
            self._rot.rate = self._rot.rate + out[1]
        out = self._pos.correct(p_meas - self.p, cov_pos)
        restart_p = out is None
        if not restart_p:
            self.p = self.p + out[0]
            self._pos.rate = self._pos.rate + out[1]
        if restart_r:
            self.r = r_meas.copy()
            self._rot.start(None, cov_rot)
        if restart_p:
            self.p = p_meas.copy()
            self._pos.start(None, cov_pos)
        return self.r.copy(), self.p.copy()


# -- stage 1: MediaPipe's own landmarks, before anything is computed from them ---------------------------------------------------------------

LANDMARK_MIN_CUTOFF = 1.0     # Hz: how fast a landmark may change while the hand is steady (low = steadier)
LANDMARK_BETA = 15.0          # cutoff rises by this many Hz per (hand length / s) of speed, so a moving hand is not lagged
LANDMARK_DERIVATIVE_CUTOFF = 2.0
SPIKE_FRACTION = 0.3          # a landmark that jumps this fraction of the hand's size relative to the rest of the hand, in one frame, is held back one frame


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class LandmarkFilter:
    """Temporal filter for the 21 landmarks (pixels and MediaPipe's world coordinates), per landmark, before the hand model sees them.

    MediaPipe's landmarks shake from frame to frame, some landmarks (hidden fingers, fingertips against the palm, an edge-on hand) far more than
    others, and sometimes one landmark jumps for a single frame. Two things:
      * a One Euro filter on each landmark: strong smoothing while it is steady, a cutoff that rises with its speed (measured in hand lengths per
        second, so it does not depend on how far the hand is) so that real motion is not lagged;
      * spike rejection: a landmark that moves far relative to the rest of the hand in one frame is not believed at once. If the next frame
        confirms it (it stays there) it was a real movement and is taken; if it falls back it was a glitch and is dropped.
    """

    def __init__(self, min_cutoff=LANDMARK_MIN_CUTOFF, beta=LANDMARK_BETA):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self):
        self._x = None

    def update(self, world, pixels, dt):
        """(world 21x3, pixels 21x2) filtered; `dt` seconds since the last frame."""
        world, pixels = np.asarray(world, dtype=float), np.asarray(pixels, dtype=float)
        dt = float(np.clip(dt if dt else 1 / 30, 1 / 120, 0.2))
        z = np.concatenate([pixels, world], axis=1)                       # 21 x 5: u, v, x, y, z
        if self._x is None:
            self._x, self._dx, self._last = z.copy(), np.zeros_like(z), z.copy()
            self._pending = np.zeros(21, bool)
            self._pending_at = pixels.copy()
            self._size = max(np.linalg.norm(pixels[9] - pixels[0]), 1.0)
            self._world_size = max(np.linalg.norm(world[9] - world[0]), 0.02)
            return world.copy(), pixels.copy()
        # the hand's size in pixels (a hand seen end-on looks short: do not let that make the thresholds tiny)
        self._size += 0.2 * (max(np.linalg.norm(pixels[9] - pixels[0]), 0.5 * self._size) - self._size)

        # spike rejection: how far each landmark moved relative to the rest of the hand since the last frame
        common = np.median(z - self._last, axis=0)
        relative = np.linalg.norm((pixels - self._last[:, :2]) - common[:2], axis=1)
        limit = SPIKE_FRACTION * self._size
        jump = relative > limit
        stayed = self._pending & (np.linalg.norm(pixels - self._pending_at, axis=1) < 0.5 * limit)
        glitch = jump & ~stayed
        self._pending_at = np.where(glitch[:, None], pixels, self._pending_at)
        self._pending = glitch
        measurement = np.where(glitch[:, None], self._x + common, z)       # a landmark not believed yet is assumed to have moved with the hand
        self._last = np.where(glitch[:, None], self._last + common, z)
        for i in np.where(stayed)[0]:                                        # it stayed there: a real movement, take it without lag
            self._x[i], self._dx[i] = z[i], 0.0

        # One Euro filter, per landmark; the speed that opens the cutoff is in hand lengths per second
        dx = (measurement - self._x) / dt
        self._dx += _alpha(LANDMARK_DERIVATIVE_CUTOFF, dt) * (dx - self._dx)
        speed_px = np.linalg.norm(self._dx[:, :2], axis=1) / self._size
        speed_world = np.linalg.norm(self._dx[:, 2:], axis=1) / self._world_size
        cutoff = self.min_cutoff + self.beta * np.column_stack([speed_px] * 2 + [speed_world] * 3)
        self._x += _alpha(cutoff, dt) * (measurement - self._x)
        return self._x[:, 2:].copy(), self._x[:, :2].copy()
