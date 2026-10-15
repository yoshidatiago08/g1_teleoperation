"""Record what MediaPipe saw (landmarks, not images) and replay it, to study the hand model on a real hand offline.

    VM:   RECORD=recordings/take1.npz docker compose --profile vm up          (also: python -m dex_hand.vm --record FILE)
    then: python tools/analyse_recording.py recordings/take1.npz               (or --source replay --replay FILE to drive the whole VM with it)

Only numbers are stored: per frame the 21 world landmarks, the 21 pixels, the wrist depth when there is a sensor, the camera's
focal length / principal point, the IMU's up vector, the time, and whether the operator was calibrating. No camera images.
"""
import time

import numpy as np

from dex_hand.sources import Observation


class Recorder:
    def __init__(self, path, flush_every=150):
        self.path, self.flush_every, self.rows = path, flush_every, []

    def add(self, obs, focal, cx, cy, up, calibrating, t=None):
        nan = np.full((21, 3), np.nan)
        self.rows.append({'t': time.time() if t is None else t, 'seen': obs.world is not None,
                          'world': nan if obs.world is None else obs.world, 'pixels': nan[:, :2] if obs.pixels is None else obs.pixels,
                          'depth': np.nan if obs.depth is None else obs.depth, 'intrinsics': (focal, cx, cy),
                          'up': np.full(3, np.nan) if up is None else np.asarray(up, dtype=float), 'calibrating': bool(calibrating)})
        if len(self.rows) % self.flush_every == 0:
            self.save()

    def save(self):
        if not self.rows:
            return
        r = self.rows
        np.savez_compressed(self.path, t=[x['t'] for x in r], seen=[x['seen'] for x in r], world=[x['world'] for x in r],
                            pixels=[x['pixels'] for x in r], depth=[x['depth'] for x in r],
                            intrinsics=[x['intrinsics'] for x in r], up=[x['up'] for x in r],
                            calibrating=[x['calibrating'] for x in r])


def load(path):
    return dict(np.load(path))


class ReplaySource:
    """Same interface as the camera sources: read() -> Observation. Plays a recording at its own speed, once, then repeats the last frame (hand lost)."""

    def __init__(self, path, speed=1.0):
        self.data, self.speed, self.i, self.start = load(path), speed, 0, None
        self.focal_px, self.cx, self.cy = self.data['intrinsics'][0]
        self.calibrating = False

    def read(self):
        d, n = self.data, len(self.data['t'])
        if self.start is None:
            self.start = time.monotonic()
        if self.i >= n:
            time.sleep(0.03)
            return Observation(None, None, None, None)
        due = self.start + (d['t'][self.i] - d['t'][0]) / self.speed
        time.sleep(max(due - time.monotonic(), 0.0))
        i, self.i = self.i, self.i + 1
        self.calibrating = bool(d['calibrating'][i])
        if not d['seen'][i]:
            return Observation(None, None, None, None)
        depth = None if np.isnan(d['depth'][i]) else float(d['depth'][i])
        return Observation(d['world'][i], d['pixels'][i], depth, None)

    def close(self):
        pass
