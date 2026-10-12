"""The guided calibration that starts hands-only mode. The robot holds still while it runs.

  approach  move closer until only your hands are in view, in the distance range
  open      open both hands flat and hold: learns this person's open-hand finger readings at close range
  fist      close both hands and hold: the closed-hand readings
  neutral   hold the hands as you will start (the control is relative to this pose) and keep still

Pure logic: it is given, every frame, what each hand is doing, and answers with what to tell the operator.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

STEPS = ('approach', 'open', 'fist', 'neutral')
OPEN_MAX_SUM = 1.2         # rad: summed flexion of the four fingers (mean) below this is an open hand ...
FIST_MIN_SUM = 2.6         # ... above this a closed one (the usual range is 0.4 open ... 3.6 closed)
STILL_M = 0.03             # the palm may not move more than this over the last half second


class HandsEnterCalibration:
    def __init__(self, sides, depth_range, hold_s: float = 1.0, timeout_s: float = 90.0):
        self.sides, self.depth_range, self.hold_s, self.timeout_s = list(sides), depth_range, hold_s, timeout_s
        self.step = 0
        self.t_start: Optional[float] = None
        self.since: Optional[float] = None
        self.samples: Dict[str, List[np.ndarray]] = {s: [] for s in self.sides}
        self.history = {s: deque() for s in self.sides}                # (t, palm position)
        self.open_raw: Dict[str, np.ndarray] = {}
        self.closed_raw: Dict[str, np.ndarray] = {}
        self.warning = ''

    @property
    def name(self) -> str:
        return STEPS[min(self.step, len(STEPS) - 1)]

    def _still(self, side: str, t: float, p) -> bool:
        h = self.history[side]
        h.append((t, np.asarray(p, dtype=float)))
        while h and t - h[0][0] > 0.5:
            h.popleft()
        pts = np.array([q for _, q in h])
        return len(pts) >= 3 and float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) < STILL_M

    def _advance(self):
        self.step += 1
        self.since = None
        self.samples = {s: [] for s in self.sides}

    def update(self, t: float, hands: Dict[str, dict]) -> Tuple[str, bool, bool]:
        """hands[side] = {'depth': palm distance (m), 'raw': 6 raw finger features or None, 'p': palm position}.
        Returns (what to tell the operator, finished, failed)."""
        if self.t_start is None:
            self.t_start = t
        if t - self.t_start > self.timeout_s:
            return 'Hands-only calibration timed out', False, True
        if self.step >= len(STEPS):
            return '', True, False
        missing = [s for s in self.sides if s not in hands]
        if missing:
            self.since = None
            return f'Show both hands to the camera ({", ".join(missing)} not seen)', False, False
        still = all(self._still(s, t, hands[s]['p']) for s in self.sides)
        name = self.name
        lo, hi = self.depth_range
        raw = {s: hands[s]['raw'] for s in self.sides}
        flex = {s: (None if raw[s] is None else float(np.mean(raw[s][:4]))) for s in self.sides}
        ok, text = False, ''
        if name == 'approach':
            depths = [hands[s]['depth'] for s in self.sides]
            if max(depths) > hi:
                text = f'Move closer: hands at {max(depths):.2f} m, want {lo:.2f}-{hi:.2f} m. Keep your face in view.'
            elif min(depths) < lo:
                text = f'A little further: hands at {min(depths):.2f} m, want {lo:.2f}-{hi:.2f} m'
            else:
                ok, text = True, 'Good distance. Hold it'
        elif name == 'open':
            ok = still and all(v is not None and v < OPEN_MAX_SUM for v in flex.values())
            text = 'Open both hands flat, palms to the camera, and hold still'
        elif name == 'fist':
            ok = still and all(v is not None and v > FIST_MIN_SUM for v in flex.values())
            text = 'Close both hands into fists and hold still'
        else:
            ok = still
            text = 'Hold your hands as you want to start (relaxed), and keep still'
        if not ok:
            self.since = None
            self.samples = {s: [] for s in self.sides}
            return f'[{self.step + 1}/4] {text}', False, False
        if self.since is None:
            self.since = t
        for s in self.sides:
            if raw[s] is not None:
                self.samples[s].append(np.asarray(raw[s], dtype=float))
        left = self.hold_s - (t - self.since)
        if left > 0:
            return f'[{self.step + 1}/4] {text} ({left:.1f} s)', False, False
        if name == 'open':
            self.open_raw = {s: np.mean(self.samples[s], axis=0) for s in self.sides if self.samples[s]}
        elif name == 'fist':
            self.closed_raw = {s: np.mean(self.samples[s], axis=0) for s in self.sides if self.samples[s]}
            self._check_separation()
        self._advance()
        return ('Hands-only ready' if self.step >= len(STEPS) else f'[{self.step + 1}/4] next...'), self.step >= len(STEPS), False

    def _check_separation(self):
        bad = [s for s in self.sides if s in self.open_raw and s in self.closed_raw
               and float(np.mean(self.closed_raw[s][:4] - self.open_raw[s][:4])) < 1.0]
        self.warning = f'open and closed hand look alike for the {", ".join(bad)} hand: fingers may be poor' if bad else ''
