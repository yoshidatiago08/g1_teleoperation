"""Calibration: what the camera cannot know by itself.

A full calibration walks the operator through a few poses and learns
  * bone lengths (shoulder-elbow, elbow-wrist) from every pose, so depth noise averages out and the
    result is checked for consistency between poses and between the two arms
  * "up": from the IMU when the camera has one, otherwise from the arms themselves (hanging arms
    are vertical, arms held out are horizontal), so the body frame is not tilted by the camera
  * the body frame: facing = the shoulder line now, up = gravity
  * the shoulder width (to place a shoulder that is briefly hidden)
  * this person's finger range (open hand, then fist)
The person-specific part (bones, shoulder width, finger range) is saved to a profile; next time
only the rest pose is needed (a few seconds) to re-establish the body frame.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from g1tele.config import CalibrationCfg, PerceptionCfg
from g1tele.geometry import frame_from, unit
from g1tele.hand import DEFAULT_CLOSED, DEFAULT_OPEN, curls
from g1tele.protocol import BoneLengths

POSE_PROMPTS = {
    'rest': 'Arms hanging relaxed at your sides. Hold still.',
    'front': 'Both arms straight out in FRONT of you, at shoulder height. Hold still.',
    'side': 'Both arms straight out to the SIDES (T-pose). Hold still.',
    'up': 'Both arms straight UP. Hold still.',
    'hand_open': 'Open both hands wide, fingers spread. Hold still.',
    'hand_fist': 'Make two tight fists. Hold still.',
}


@dataclass
class Profile:
    """Everything calibration learns about a person (and, for `frame`, this session)."""
    bones: Dict[str, BoneLengths] = field(default_factory=dict)
    shoulder_width: float = 0.0
    hand_open: List[float] = field(default_factory=lambda: DEFAULT_OPEN.tolist())
    hand_closed: List[float] = field(default_factory=lambda: DEFAULT_CLOSED.tolist())
    # session: body frame (columns forward, left, up) in camera coordinates, and "up" source
    frame: Optional[np.ndarray] = None
    up: Optional[np.ndarray] = None
    up_source: str = ''
    report: Dict[str, float] = field(default_factory=dict)

    def to_json(self):
        return {'bones': {s: [b.upper, b.fore] for s, b in self.bones.items()},
                'shoulder_width': self.shoulder_width, 'hand_open': self.hand_open,
                'hand_closed': self.hand_closed, 'report': self.report}

    @classmethod
    def from_json(cls, d):
        return cls(bones={s: BoneLengths(*v) for s, v in d['bones'].items()},
                   shoulder_width=d['shoulder_width'], hand_open=d['hand_open'],
                   hand_closed=d['hand_closed'], report=d.get('report', {}))

    def save(self, directory: str, operator: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f'{operator}.json'), 'w') as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, directory: str, operator: str) -> Optional['Profile']:
        path = os.path.join(directory, f'{operator}.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls.from_json(json.load(f))


def estimate_up(directions: List[tuple]) -> Optional[np.ndarray]:
    """Least-squares "up" in the camera frame from arm directions with known relation to gravity.

    `directions`: (unit vector shoulder->wrist, kind) with kind 'down' (hanging arm: vector is
    -up), 'level' (arm held horizontally: vector is perpendicular to up) or 'up'.
    """
    rows, rhs = [], []
    for d, kind in directions:
        if kind == 'down':
            rows.append(-d), rhs.append(1.0)
        elif kind == 'up':
            rows.append(d), rhs.append(1.0)
        else:
            rows.append(d), rhs.append(0.0)
    if len(rows) < 3:
        return None
    a, b = np.array(rows), np.array(rhs)
    # also pin the scale: |up| = 1 is imposed by normalising afterwards
    u, *_ = np.linalg.lstsq(a, b, rcond=None)
    return unit(u)


class Calibrator:
    """Stage machine. Feed it `update(...)` every frame; it returns the prompt to show."""

    def __init__(self, cfg: CalibrationCfg, perception: PerceptionCfg, existing: Optional[Profile] = None,
                 full: bool = True):
        self.cfg, self.pcfg = cfg, perception
        self.sides = list(perception.arms)
        have_person = existing is not None and all(s in existing.bones and existing.bones[s].valid()
                                                   for s in self.sides)
        self.profile = existing if existing is not None else Profile()
        if full or not have_person:
            self.stages = [p for p in cfg.poses] + (['hand_open', 'hand_fist']
                                                     if cfg.hand_poses and perception.hands else [])
        else:
            self.stages = ['rest']
        if 'rest' not in self.stages:
            self.stages.insert(0, 'rest')
        self.full = full or not have_person
        self.stage_idx = 0
        self.samples: List[dict] = []
        self.hold_start: Optional[float] = None
        self.results: Dict[str, dict] = {}
        self.done = False
        self.message = ''
        self.gravity_up: Optional[np.ndarray] = None
        self.floor_samples: List[np.ndarray] = []

    @property
    def stage(self) -> Optional[str]:
        return None if self.done else self.stages[self.stage_idx]

    def prompt(self) -> str:
        if self.done:
            return 'Calibration done.'
        idx = f'[{self.stage_idx + 1}/{len(self.stages)}] '
        return idx + POSE_PROMPTS[self.stage] + (f'  ({self.message})' if self.message else '')

    # ------------------------------------------------------------- pose checks
    def _pose_ok(self, stage: str, pts: Dict[str, Dict[str, np.ndarray]], up0: np.ndarray,
                 left: Optional[np.ndarray]) -> str:
        """'' if the operator is in the pose, else what is wrong."""
        for side, p in pts.items():
            s, e, w = p.get('S'), p.get('E'), p.get('W')
            if s is None or e is None or w is None:
                return f'{side} arm not fully visible'
            d = unit(w - s)
            length = np.linalg.norm(w - s)
            if d is None or not 0.25 < length < 1.0:
                return f'{side} arm length looks wrong'
            vertical = float(d @ up0)
            if stage == 'rest':
                if vertical > -np.cos(np.radians(35)):
                    return f'let your {side} arm hang down'
            elif stage == 'up':
                if vertical < np.cos(np.radians(30)):
                    return f'raise your {side} arm higher'
            elif stage in ('front', 'side'):
                if abs(vertical) > np.sin(np.radians(25)):
                    return f'keep your {side} arm level'
                if left is not None:
                    across = float(d @ left)
                    out = -1.0 if side == 'right' else 1.0
                    if stage == 'front' and abs(across) > np.sin(np.radians(30)):
                        return f'point your {side} arm straight ahead'
                    if stage == 'side' and out * across < np.cos(np.radians(30)):
                        return f'point your {side} arm out to the side'
        return ''

    # ------------------------------------------------------------- per frame
    def update(self, t: float, pts: Dict[str, Dict[str, np.ndarray]], shoulders: Optional[tuple],
               up_cam: np.ndarray, hand_raw: Dict[str, Optional[np.ndarray]],
               gravity_up: Optional[np.ndarray] = None,
               floor_up: Optional[np.ndarray] = None) -> str:
        """pts[side] = {'S','E','W'} camera-frame points (None if missing); shoulders = (right, left)
        camera points or None; up_cam = best current guess of up; hand_raw[side] = raw hand features;
        gravity_up = "up" from the camera's IMU, if it has one."""
        if self.done:
            return self.prompt()
        if gravity_up is not None:
            self.gravity_up = gravity_up
        if floor_up is not None:
            self.floor_samples.append(floor_up)
        stage = self.stage
        left = None if shoulders is None or shoulders[0] is None or shoulders[1] is None \
            else unit(shoulders[1] - shoulders[0])
        if stage.startswith('hand'):
            problem = ''
            if not all(hand_raw.get(s) is not None for s in self.sides):
                problem = 'show both hands'
            else:
                # the pose must really be an open hand / a fist (judged with the default range)
                for side in self.sides:
                    level = float(np.mean(curls(hand_raw[side])[:4]))
                    if stage == 'hand_open' and level > 0.35:
                        problem = f'open your {side} hand wider'
                    elif stage == 'hand_fist' and level < 0.65:
                        problem = f'close your {side} hand into a tighter fist'
        else:
            problem = self._pose_ok(stage, pts, up_cam, left)
        if problem:
            self.message, self.samples, self.hold_start = problem, [], None
            return self.prompt()
        sample = {'t': t, 'pts': {s: {k: v.copy() for k, v in p.items() if v is not None}
                                  for s, p in pts.items()},
                  'sh': None if shoulders is None else tuple(None if x is None else x.copy() for x in shoulders),
                  'hand': {s: None if v is None else v.copy() for s, v in hand_raw.items()}}
        # stillness: the wrist must stay close to where it was at the start of the hold
        if self.hold_start is None:
            self.hold_start = t
            self.samples = []
        self.samples.append(sample)
        if not stage.startswith('hand') and len(self.samples) > 3:
            wob = []
            for side in self.sides:
                arr = np.array([sm['pts'][side]['W'] - sm['pts'][side]['S'] for sm in self.samples])
                wob.append(np.percentile(np.linalg.norm(arr - np.median(arr, axis=0), axis=1), 90))
            if max(wob) > self.cfg.still_cm / 100.0:
                self.message, self.samples, self.hold_start = 'hold still', [], None
                return self.prompt()
        need = self.cfg.rest_seconds if stage == 'rest' else self.cfg.pose_seconds
        remaining = need - (t - self.hold_start)
        if remaining > 0 or len(self.samples) < 10:
            self.message = f'{max(remaining, 0):.1f} s'
            return self.prompt()
        self.results[stage] = self._summarise(stage)
        self.samples, self.hold_start, self.message = [], None, ''
        self.stage_idx += 1
        if self.stage_idx >= len(self.stages):
            self._finish()
        return self.prompt()

    def _summarise(self, stage: str) -> dict:
        out: dict = {}
        if stage.startswith('hand'):
            out['raw'] = {s: np.median(np.array([sm['hand'][s] for sm in self.samples
                                                 if sm['hand'].get(s) is not None]), axis=0)
                          for s in self.sides}
            return out
        for side in self.sides:
            S = np.array([sm['pts'][side]['S'] for sm in self.samples])
            E = np.array([sm['pts'][side]['E'] for sm in self.samples])
            W = np.array([sm['pts'][side]['W'] for sm in self.samples])
            d = unit(np.median(W - S, axis=0))
            ray = unit(np.median(S, axis=0))
            out[side] = {'upper': float(np.median(np.linalg.norm(E - S, axis=1))),
                         'fore': float(np.median(np.linalg.norm(W - E, axis=1))),
                         'dir': d, 'S': np.median(S, axis=0),
                         # an arm pointing at the camera hides its own elbow and shoulder from
                         # the depth sensor: such a pose says little about the bone lengths
                         'end_on': bool(d is not None and abs(float(d @ ray)) > 0.7)}
        shoulders = [sm['sh'] for sm in self.samples
                     if sm['sh'] is not None and all(x is not None for x in sm['sh'])]
        if shoulders:
            out['shoulders'] = (np.median([s[0] for s in shoulders], axis=0),
                                np.median([s[1] for s in shoulders], axis=0))
        return out

    # ------------------------------------------------------------- result
    def _finish(self):
        gravity_up = self.gravity_up
        p = self.profile
        pose_stages = [s for s in self.stages if not s.startswith('hand') and s in self.results]
        if self.full:
            usable = [st for st in pose_stages
                      if not any(self.results[st][s]['end_on'] for s in self.sides)] or pose_stages
            uppers = {s: [self.results[st][s]['upper'] for st in usable] for s in self.sides}
            fores = {s: [self.results[st][s]['fore'] for st in usable] for s in self.sides}
            u_med = {s: float(np.median(uppers[s])) for s in self.sides}
            f_med = {s: float(np.median(fores[s])) for s in self.sides}
            # Arms are nearly symmetric (a few mm): averaging both sides halves the noise
            u_all, f_all = float(np.mean(list(u_med.values()))), float(np.mean(list(f_med.values())))
            for s in self.sides:
                p.bones[s] = BoneLengths(u_all, f_all)
            spreads = [np.ptp(v) for v in list(uppers.values()) + list(fores.values()) if len(v) > 1]
            p.report['bone_spread_cm'] = round(100 * float(max(spreads, default=0.0)), 2)
            p.report['asymmetry_cm'] = round(100 * float(max(
                abs(u_med[a] - u_med[b]) for a in self.sides for b in self.sides)), 2)
        # shoulder width and the rest-pose body frame
        rest = self.results['rest']
        if 'shoulders' in rest:
            r_sh, l_sh = rest['shoulders']
            p.shoulder_width = float(np.linalg.norm(l_sh - r_sh))
        # "up"
        dirs = []
        for st in pose_stages:
            kind = {'rest': 'down', 'up': 'up'}.get(st, 'level')
            if any(self.results[st][s]['end_on'] for s in self.sides):
                continue     # arms pointing at the camera: the shoulder depth is unreliable
            for s in self.sides:
                d = self.results[st][s]['dir']
                if d is not None:
                    dirs.append((d, kind))
        up_est = estimate_up(dirs) if self.full else None
        floor_up = None
        if len(self.floor_samples) >= 5:
            fl = unit(np.median(np.array(self.floor_samples), axis=0))
            spread = max(float(np.degrees(np.arccos(np.clip(fl @ x, -1, 1)))) for x in self.floor_samples)
            if fl is not None and spread < 3.0:
                floor_up = fl
        if gravity_up is not None:
            up, src = gravity_up, 'imu'
            if up_est is not None:
                p.report['imu_vs_arms_deg'] = round(float(np.degrees(np.arccos(np.clip(up @ up_est, -1, 1)))), 1)
        elif floor_up is not None:
            up, src = floor_up, 'floor'
            if up_est is not None:
                p.report['floor_vs_arms_deg'] = round(float(np.degrees(np.arccos(np.clip(up @ up_est, -1, 1)))), 1)
        elif up_est is not None:
            up, src = up_est, 'arms'
        else:
            up, src = self._default_up(), 'config'
        p.up, p.up_source = up, src
        if 'shoulders' in rest:
            p.frame = frame_from(rest['shoulders'][1] - rest['shoulders'][0], up)
        # fingers
        if 'hand_open' in self.results and 'hand_fist' in self.results:
            o = np.mean([self.results['hand_open']['raw'][s] for s in self.sides], axis=0)
            c = np.mean([self.results['hand_fist']['raw'][s] for s in self.sides], axis=0)
            if np.all(np.sign(c - o) == np.sign(np.asarray(DEFAULT_CLOSED) - np.asarray(DEFAULT_OPEN))) \
                    and np.all(np.abs(c - o) > 0.2 * np.abs(np.asarray(DEFAULT_CLOSED) - np.asarray(DEFAULT_OPEN))):
                p.hand_open, p.hand_closed = o.tolist(), c.tolist()
            else:
                p.report['hand_range'] = 0.0   # the two hand poses were not distinct: kept defaults
        self.done = True

    def _default_up(self):
        t = np.radians(self.pcfg.camera_tilt_deg)
        return np.array([0.0, -np.cos(t), -np.sin(t)])
