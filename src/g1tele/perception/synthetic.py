"""A synthetic operator and camera: ground truth for tests, benchmarks and camera-free demos.

SyntheticRig produces RGBD frames of a scripted operator (the calibration poses, then free arm
motion) with RealSense-like depth noise, dropouts, limb-edge background bleed and optional
occlusion windows, plus stand-in pose/hand estimators that return the (noisy) projected truth.
Because the truth is known, the metrics of `g1tele-bench` (accuracy, jitter, latency) are exact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from g1tele.geometry import frame_from, rot_x, unit
from g1tele.hand import DEFAULT_CLOSED, DEFAULT_OPEN
from g1tele.perception.camera import Intrinsics, RGBDFrame
from g1tele.perception.lifting import project
from g1tele.perception.pose_mediapipe import ELBOW, HAND_GUIDE, SHOULDER, WRIST, PoseResult

DOWN, FORWARD = np.array([0, 0, -1.0]), np.array([1.0, 0, 0])
UP_V = np.array([0, 0, 1.0])
KEYFRAMES = {   # (right upper-arm dir, right forearm dir); the left arm is mirrored in y
    'rest': (unit(np.array([0, -0.10, -1.0])), unit(np.array([0.1, -0.10, -1.0]))),
    'front': (FORWARD, FORWARD),
    'side': (np.array([0, -1.0, 0]), np.array([0, -1.0, 0])),
    'up': (UP_V, UP_V),
}


def slerp_dir(a, b, s):
    a, b = unit(a), unit(b)
    omega = np.arccos(np.clip(a @ b, -1, 1))
    if omega < 1e-6:
        return a
    return (np.sin((1 - s) * omega) * a + np.sin(s * omega) * b) / np.sin(omega)


def smoothstep(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def mirror(v):
    return np.array([v[0], -v[1], v[2]])


@dataclass
class Truth:
    t: float
    joints: Dict[str, Dict[str, np.ndarray]]    # side -> {'S','E','W'} in the body frame
    shoulders: Tuple[np.ndarray, np.ndarray]    # right, left, body frame
    hand_R: Dict[str, np.ndarray]               # body frame
    curls: Dict[str, np.ndarray]                # 6 values 0 open ... 1 closed
    occluded: Dict[str, bool]


class SyntheticOperator:
    """Scripted operator. Body frame: x forward, y left, z up (REP-103), origin mid-shoulders."""

    def __init__(self, upper=0.30, fore=0.27, shoulder_width=0.38, seed=0, free_after=None,
                 poses=('rest', 'front', 'side'), hand_poses=True, scenario='random', converge=False):
        self.converge = converge         # table scenario: the two hands move together and apart
        self.scenario = scenario         # free motion after the calibration: 'random' arm waypoints, or 'table'
        self.upper, self.fore, self.width = upper, fore, shoulder_width
        self.rng = np.random.default_rng(seed)
        # calibration choreography: (pose, seconds); hands: ('hand_open'|'hand_fist')
        self.script = [(p, 6.0 if p == 'rest' else 5.0) for p in poses]
        if hand_poses:
            self.script += [('hand_open', 5.0), ('hand_fist', 5.0)]
        self.script += [('rest', 3.0)]
        self.free_start = sum(d for _, d in self.script) if free_after is None else free_after
        self.transition = 1.4
        self.occlusions: List[Tuple[str, float, float]] = []   # (side, t0, t1) wrist hidden
        # free-motion waypoints, generated once
        self.waypoints = [self._random_arm() for _ in range(200)]

    def _random_arm(self):
        u = unit(np.array([self.rng.uniform(-0.2, 0.7), self.rng.uniform(-0.9, 0.1), self.rng.uniform(-1.0, 0.3)]))
        f = unit(u + np.array([self.rng.uniform(0.2, 1.0), self.rng.uniform(-0.6, 0.3), self.rng.uniform(-0.8, 0.5)]))
        return u, f

    def _segment(self, t):
        acc = 0.0
        for i, (name, dur) in enumerate(self.script):
            if t < acc + dur:
                return i, t - acc
            acc += dur
        return None, t - acc

    def _pose_dirs(self, name):
        return KEYFRAMES['rest'] if name.startswith('hand') else KEYFRAMES[name]

    def arm_dirs(self, side: str, t: float):
        i, local = self._segment(t)
        if i is not None:
            name = self.script[i][0]
            prev = self.script[i - 1][0] if i > 0 else 'rest'
            a_u, a_f = self._pose_dirs(prev)
            b_u, b_f = self._pose_dirs(name)
            s = smoothstep(local / self.transition)
            u, f = slerp_dir(a_u, b_u, s), slerp_dir(a_f, b_f, s)
        elif self.scenario == 'table':
            u, f = self._table_dirs(side, t)
            return u, f
        else:
            tf = t - self.free_start
            k, frac = int(tf // 1.6), (tf % 1.6) / 1.6
            (u0, f0), (u1, f1) = self.waypoints[k % 200], self.waypoints[(k + 1) % 200]
            s = smoothstep(frac)
            u, f = slerp_dir(u0, u1, s), slerp_dir(f0, f1, s)
            if tf < self.transition:          # leave the last calibration pose smoothly
                r_u, r_f = KEYFRAMES['rest']
                m = smoothstep(tf / self.transition)
                u, f = slerp_dir(r_u, u, m), slerp_dir(r_f, f, m)
        if side == 'left':
            u, f = mirror(u), mirror(f)
            if i is None:   # free motion: left arm follows an independent waypoint set
                tf = t - self.free_start
                k, frac = int(tf // 1.6) + 7, (tf % 1.6) / 1.6
                (u0, f0), (u1, f1) = self.waypoints[k % 200], self.waypoints[(k + 1) % 200]
                s = smoothstep(frac)
                u, f = slerp_dir(u0, u1, s), slerp_dir(f0, f1, s)
                if tf < self.transition:
                    r_u, r_f = KEYFRAMES['rest']
                    m = smoothstep(tf / self.transition)
                    u, f = slerp_dir(r_u, u, m), slerp_dir(r_f, f, m)
                u, f = mirror(u), mirror(f)
        return u, f

    def _table_dirs(self, side, t):
        """Both hands reaching forward and down to a table, as when manipulating an object: the forearms
        point at the camera (the case where one limb hides another), with small circular motions."""
        tf = max(t - self.free_start, 0.0)
        ramp = smoothstep(tf / 2.5)                    # reach from the rest pose
        ph = 0.0 if side == 'right' else 1.7
        wob = np.array([0.20 * np.sin(2 * np.pi * 0.45 * tf + ph), 0.18 * np.cos(2 * np.pi * 0.35 * tf + ph),
                        0.12 * np.sin(2 * np.pi * 0.55 * tf + 2 * ph)]) * ramp
        sgn = -1.0 if side == 'right' else 1.0
        kappa = 0.5 * (1 - np.cos(2 * np.pi * 0.08 * tf)) * ramp if self.converge else 0.0   # the hands come together
        u_t = unit(np.array([0.65, sgn * (0.25 - 0.55 * kappa), -0.70]) + wob * 0.5)
        f_t = unit(np.array([0.95, sgn * (0.06 - 0.60 * kappa), -0.28]) + wob)
        r_u, r_f = KEYFRAMES['rest']
        if side == 'left':
            r_u, r_f = mirror(r_u), mirror(r_f)
        return slerp_dir(r_u, u_t, ramp), slerp_dir(r_f, f_t, ramp)

    def curls(self, t):
        i, local = self._segment(t)
        name = self.script[i][0] if i is not None else 'free'
        prev = self.script[i - 1][0] if (i is not None and i > 0) else 'rest'
        def level(n):
            return 1.0 if n == 'hand_fist' else 0.0
        if name == 'free':
            tf = t - self.free_start
            wave = 0.5 + 0.5 * np.sin(tf * 1.3 + np.arange(6) * 0.7)
            return np.clip(wave, 0, 1)
        s = smoothstep(local / 0.8)
        c = level(prev) + (level(name) - level(prev)) * s
        return np.full(6, c)

    def hand_roll(self, side, t):
        return 0.5 * np.sin(0.7 * t) + (0.0 if side == 'right' else 0.3)

    def at(self, t: float) -> Truth:
        S = {'right': np.array([0, -self.width / 2, 0.0]), 'left': np.array([0, self.width / 2, 0.0])}
        sway = np.array([0.01 * np.sin(0.5 * t), 0.01 * np.sin(0.37 * t), 0.004 * np.sin(0.8 * t)])
        joints, hand_R, curls, occ = {}, {}, {}, {}
        for side in ('right', 'left'):
            u, f = self.arm_dirs(side, t)
            s0 = S[side] + sway
            e = s0 + self.upper * u
            w = e + self.fore * f
            joints[side] = {'S': s0, 'E': e, 'W': w}
            # hand frame: x along the forearm, palm normal turned by the roll about it
            n_ref = np.array([0, 1.0, 0]) if side == 'right' else np.array([0, -1.0, 0])
            n = n_ref - (n_ref @ f) * f
            n = unit(n) if np.linalg.norm(n) > 1e-3 else unit(np.array([0, 0, 1.0]) - (UP_V @ f) * f)
            roll = self.hand_roll(side, t)
            n = np.cos(roll) * n + np.sin(roll) * np.cross(f, n)
            hand_R[side] = np.column_stack([f, np.cross(n, f), n])
            curls[side] = self.curls(t)
            occ[side] = any(s == side and a <= t < b for s, a, b in self.occlusions)
        return Truth(t, joints, (S['right'] + sway, S['left'] + sway), hand_R, curls, occ)


def synthetic_hand_landmarks(R_cam: np.ndarray, curls: np.ndarray, side: str) -> np.ndarray:
    """21 MediaPipe-layout landmarks (metres, camera axes, origin at the wrist) of a hand with the
    given orientation (columns x fingers, y, z palm normal) and curls in Inspire order."""
    s_idx = -1.0 if side == 'right' else 1.0       # index finger side in the hand's y axis
    mcp = {'index': (0.095, 0.032 * s_idx), 'middle': (0.098, 0.010 * s_idx),
           'ring': (0.093, -0.011 * s_idx), 'pinky': (0.083, -0.032 * s_idx)}
    seg = {'index': (0.040, 0.025, 0.022), 'middle': (0.044, 0.028, 0.024),
           'ring': (0.041, 0.026, 0.023), 'pinky': (0.032, 0.020, 0.020)}
    base = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
    channel = {'pinky': 0, 'ring': 1, 'middle': 2, 'index': 3}
    pts = np.zeros((21, 3))
    for name, (x, y) in mcp.items():
        c = float(np.clip(curls[channel[name]], 0, 1))
        total = DEFAULT_OPEN[channel[name]] + c * (DEFAULT_CLOSED[channel[name]] - DEFAULT_OPEN[channel[name]])
        angles = np.cumsum(np.array([0.35, 0.40, 0.25]) * total)
        p = np.array([x, y, 0.0])
        pts[base[name]] = p
        for k in range(3):
            d = np.array([np.cos(angles[k]), 0.0, np.sin(angles[k])])
            p = p + seg[name][k] * d
            pts[base[name] + 1 + k] = p
    # thumb: bend (two equal joints) and a base direction chosen to give the wanted tip distance
    bend_raw = DEFAULT_OPEN[4] + float(np.clip(curls[4], 0, 1)) * (DEFAULT_CLOSED[4] - DEFAULT_OPEN[4])
    ratio = DEFAULT_OPEN[5] + float(np.clip(curls[5], 0, 1)) * (DEFAULT_CLOSED[5] - DEFAULT_OPEN[5])
    palm = np.linalg.norm(pts[9])
    cmc = np.array([0.030, 0.022 * s_idx, 0.0])
    best, best_err = None, 1e9
    for beta in np.linspace(-0.4, 2.4, 141):
        d0 = np.array([np.cos(beta) * 1.0, np.sin(beta) * s_idx, 0.0])   # in the palm plane
        a1 = bend_raw / 2
        p1 = cmc + 0.040 * d0
        d1 = np.array([d0[0] * np.cos(a1), d0[1] * np.cos(a1), np.sin(a1)])
        p2 = p1 + 0.032 * d1
        a2 = bend_raw
        d2 = np.array([d0[0] * np.cos(a2), d0[1] * np.cos(a2), np.sin(a2)])
        p3 = p2 + 0.028 * d2
        err = abs(np.linalg.norm(p3 - pts[17]) / palm - ratio)
        if err < best_err:
            best, best_err = (p1, p2, p3), err
    pts[1], (pts[2], pts[3], pts[4]) = cmc, best
    return pts @ R_cam.T       # hand frame -> camera axes (columns of R_cam are the hand axes)


class SyntheticRig:
    """Camera + estimators + truth for one scripted operator."""

    finite = False

    def __init__(self, operator: Optional[SyntheticOperator] = None, width=848, height=480, fps=30.0,
                 distance=1.8, tilt_deg=0.0, pixel_noise=1.2, depth_noise=0.002, dropout=0.05,
                 edge_bleed=0.10, seed=1, gravity=False, realtime=False, hand_noise=0.003, radius_scale=1.0, floor_below=1.4, wall=5.0, erratic=False, hand_glitch=False, palm_plate=True, hand_flip_rate=0.0):
        self.op = operator or SyntheticOperator()
        self.k = Intrinsics(fx=0.78 * width * 1.0, fy=0.78 * width * 1.0, cx=width / 2, cy=height / 2,
                            width=width, height=height)
        self.fps, self.dt = fps, 1.0 / fps
        self.rng = np.random.default_rng(seed)
        self.pixel_noise, self.depth_noise, self.dropout = pixel_noise, depth_noise, dropout
        self.edge_bleed, self.hand_noise = edge_bleed, hand_noise
        self.radius_scale = radius_scale
        self.floor_below = floor_below
        self.hand_glitch = hand_glitch
        self.palm_plate = palm_plate
        self.hand_flip_rate = hand_flip_rate          # share of readings that come back mirrored, at any orientation
        self.erratic = erratic            # occluded joints get noisy, flickering landmarks (like MediaPipe)
        self.wall = wall
        self._rays = None
        self.tilt = np.radians(tilt_deg)
        self.R_face = np.column_stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])   # body axes in camera coords
        self.R_cb = rot_x(self.tilt) @ self.R_face
        self.origin = rot_x(self.tilt) @ np.array([0.05, -0.15, distance])
        self.gravity = gravity
        self.realtime = realtime
        self.index = 0
        self.t0_wall = time.time()
        self.truth: Optional[Truth] = None
        self.pose = _FakePose(self)
        self.hands = _FakeHands(self)

    # ------------------------------------------------------------------ geometry
    def to_cam(self, p_body: np.ndarray) -> np.ndarray:
        return self.origin + self.R_cb @ p_body

    @property
    def up_cam(self) -> np.ndarray:
        return self.R_cb[:, 2]

    def _background(self) -> np.ndarray:
        """A wall 4 m away and (optionally) a floor plane `floor_below` m under the shoulders."""
        k = self.k
        if self._rays is None:
            vs, us = np.mgrid[0:k.height, 0:k.width]
            r = np.stack([(us - k.cx) / k.fx, (vs - k.cy) / k.fy, np.ones_like(us, dtype=float)], axis=-1)
            self._rays = (r / np.linalg.norm(r, axis=-1, keepdims=True)).astype(np.float32)
        depth = np.full((k.height, k.width), self.wall, dtype=np.float32)
        if self.floor_below is not None:
            up = self.up_cam
            plane_pt = self.origin - up * self.floor_below
            denom = self._rays @ (-up).astype(np.float32)          # ray . (down)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (plane_pt @ -up) / denom
            hit = (denom > 1e-3) & (t > 0) & (t * self._rays[..., 2] < self.wall)
            # range along the ray -> z depth
            depth = np.where(hit, t * self._rays[..., 2], depth).astype(np.float32)
        return depth

    def _draw_hand(self, depth, layer, side, truth):
        """The palm as a thin flat plate in the hand's own plane (what makes its orientation visible in depth)
        and the fingers as thin lines along the bones of the posed hand (so curled fingers are in front of, or
        hidden behind, the palm like real ones). Composited by "nearest wins"."""
        k = self.k
        R = self.R_cb @ truth.hand_R[side]
        w = self.to_cam(truth.joints[side]['W'])
        n = R[:, 2]
        n_face_z = n[2] if n[2] < 0 else -n[2]
        corners = [w + R[:, 0] * x + R[:, 1] * y for x, y in ((-0.03, -0.045), (0.095, -0.045), (0.095, 0.045), (-0.03, 0.045))]
        uv = [project(c, k) for c in corners]
        if any(c is None for c in uv):
            return
        poly = np.round(np.array(uv)).astype(np.int32)
        u0, v0 = np.maximum(poly.min(axis=0) - 1, 0)
        u1, v1 = np.minimum(poly.max(axis=0) + 2, [k.width, k.height])
        if u1 - u0 > 2 and v1 - v0 > 2:
            mask = np.zeros((v1 - v0, u1 - u0), np.uint8)
            cv2.fillConvexPoly(mask, poly - [u0, v0], 1)
            vs, us = np.nonzero(mask)
            d = np.column_stack([(us + u0 - k.cx) / k.fx, (vs + v0 - k.cy) / k.fy, np.ones(len(us))])
            denom = d @ n
            ok = np.abs(denom) > 1e-3
            z = (n @ w) / np.where(ok, denom, 1.0) + 0.010 * n_face_z
            ok &= z > 0.1
            patch = depth[v0:v1, u0:u1]
            patch[vs[ok], us[ok]] = np.minimum(patch[vs[ok], us[ok]], z[ok]).astype(np.float32)
        lm = synthetic_hand_landmarks(R, truth.curls[side], side) + w
        segs = []
        for base in (1, 5, 9, 13, 17):
            for a, b in zip(range(base, base + 3), range(base + 1, base + 4)):
                pa, pb = project(lm[a], k), project(lm[b], k)
                if pa is not None and pb is not None:
                    segs.append((0.5 * (lm[a][2] + lm[b][2]) - 0.008, pa, pb, max(1, int(round(k.fx * 0.016 / lm[a][2])))))
        for z, pa, pb, thick in sorted(segs, key=lambda sg: -sg[0]):
            cv2.line(layer, (int(round(pa[0])), int(round(pa[1]))), (int(round(pb[0])), int(round(pb[1]))), float(z), thick)

    def _render_depth(self, truth: Truth) -> np.ndarray:
        k = self.k
        depth = self._background()
        discs = []     # (z_surface, u, v, r_px)
        def add_capsule(a, b, r, n=14):
            for s in np.linspace(0, 1, n):
                p = a + (b - a) * s
                uv = project(p, k)
                if uv is None:
                    continue
                zs = p[2] - r
                discs.append((zs, uv[0], uv[1], max(1.0, k.fx * r / p[2])))
        for side in ('right', 'left'):
            j = {n: self.to_cam(p) for n, p in truth.joints[side].items()}
            # radii match PerceptionCfg.radii by default (elbow 4 cm, wrist 3 cm, shoulder 5.5 cm);
            # `radius_scale` makes the world deviate from what the pipeline assumes
            add_capsule(j['S'], j['E'], 0.040 * self.radius_scale)
            add_capsule(j['E'], j['W'], 0.030 * self.radius_scale)
            w, e = j['W'], j['E']
            hand_end = w + unit(w - e) * 0.15
            if not self.palm_plate:                          # with a palm plate the hand is drawn below, not as a cylinder
                add_capsule(w, hand_end, 0.03 * self.radius_scale, n=6)
        # torso: a slab between the shoulders and 0.5 m down (drawn as discs on a grid)
        r_sh, l_sh = (self.to_cam(p) for p in truth.shoulders)
        for a in np.linspace(0, 1, 12):
            for b in np.linspace(0, 1, 6):
                p = r_sh + (l_sh - r_sh) * a - self.R_cb[:, 2] * 0.5 * b
                uv = project(p, k)
                if uv is not None:
                    discs.append((p[2] - 0.055 * self.radius_scale, uv[0], uv[1], max(1.0, k.fx * 0.06 / p[2])))
        for zs, u, v, r in sorted(discs, key=lambda d: -d[0]):      # far to near: near wins
            cv2.circle(depth, (int(round(u)), int(round(v))), int(round(r)), float(zs), -1)
        if self.palm_plate:
            layer = np.full(depth.shape, np.inf, np.float32)         # fingers of both hands, nearest wins
            for side in ('right', 'left'):
                self._draw_hand(depth, layer, side, truth)
            np.minimum(depth, layer, out=depth)
        sigma = self.depth_noise * depth ** 2
        depth = depth + self.rng.normal(0, 1, depth.shape).astype(np.float32) * sigma
        body = depth < self.wall - 0.1
        # limb-edge bleed: a share of the pixels on the silhouette read the background
        edge = body & ~cv2.erode(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bleed = edge & (self.rng.random(depth.shape) < self.edge_bleed * 4)
        depth[bleed] = self.wall
        depth[self.rng.random(depth.shape) < self.dropout] = 0.0
        return depth

    # ------------------------------------------------------------------ source interface
    def read(self, timeout_ms: int = 0) -> Optional[RGBDFrame]:
        t = self.index * self.dt
        if self.realtime:
            delay = self.t0_wall + t - time.time()
            if delay > 0:
                time.sleep(delay)
        self.truth = self.op.at(t)
        depth = self._render_depth(self.truth)
        color = np.zeros((self.k.height, self.k.width, 3), np.uint8)
        self.index += 1
        t_cap = (self.t0_wall + t) if self.realtime else t
        grav = -self.up_cam if self.gravity else None
        return RGBDFrame(color, depth, self.k, t_cap, self.index, grav)

    def close(self):
        pass


class _FakePose:
    def __init__(self, rig: SyntheticRig):
        self.rig = rig

    def process(self, rgb) -> Optional[PoseResult]:
        rig = self.rig
        truth = rig.truth
        uv = np.zeros((33, 2))
        vis = np.zeros(33)
        for side in ('right', 'left'):
            j_cam = {k: rig.to_cam(v) for k, v in truth.joints[side].items()}
            fore_dir = unit(j_cam['W'] - j_cam['E'])
            ray_e = unit(j_cam['E'])
            # a forearm pointing at the camera hides the elbow: MediaPipe then guesses (noisy, low and
            # erratic visibility), which is what real data looks like
            end_on_elbow = abs(float(fore_dir @ ray_e))
            for key, idx in (('S', SHOULDER[side]), ('E', ELBOW[side]), ('W', WRIST[side])):
                pc = j_cam[key]
                p = project(pc, rig.k)
                noise = rig.pixel_noise
                erratic = rig.erratic and key == 'E' and end_on_elbow > 0.8
                if erratic:
                    noise *= 3.0
                uv[idx] = np.array(p) + rig.rng.normal(0, noise, 2)
                inside = 0 <= uv[idx][0] < rig.k.width and 0 <= uv[idx][1] < rig.k.height
                vis[idx] = 0.1 if (not inside or (key == 'W' and truth.occluded[side])) else 0.99
                if erratic and inside:
                    vis[idx] = float(rig.rng.uniform(0.3, 1.0))
            # hand guide points sit around the wrist
            for g in HAND_GUIDE[side][1:]:
                uv[g] = uv[WRIST[side]] + rig.rng.normal(0, 4.0, 2)
                vis[g] = vis[WRIST[side]]
        return PoseResult(uv, vis)


class _FakeHands:
    """Stand-in for MediaPipe Hands on a crop. With rig.hand_glitch it misbehaves like the real thing near the
    other hand and end on: it can return the other hand, and a flipped (mirrored) palm."""

    def __init__(self, rig: SyntheticRig):
        self.rig = rig

    def _hand(self, side, truth, rig):
        R_cam = rig.R_cb @ truth.hand_R[side]
        world = synthetic_hand_landmarks(R_cam, truth.curls[side], side)
        world = world + rig.rng.normal(0, rig.hand_noise, world.shape)
        w_cam = rig.to_cam(truth.joints[side]['W'])
        w_px = np.array(project(w_cam, rig.k))
        ray = w_cam / np.linalg.norm(w_cam)
        if rig.hand_glitch and abs(float(R_cam[:, 0] @ ray)) > 0.8 and rig.rng.random() < 0.15:
            world = world * np.array([1.0, 1.0, -1.0])           # depth-ambiguous: the palm comes out flipped
        if rig.hand_flip_rate and rig.rng.random() < rig.hand_flip_rate:
            world = world * np.array([1.0, 1.0, -1.0])
        image = w_px + world[:, :2] * rig.k.fx / w_cam[2]
        return world, image, side

    def detect_all(self, rgb, center_px, hand_px):
        rig, truth = self.rig, self.rig.truth
        radius = 1.1 * 2.2 * hand_px                              # what the crop covers
        found = []
        for side in ('right', 'left'):
            p = np.array(project(rig.to_cam(truth.joints[side]['W']), rig.k))
            if np.hypot(*(p - center_px)) < radius and not truth.occluded[side]:
                found.append(self._hand(side, truth, rig))
        if rig.hand_glitch and len(found) > 1 and rig.rng.random() < 0.5:
            found = found[::-1]                                   # the model returns them in no particular order
        return [(w, i) for w, i, _ in found]

    def detect(self, rgb, center_px, hand_px):
        """The old behaviour: whatever the model returned first."""
        found = self.detect_all(rgb, center_px, hand_px)
        return found[0] if found else None
