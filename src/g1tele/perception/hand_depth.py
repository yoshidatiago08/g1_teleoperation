"""What the depth image says about a hand, to correct what the single-image hand model guesses.

The hand model (MediaPipe) sees one RGB crop. Two things go wrong with it:
  * the palm normal flips or is noisy when the palm is edge-on, because its depth is inferred;
  * a landmark hidden behind another object is still reported, with a plausible but invented position.
The D455 measures both directly, so:

  palm_normal_from_depth   a plane fitted to the depth pixels inside the palm polygon gives the palm
                           normal; the 2D winding of the landmarks (which never flips) says whether
                           the palm or the back of the hand faces the camera
  occluded_landmarks       a finger landmark whose depth is well in front of where the hand is, and not on
                           the hand's own palm plane, is hidden by something else
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from g1tele.geometry import unit
from g1tele.perception.camera import Intrinsics

PALM_POLYGON = (0, 1, 5, 9, 13, 17)        # wrist, thumb base, then the four knuckles
MIN_PIXELS = 40


@dataclass
class PalmPlane:
    normal: np.ndarray          # unit, camera axes, pointing at the camera
    centroid: np.ndarray        # camera axes, m
    rms: float                  # m, residual of the inliers
    pixels: int
    confidence: float           # 0..1


def _palm_pixels(depth, image_pts, lo, hi, shrink=0.8):
    """Valid depth pixels inside the (slightly shrunk) palm polygon: (u, v, z) arrays."""
    poly = np.asarray(image_pts)[list(PALM_POLYGON)]
    c = poly.mean(axis=0)
    poly = c + (poly - c) * shrink                      # stay off the edge, where depth bleeds
    u0, v0 = np.floor(poly.min(axis=0)).astype(int) - 1
    u1, v1 = np.ceil(poly.max(axis=0)).astype(int) + 2
    h, w = depth.shape[:2]
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, w), min(v1, h)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return None
    mask = np.zeros((v1 - v0, u1 - u0), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(poly - [u0, v0]).astype(np.int32)), 1)
    patch = depth[v0:v1, u0:u1]
    ok = (mask > 0) & (patch > lo) & (patch < hi)
    vs, us = np.nonzero(ok)
    if vs.size < MIN_PIXELS:
        return None
    return us + u0, vs + v0, patch[vs, us].astype(float)


def fit_palm_plane(depth: np.ndarray, k: Intrinsics, image_pts: np.ndarray,
                   lo: float = 0.25, hi: float = 4.0) -> Optional[PalmPlane]:
    got = _palm_pixels(depth, image_pts, lo, hi)
    if got is None:
        return None
    us, vs, z = got
    keep = np.abs(z - np.median(z)) < 0.05              # one surface: drop background bleed and fingers in front
    if keep.sum() < MIN_PIXELS:
        return None
    us, vs, z = us[keep], vs[keep], z[keep]
    pts = np.column_stack([(us - k.cx) * z / k.fx, (vs - k.cy) * z / k.fy, z])
    inl = np.ones(len(pts), bool)
    for _ in range(3):                                  # trimmed least squares
        c = pts[inl].mean(axis=0)
        _, sv, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
        n = vt[2]
        res = (pts - c) @ n
        s = max(1.4826 * np.median(np.abs(res[inl] - np.median(res[inl]))), 0.002)
        new = np.abs(res) < 2.5 * s
        if new.sum() < MIN_PIXELS or (new == inl).all():
            break
        inl = new
    n_in = int(inl.sum())
    if n_in < MIN_PIXELS:
        return None
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    # the in-plane spread must be real in both directions: a sliver is an edge-on palm
    spread = sv[1] / np.sqrt(n_in)
    if n @ (-c) < 0:
        n = -n
    view = abs(float(n @ c)) / float(np.linalg.norm(c))          # 1 face-on, 0 edge-on: depth noise grows at grazing angles
    conf = float(np.clip(n_in / 150.0, 0, 1) * np.exp(-(rms / 0.008) ** 2) * np.clip(spread / 0.02, 0, 1)
                 * np.clip((view - 0.15) / 0.45, 0, 1))
    return PalmPlane(n, c, rms, n_in, conf)


def facing_camera(image_pts: np.ndarray, side: str) -> float:
    """Which face of the hand points at the camera, from the 2D landmarks alone (they never flip in depth).
    +1 palm, -1 back of the hand; the magnitude (0..1) is how sure: 0 is an edge-on hand.

    The sign is the winding of (pinky -> index knuckle) against (wrist -> middle knuckle), the same
    construction as g1tele.hand.hand_rotation with the depth set to zero: the palm normal points at the
    camera (negative z) when the palm faces it. The magnitude is how wide the knuckle line still looks
    compared with a face-on hand (about 0.65 of the palm length)."""
    p = np.asarray(image_pts, dtype=float)
    f = p[9] - p[0]
    length = float(np.linalg.norm(f))
    if length < 1e-6:
        return 0.0
    f = f / length
    t = p[5] - p[17]
    t_perp = t - (t @ f) * f
    cross_z = f[0] * t[1] - f[1] * t[0]          # z of f x t; the sign of the palm normal's z is the opposite for a right hand
    # hand_rotation: right n = t x f, left n = f x t  ->  n_z = -(f x t)_z for right, +(f x t)_z for left
    n_z = -cross_z if side == 'right' else cross_z
    sure = float(np.clip(np.linalg.norm(t_perp) / (0.65 * length), 0.0, 1.0))
    return sure if n_z < 0 else -sure


@dataclass
class PalmCue:
    normal: np.ndarray        # palm normal (out of the palm), camera axes
    confidence: float         # 0..1
    can_override: bool        # the 2D winding was decisive, so the depth may overrule a flipped landmark normal
    plane: PalmPlane


FACE_SURE = 0.5               # |facing_camera| above this: the winding of the 2D landmarks can be believed


def palm_normal_from_depth(depth: np.ndarray, k: Intrinsics, image_pts: np.ndarray, side: str,
                           lo: float = 0.25, hi: float = 4.0, prev_normal: Optional[np.ndarray] = None,
                           landmark_normal: Optional[np.ndarray] = None) -> Optional[PalmCue]:
    """The palm normal from the depth image, or None.

    The plane gives the normal up to its sign. Which face is visible comes from the 2D winding when the
    palm is clearly face-on or back-on (then the depth may overrule a flipped landmark normal). Near
    edge-on the winding is noise: the sign is taken from the landmark (or the previous) normal instead, so
    depth can sharpen the tilt but never flip the hand."""
    plane = fit_palm_plane(depth, k, image_pts, lo, hi)
    if plane is None:
        return None
    face = facing_camera(image_pts, side)
    if abs(face) >= FACE_SURE:
        return PalmCue((1.0 if face > 0 else -1.0) * plane.normal, plane.confidence, True, plane)
    ref = landmark_normal if landmark_normal is not None else prev_normal
    if ref is None:
        return None
    sign = 1.0 if ref @ plane.normal > 0 else -1.0
    return PalmCue(sign * plane.normal, plane.confidence * 0.7, False, plane)


def fuse_palm_rotation(r_lm: np.ndarray, cue: PalmCue, weight: float, flip_deg: float) -> Tuple[np.ndarray, str]:
    """Move the palm normal of the landmark frame `r_lm` (columns: fingers, y, normal) toward `n_depth`.

    Landmarks and depth within `flip_deg`: blend by `weight * conf`. Further apart: the landmarks are
    wrong (a mirrored hand is wrong in its finger axis too, not only its normal), so the depth normal is
    used as it is. Either way the finger axis is put back into the palm plane, which is where it lives."""
    f, n_lm = r_lm[:, 0], r_lm[:, 2]
    n_depth, conf = cue.normal, cue.confidence
    cos = float(np.clip(n_lm @ n_depth, -1, 1))
    if cue.can_override and conf >= 0.4 and np.degrees(np.arccos(cos)) > flip_deg:
        w, source = min(1.0, conf * 2.0), 'depth'
    else:
        w, source = weight * conf, 'fused'
    if w < 0.05:
        return r_lm, 'landmarks'
    n = unit((1 - w) * n_lm + w * n_depth)
    if n is None:
        return r_lm, 'landmarks'
    f = unit(f - (f @ n) * n)                       # fingers lie in the palm plane
    if f is None:
        return r_lm, 'landmarks'
    return np.column_stack([f, np.cross(n, f), n]), source


FINGER_LANDMARKS = (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)     # the joints that can be hidden


def occluded_landmarks(depth: np.ndarray, image_pts: np.ndarray, world: np.ndarray, plane: PalmPlane,
                       k: Intrinsics, front: float, lo: float = 0.25) -> np.ndarray:
    """Boolean 21: True where a finger landmark is hidden behind something that is not the hand.

    The hand model says how high each landmark is above the palm (`world`, camera axes); added to the
    measured palm plane that is where its depth should be. A reading much nearer is an occluder (the other
    hand, an object) unless it lies on the hand's own palm plane: the back of the hand hiding a curled finger
    is something the hand model knows about already. Readings much FARTHER are not used: at 2 m a finger
    is a few pixels wide and the depth there is background half the time.
    Only the finger joints are judged; the wrist and knuckles sit where the palm plane was measured."""
    h, w = depth.shape[:2]
    n, c = plane.normal, plane.centroid
    ref = np.asarray(world)[list(PALM_POLYGON)].mean(axis=0)
    height = (np.asarray(world) - ref) @ n                       # + towards the camera
    out = np.zeros(21, bool)
    for i in FINGER_LANDMARKS:
        u, v = image_pts[i]
        ui, vi = int(round(u)), int(round(v))
        if not (1 <= ui < w - 1 and 1 <= vi < h - 1):
            continue
        patch = depth[vi - 1:vi + 2, ui - 1:ui + 2]
        valid = patch[patch > lo]
        if valid.size < 3:
            continue                                  # no reading: nothing to judge by
        z = float(np.percentile(valid, 25))
        d = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
        denom = float(n @ d)
        if abs(denom) < 1e-3:
            continue
        z_plane = float(n @ c) / denom
        z_exp = z_plane + height[i] * n[2]
        out[i] = z < z_exp - front and abs(z - z_plane) > 0.025
    return out
