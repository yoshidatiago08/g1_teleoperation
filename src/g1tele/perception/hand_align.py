"""Rigid alignment of the hand model to the depth image: the global hand orientation from measurement.

MediaPipe's world landmarks are the hand's shape in a frame whose orientation it guesses from one RGB crop.
The D455 measures the surface of the palm. Aligning the model's palm to those measured 3D points (a Kabsch /
Procrustes fit: the rotation that best maps one point set onto the other, the 3D-3D form of the PnP idea) gives a
rotation that comes from measurement, not from the network's guess. Two things make it robust:

  * Hundreds of correspondences, not six landmarks. Every depth pixel inside the palm polygon is paired with
    the model point at the same barycentric position of the polygon (the palm is nearly planar), so the
    depth noise averages out instead of tilting the hand.
  * The mirrored candidate. The classic failure of the hand model is a depth flip: the whole hand comes back
    mirrored along the viewing axis. The fit is done for the landmarks and for their z-mirror, and the better
    one (smaller residual) is kept.

`align_hand` returns the landmarks to use and the rotation that moves them onto the measurement, with a
confidence that falls with the residual, so a poor depth reading changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay

from g1tele.perception.camera import Intrinsics
from g1tele.perception.hand_depth import _palm_pixels

MIN_PAIRS = 40
# thumb base and the four knuckles. The wrist is left out on purpose: the forearm joins the palm there and its
# depth is not the palm's (it is thicker, and curved).
ALIGN_POLYGON = (1, 5, 9, 13, 17)
MIRROR = np.array([1.0, 1.0, -1.0])


@dataclass
class Alignment:
    world: np.ndarray           # the 21 landmarks to use (mirrored if the mirrored hand fits the depth better)
    R: np.ndarray               # rotation (camera axes) that takes `world` onto the measured palm
    rms: float                  # m, residual of the fit
    pairs: int
    mirrored: bool
    confidence: float           # 0..1


def kabsch(a: np.ndarray, b: np.ndarray):
    """Rotation R minimising sum |R a_i - b_i|^2 (both centred). Returns (R, rms)."""
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, float(np.sqrt(np.mean(np.sum((a @ r.T - b) ** 2, axis=1))))


def palm_pairs(depth: np.ndarray, k: Intrinsics, image_pts: np.ndarray, world: np.ndarray, lo: float, hi: float):
    """(model points, measured points): each depth pixel in the palm paired with the model point at the same
    barycentric position of the palm polygon. None if there are too few."""
    got = _palm_pixels(depth, image_pts, lo, hi, shrink=0.9, polygon=ALIGN_POLYGON)
    if got is None:
        return None
    us, vs, z = got
    keep = np.abs(z - np.median(z)) < 0.04                # one surface: not the background or a finger in front
    if keep.sum() < MIN_PAIRS:
        return None
    us, vs, z = us[keep], vs[keep], z[keep]
    # curled fingers lie over the knuckles and are nearer than the palm: keep only the pixels on the dominant plane
    pts = np.column_stack([(us - k.cx) * z / k.fx, (vs - k.cy) * z / k.fy, z])
    on = np.ones(len(pts), bool)
    for _ in range(4):
        c = pts[on].mean(axis=0)
        n = np.linalg.svd(pts[on] - c, full_matrices=False)[2][2]
        res = np.abs((pts - c) @ n)
        new_on = res < max(2.5 * 1.4826 * np.median(res[on]), 0.003)
        if new_on.sum() < MIN_PAIRS or (new_on == on).all():
            break
        on = new_on
    if on.sum() < MIN_PAIRS:
        return None
    us, vs, z = us[on], vs[on], z[on]
    poly_img = np.asarray(image_pts)[list(ALIGN_POLYGON)]
    try:
        tri = Delaunay(poly_img)
    except Exception:                                      # degenerate polygon (all points on a line)
        return None
    px = np.column_stack([us, vs]).astype(float)
    simplex = tri.find_simplex(px)
    ok = simplex >= 0
    if ok.sum() < MIN_PAIRS:
        return None
    T = tri.transform[simplex[ok]]
    bary2 = np.einsum('nij,nj->ni', T[:, :2, :], px[ok] - T[:, 2, :])
    bary = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
    verts = tri.simplices[simplex[ok]]
    model = np.einsum('ni,nij->nj', bary, np.asarray(world)[list(ALIGN_POLYGON)][verts])
    measured = np.column_stack([(us[ok] - k.cx) * z[ok] / k.fx, (vs[ok] - k.cy) * z[ok] / k.fy, z[ok]])
    return model, measured


def _fit(model: np.ndarray, measured: np.ndarray):
    """Trimmed Kabsch: drop the worst pairs once and fit again."""
    a, b = model - model.mean(axis=0), measured - measured.mean(axis=0)
    r, rms = kabsch(a, b)
    res = np.linalg.norm(a @ r.T - b, axis=1)
    keep = res < max(2.5 * 1.4826 * np.median(res), 0.004)
    if MIN_PAIRS <= keep.sum() < len(keep):
        a, b = model[keep] - model[keep].mean(axis=0), measured[keep] - measured[keep].mean(axis=0)
        r, rms = kabsch(a, b)
    return r, rms, int(keep.sum())


def align_hand(depth: np.ndarray, k: Intrinsics, image_pts: np.ndarray, world: np.ndarray,
               lo: float = 0.25, hi: float = 4.0, noise_m: float = 0.012,
               mirror_margin: float = 0.75) -> Optional[Alignment]:
    """Align the hand model `world` (21x3, camera axes) to the depth image. None if the depth gives too little.

    noise_m: the residual that counts as a typical good fit; confidence falls off beyond it."""
    pairs = palm_pairs(depth, k, image_pts, world, lo, hi)
    if pairs is None:
        return None
    model, measured = pairs
    r0, rms0, n0 = _fit(model, measured)
    # the mirrored hand: the same barycentric pairs, with the model flipped along the viewing axis
    model_m = model * MIRROR
    r1, rms1, n1 = _fit(model_m, measured)
    mirrored = rms1 < mirror_margin * rms0
    r, rms, n = (r1, rms1, n1) if mirrored else (r0, rms0, n0)
    used = np.asarray(world) * MIRROR if mirrored else np.asarray(world)
    conf = float(np.clip(n / 150.0, 0.0, 1.0) * np.exp(-(rms / noise_m) ** 2))
    return Alignment(used, r, rms, n, mirrored, conf)


def blend(r: np.ndarray, weight: float) -> np.ndarray:
    """The rotation `weight` of the way from the identity to r (along the geodesic)."""
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(cos))
    if ang < 1e-9 or weight <= 0.0:
        return np.eye(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:                                         # 180 deg: take the axis from the symmetric part
        w, v = np.linalg.eigh((r + r.T) / 2.0)
        axis = v[:, np.argmax(w)]
    else:
        axis = axis / nrm
    a = ang * weight
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1.0 - np.cos(a)) * (kx @ kx)


GOOD_FIT = 0.7             # confidence from which the measurement is believed even for a large correction


def aligned_landmarks(al: Alignment, weight: float, max_deg: float) -> np.ndarray:
    """The landmarks rotated `weight * confidence` of the way onto the measured palm.

    A correction larger than `max_deg` is usually the depth being wrong, so it is ignored, unless the fit is
    excellent (confidence >= GOOD_FIT: many pairs, tiny residual). Undoing a mirrored hand needs a large rotation,
    and that is exactly what clean close-range depth should be allowed to do."""
    ang = np.degrees(np.arccos(np.clip((np.trace(al.R) - 1.0) / 2.0, -1.0, 1.0)))
    if ang > max_deg and al.confidence < GOOD_FIT:
        return al.world
    return al.world @ blend(al.R, weight * al.confidence).T
