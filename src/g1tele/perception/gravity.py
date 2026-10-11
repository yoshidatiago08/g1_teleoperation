"""Find "up" in the camera frame from the scene itself, for cameras without an IMU.

The floor (or a desk) is a large horizontal plane. RANSAC on the lower part of the depth image
finds the biggest plane whose normal is near the expected up; its normal IS up, to within a degree,
independent of how the operator stands. Walls are excluded by the angle limit, the operator by
RANSAC (a body is not a plane).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from g1tele.perception.camera import Intrinsics


def estimate_up_from_floor(depth: np.ndarray, k: Intrinsics, up_guess: np.ndarray,
                           rng: Optional[np.random.Generator] = None, max_angle_deg: float = 35.0,
                           iterations: int = 200,
                           min_inlier_frac: float = 0.20) -> Optional[Tuple[np.ndarray, float]]:
    """Returns (unit up vector in the camera frame, inlier fraction) or None."""
    rng = rng or np.random.default_rng(0)
    h, w = depth.shape
    v0 = int(0.40 * h)
    vs, us = np.mgrid[v0:h:6, 0:w:6]
    z = depth[vs, us]
    ok = (z > 0.4) & (z < 3.5)       # beyond this the depth noise swamps the plane
    if ok.sum() < 200:
        return None
    z, u, v = z[ok], us[ok].astype(float), vs[ok].astype(float)
    pts = np.column_stack([(u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z])
    cos_lim = np.cos(np.radians(max_angle_deg))
    thr = 0.008 + 0.005 * z ** 2          # depth noise grows with the square of the distance
    best_n, best_d, best_count = None, 0.0, 0
    n_pts = len(pts)
    for _ in range(iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up_guess < 0:
            n = -n
        if n @ up_guess < cos_lim:
            continue
        d = n @ a
        count = int(np.sum(np.abs(pts @ n - d) < thr))
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None or best_count < min_inlier_frac * n_pts:
        return None
    # refine on the inliers
    inl = pts[np.abs(pts @ best_n - best_d) < thr]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[-1]
    if n @ up_guess < 0:
        n = -n
    return n / np.linalg.norm(n), best_count / n_pts
