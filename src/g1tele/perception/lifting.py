"""From a 2D landmark and a depth image to a 3D joint position (camera frame: x right, y down, z forward).

Two details matter for precision:
  * the depth image sees the SKIN in front of a joint, not the joint centre, so the point is moved
    along the viewing ray by the limb's radius;
  * a landmark on a limb edge samples background too: the depth of the NEAR part of the window
    (a low percentile) is used instead of the median.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1tele.perception.camera import Intrinsics


def robust_depth(depth: np.ndarray, u: float, v: float, window: int, percentile: float,
                 lo: float, hi: float) -> Optional[float]:
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    half = window // 2
    patch = depth[max(0, vi - half):vi + half + 1, max(0, ui - half):ui + half + 1]
    valid = patch[(patch > lo) & (patch < hi)]
    if valid.size < max(3, 0.15 * patch.size):
        return None
    return float(np.percentile(valid, percentile))


def pixel_ray(u: float, v: float, k: Intrinsics) -> np.ndarray:
    """Unit direction of the camera ray through pixel (u, v)."""
    d = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
    return d / np.linalg.norm(d)


def deproject(u: float, v: float, z: float, k: Intrinsics) -> np.ndarray:
    return np.array([(u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z])


def project(p: np.ndarray, k: Intrinsics):
    """Pixel (u, v) of a camera-frame point, or None if it is behind the camera."""
    if p[2] <= 1e-6:
        return None
    return k.fx * p[0] / p[2] + k.cx, k.fy * p[1] / p[2] + k.cy


def lift(u: float, v: float, depth: np.ndarray, k: Intrinsics, radius: float, window: int,
         percentile: float, lo: float, hi: float) -> Optional[np.ndarray]:
    """3D joint centre for the landmark at (u, v), or None if the depth there is unusable."""
    z = robust_depth(depth, u, v, window, percentile, lo, hi)
    if z is None:
        return None
    p = deproject(u, v, z, k)
    return p * (1.0 + radius / np.linalg.norm(p))


def ray_sphere(ray: np.ndarray, center: np.ndarray, radius: float, hint_t: Optional[float] = None):
    """Point on the camera ray (through the origin, direction `ray`) at `radius` from `center`.

    This is the bone-length constraint used as a depth sensor: when the 2D landmark is good but the
    depth is not (hole, background bleed, occlusion), the joint must still lie on the ray AND at
    bone length from its parent joint. Of the two intersections the one nearest `hint_t` (the
    measured or last known distance along the ray) is taken. Returns (point, exact) where exact is
    False if the ray misses the sphere and the closest point on the ray is returned instead.
    """
    b = float(ray @ center)
    disc = b * b - float(center @ center) + radius * radius
    if disc < 0:
        return ray * max(b, 0.05), False
    s = np.sqrt(disc)
    roots = [b - s, b + s]
    if hint_t is None:
        t = min(roots, key=lambda r: abs(r - b))
    else:
        t = min(roots, key=lambda r: abs(r - hint_t))
    return ray * max(t, 0.05), True


def depth_clusters(depth: np.ndarray, u: float, v: float, window: int, lo: float, hi: float,
                   gap: float = 0.035, min_frac: float = 0.12):
    """The distinct surfaces inside the window around a landmark, nearest first: [(depth, share), ...].

    At the edge of a hand in front of a wrist the window holds TWO surfaces (hand and wrist/forearm).
    A percentile of such a window flips between them from frame to frame; separate clusters can be tracked."""
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return []
    half = window // 2
    patch = depth[max(0, vi - half):vi + half + 1, max(0, ui - half):ui + half + 1]
    vals = np.sort(patch[(patch > lo) & (patch < hi)])
    if vals.size < max(3, 0.15 * patch.size):
        return []
    splits = np.where(np.diff(vals) > gap)[0] + 1
    out = []
    for part in np.split(vals, splits):
        if part.size >= max(3, min_frac * vals.size):
            out.append((float(np.median(part)), part.size / vals.size))
    return out
