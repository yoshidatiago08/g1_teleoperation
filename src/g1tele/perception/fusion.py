"""Kinematic depth fusion for one arm.

MediaPipe's 2D landmarks are good; the depth image at a landmark is good UNLESS something nearer
covers it (a forearm pointing at the camera hides the elbow; a hand hides the wrist; limb edges
read background). So the 3D arm is estimated, not read off:

    unknowns   a = distance of the elbow along its pixel ray,  b = same for the wrist
    evidence   depth readings (robust: a gross outlier is discounted instead of believed)
               |elbow - shoulder| = upper-arm length,  |wrist - elbow| = forearm length
               (weak) staying close to the previous frame

Solved per frame by a few robust Gauss-Newton steps from several starting points (the measured
depths, and both intersections of each ray with the bone sphere), keeping the cheapest solution.
When the depth is clean this returns the depth; when it is wrong, the bones decide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FusionParams:
    depth_sigma: float = 0.012       # m: expected error of a good depth reading (noise + radius error)
    bone_sigma: float = 0.012        # m: how rigid the bones are (calibration error, skin motion)
    robust_scale: float = 3.0        # a depth residual beyond this many sigmas is ignored (Tukey)
    select_sigma: float = 0.05       # m: scale of the wide loss that picks between candidate solutions
    temporal_sigma: float = 0.30     # m: weak prior toward the previous frame's depths (only breaks ties)
    iterations: int = 8


@dataclass
class FusionResult:
    elbow: Optional[np.ndarray]
    wrist: np.ndarray
    cost: float
    elbow_depth_weight: float = 1.0  # final robust weights: ~1 = depth believed, ~0 = depth rejected
    wrist_depth_weight: float = 1.0


def _ray_sphere_roots(ray: np.ndarray, center: np.ndarray, radius: float) -> List[float]:
    b = float(ray @ center)
    disc = b * b - float(center @ center) + radius * radius
    if disc < 0:
        return [max(b, 0.05)]       # the ray misses the sphere: closest approach
    s = np.sqrt(disc)
    return [r for r in (b - s, b + s) if r > 0.05] or [max(b, 0.05)]


def _tukey_weight(r: float, c: float) -> float:
    """IRLS weight of Tukey's biweight: 1 for a small residual, 0 beyond c (a gross outlier)."""
    u = r / c
    return (1.0 - u * u) ** 2 if abs(u) < 1.0 else 0.0


def _tukey_cost(r: float, c: float) -> float:
    u = min(abs(r) / c, 1.0)
    return (c * c / 6.0) * (1.0 - (1.0 - u * u) ** 3)


def _solve(a, b, rE, rW, S, Lu, Lf, a_m, b_m, a_prev, b_prev, prm: FusionParams):
    """Robust Gauss-Newton on (a, b). Returns (a, b, cost, weights)."""
    sd, sl, st = prm.depth_sigma, prm.bone_sigma, prm.temporal_sigma
    x = np.array([a, b], dtype=float)
    w1 = w2 = 1.0
    for _ in range(prm.iterations):
        E, W = x[0] * rE, x[1] * rW
        dES, dEW = E - S, W - E
        nES, nEW = np.linalg.norm(dES) + 1e-9, np.linalg.norm(dEW) + 1e-9
        uES, uEW = dES / nES, dEW / nEW
        res, jac = [], []
        # depth readings, robustified (Cauchy weights from the current residuals)
        if a_m is not None:
            r = (x[0] - a_m) / sd
            w1 = _tukey_weight(r, prm.robust_scale)
            res.append(np.sqrt(w1) * r), jac.append([np.sqrt(w1) / sd, 0.0])
        if b_m is not None:
            r = (x[1] - b_m) / sd
            w2 = _tukey_weight(r, prm.robust_scale)
            res.append(np.sqrt(w2) * r), jac.append([0.0, np.sqrt(w2) / sd])
        # bones
        res.append((nES - Lu) / sl), jac.append([float(uES @ rE) / sl, 0.0])
        res.append((nEW - Lf) / sl), jac.append([-float(uEW @ rE) / sl, float(uEW @ rW) / sl])
        # temporal prior
        if a_prev is not None:
            res.append((x[0] - a_prev) / st), jac.append([1.0 / st, 0.0])
        if b_prev is not None:
            res.append((x[1] - b_prev) / st), jac.append([0.0, 1.0 / st])
        r_vec, j = np.array(res), np.array(jac)
        step = np.linalg.solve(j.T @ j + 1e-6 * np.eye(2), -j.T @ r_vec)
        step = np.clip(step, -0.15, 0.15)
        x = np.maximum(x + step, 0.05)
        if np.max(np.abs(step)) < 1e-5:
            break
    # Score of this solution, used to choose between starting points (e.g. the near and far
    # intersection of a ray with a bone sphere). Depth readings are scored with a WIDE Cauchy loss:
    # a reading a few cm off (partial occlusion, a thick limb) is still far better evidence for
    # "this root" than the other root 30 cm away, whereas the tight Tukey loss above would score
    # both roots as "wrong" and pick arbitrarily.
    E, W = x[0] * rE, x[1] * rW
    cost = (((np.linalg.norm(E - S) - Lu) / sl) ** 2 + ((np.linalg.norm(W - E) - Lf) / sl) ** 2)
    ss = prm.select_sigma
    if a_m is not None:
        cost += np.log1p(((x[0] - a_m) / ss) ** 2) * 2.0
    if b_m is not None:
        cost += np.log1p(((x[1] - b_m) / ss) ** 2) * 2.0
    # The previous frame only decides between candidates for a joint WITHOUT a depth reading; a
    # joint with a reading must not be pulled to a stale root by its own history.
    if a_prev is not None and a_m is None:
        cost += ((x[0] - a_prev) / st) ** 2
    if b_prev is not None and b_m is None:
        cost += ((x[1] - b_prev) / st) ** 2
    return x[0], x[1], float(cost), (w1, w2)


def fuse_arm(S: np.ndarray, rE: Optional[np.ndarray], a_m: Optional[float],
             rW: np.ndarray, b_m: Optional[float], Lu: float, Lf: float,
             prev: Optional[Tuple[float, float]] = None,
             prm: Optional[FusionParams] = None) -> Optional[FusionResult]:
    """Elbow and wrist in the camera frame.

    S: shoulder (camera frame). rE, rW: unit pixel rays of the elbow and wrist landmarks (rE None =
    elbow not visible). a_m, b_m: measured distances along those rays (None = no usable depth).
    prev: (a, b) of the previous frame, used as a weak prior and to pick between ray/sphere roots.
    """
    prm = prm or FusionParams()
    if rE is None:
        # No elbow: all we have is the wrist ray and its depth; the arm cannot reach farther than
        # its bones, so an impossible distance is pulled back to the reach.
        if b_m is None:
            return None
        W = b_m * rW
        d = W - S
        n = np.linalg.norm(d)
        reach = Lu + Lf
        if n > 1.05 * reach:
            roots = _ray_sphere_roots(rW, S, reach)
            W = rW * min(roots, key=lambda r: abs(r - b_m))
        return FusionResult(None, W, 0.0, 1.0, 1.0)

    a_prev, b_prev = (None, None) if prev is None else prev
    a_starts = []
    if a_m is not None:
        a_starts.append(a_m)
    if a_prev is not None:
        a_starts.append(a_prev)
    a_starts += _ray_sphere_roots(rE, S, Lu)
    best: Optional[Tuple[float, float, float, tuple]] = None
    tried: List[Tuple[float, float]] = []
    for a0 in a_starts:
        E0 = a0 * rE
        b_starts = []
        if b_m is not None:
            b_starts.append(b_m)
        if b_prev is not None:
            b_starts.append(b_prev)
        b_starts += _ray_sphere_roots(rW, E0, Lf)
        for b0 in b_starts:
            if any(abs(a0 - ta) < 0.01 and abs(b0 - tb) < 0.01 for ta, tb in tried):
                continue
            tried.append((a0, b0))
            a, b, cost, w = _solve(a0, b0, rE, rW, S, Lu, Lf, a_m, b_m, a_prev, b_prev, prm)
            if best is None or cost < best[2]:
                best = (a, b, cost, w)
            # clean depth explaining everything well: no need to try the alternatives
            if len(tried) == 1 and cost < 1.5:
                break
        if best is not None and best[2] < 1.5:
            break
    if best is None:
        return None
    a, b, cost, (w1, w2) = best
    return FusionResult(a * rE, b * rW, cost, w1, w2)
