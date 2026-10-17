"""Your hand as a model: MediaPipe's 2D landmarks in, a metric 3D hand out. Plain numpy + scipy (no MediaPipe, no ROS).

MediaPipe finds the landmarks in the image well; what it estimates badly is their depth, above all for the ones that
are hidden behind other parts of the hand (fingers pointing at the camera, a fist). So the depth is not read from
MediaPipe. It comes from the geometry of YOUR hand, measured once during the calibration (`HandShape`: stretched hand,
palm to the camera, nothing hidden), and every frame `HandFit` moves that hand (wrist pose + finger joint angles) until
its perspective projection lands on MediaPipe's 2D landmarks:

    perspective + known metric hand  ->  distance to the camera (from the apparent size of the parts of the hand)
    palm rigid (wrist + 4 knuckles + thumb base)  ->  palm orientation that does not depend on any finger
    finger joints with limits  ->  where the hidden joints and tips are, so wrist -> fingertip vectors stay valid

MediaPipe's own z is kept only as a weak tie-breaker (a flat palm looks the same tilted either way) and only for the
landmarks that are not hidden by another part of the hand (`visibility`).

Frames: the hand frame H of `frames.hand_rotation` (x fingers, z palm normal, wrist at the origin); camera axes
x right, y down, z away. Joint angles are positive when the finger bends towards the palm.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from dex_hand import frames

PALM = (0, 1, 5, 9, 13, 17)                       # wrist, thumb base, 4 knuckles: the rigid part
FINGER_IDX = np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])   # index .. pinky: knuckle .. tip
TIPS = np.array(frames.TIPS)
Z_AXIS = np.array([0.0, 0.0, 1.0])

# parameters: wrist rotation increment (3), wrist translation increment (3), finger abduction (4), knuckle / middle / end
# joint flexion (4 each), thumb base rotation vector (3), thumb knuckle and end flexion (1 each)
N_PARAMS = 27
I_ROT, I_POS, I_ABD, I_F0, I_F1, I_F2, I_THUMB = 0, 3, 6, 10, 14, 18, 22
LOWER = np.array([-np.inf] * 6 + [-0.45] * 4 + [-0.5] * 4 + [-0.1] * 4 + [-0.2] * 4 + [-1.4] * 3 + [-0.5, -0.6])
UPPER = np.array([np.inf] * 6 + [0.45] * 4 + [1.75] * 4 + [2.0] * 4 + [1.6] * 4 + [1.4] * 3 + [1.5, 1.6])

# what hides what (visibility): bones are capsules, the palm is four triangles
SEGMENTS = np.array([(a, b) for chain in list(FINGER_IDX) + [np.array([1, 2, 3, 4])] for a, b in zip(chain[:-1], chain[1:])])
TRIANGLES = np.array([(0, 5, 9), (0, 9, 13), (0, 13, 17), (0, 1, 5)])
BONE_RADIUS = 0.009                               # m
HIDING_DEPTH = 0.012                              # m: the hider must be this much closer to the camera

# noise model (pixels are for a 640 px wide image and scale with the image)
SIGMA_PX = 2.5                                    # MediaPipe's 2D error on a landmark
HIDDEN_PX_FACTOR = 2.5                            # a hidden landmark's 2D is trusted less, but still used: it is what tells the model about depth
SIGMA_Z = 0.02
SIGMA_DEPTH = 0.03                                # m, depth sensor at the wrist (it can land on the background when the wrist is hidden)
SIGMA_PRIOR = 0.6                                 # rad, joints stay close to last frame's value (a hidden finger holds its pose)
SIGMA_ROT = 0.15                                  # rad per frame: tilt of a flat palm is weak in 2D, this keeps its noise out; real turning is seen in the pixels and wins
SIGMA_MOVE = 0.03                                 # m per frame (distance is the weakly observed one)
SIGMA_COUPLING = 0.5                              # rad, the end joint bends about 0.7 times the middle one
Z_BAD = 0.035                                     # m: depth disagreement that suggests the model sits on the mirror-image solution
Z_BAD_FRAMES = 6                                  # for this many frames in a row: search for the pose again
BAD_PX = 12.0                                     # median reprojection error (640 px image) above which the fit is not trusted
MIN_VISIBLE = 6
MAX_DEPTH_JUMP = 0.25                             # m: a depth sensor reading this far from the tracked wrist is not the wrist


def _cube_rotations():
    """The 24 rotations that map the coordinate axes onto themselves: a coarse cover of all orientations."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = 1.0 - 2.0 * sign
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def skew(v):
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    k[..., 1, 2], k[..., 2, 0], k[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    return k


def rotation_about(axis, angle):
    """Rodrigues: rotations by `angle` (...) about unit `axis` (...,3)."""
    k = skew(axis)
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def rotation_from_vector(v):
    """exp of rotation vectors (...,3), fine at zero."""
    theta = np.linalg.norm(v, axis=-1)[..., None, None]
    k = skew(v)
    return np.eye(3) + np.sinc(theta / np.pi) * k + 0.5 * np.sinc(theta / (2 * np.pi)) ** 2 * (k @ k)


CUBE_ROTATIONS = _cube_rotations()


def unit_rows(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def kabsch(source, target):
    """Rotation R minimising |R source - target| for centred point sets (n,3)."""
    u, _, vt = np.linalg.svd(source.T @ target)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


class HandShape:
    """Your hand's geometry, measured at the calibration: `rest` is the stretched hand, 21x3 in the hand frame H (m)."""

    def __init__(self, rest, side, scale_source='mediapipe', mp_scale=1.0):
        self.rest = np.asarray(rest, dtype=float)
        self.side, self.scale_source, self.mp_scale = side, scale_source, float(mp_scale)

    @property
    def reach(self):
        """Wrist -> tip distances (m): thumb, index, middle, ring, pinky."""
        return np.linalg.norm(self.rest[TIPS], axis=1)

    @property
    def hand_length(self):
        return float(np.linalg.norm(self.rest[12]))

    def bend(self):
        """Largest angle (rad) between consecutive bones of a finger: how stretched the hand was."""
        worst = 0.0
        for chain in FINGER_IDX:
            b = np.diff(self.rest[chain], axis=0)
            b = unit_rows(b)
            worst = max(worst, *(float(np.arccos(np.clip(b[i] @ b[i + 1], -1, 1))) for i in range(2)))
        return worst

    @classmethod
    def from_samples(cls, side, worlds, pixels=None, depths=None, focal_px=None, cx=None, cy=None, hand_length=None):
        """From the calibration frames (stretched hand, palm to the camera): MediaPipe world landmarks, pixels, the
        wrist's sensor depths. The metric scale comes from, in order: the depth sensor (real pixels at a real distance),
        `hand_length` (wrist -> middle fingertip in m, measured with a ruler), else MediaPipe's own average-hand scale."""
        shapes = []
        for w in worlds:
            r = frames.hand_rotation(w, side)
            if r is not None:
                shapes.append((np.asarray(w, dtype=float) - w[0]) @ r)
        if not shapes:
            return None
        rest = np.median(shapes, axis=0)
        scale, source = 1.0, 'mediapipe'
        sensor = [] if depths is None else [i for i, d in enumerate(depths) if d is not None and i < len(worlds)]
        if sensor and pixels is not None:
            ratios = []
            for i in sensor:
                w, p, d = np.asarray(worlds[i], dtype=float), np.asarray(pixels[i], dtype=float), depths[i]
                z = d + (w[:, 2] - w[0, 2])
                metric = np.column_stack([(p[:, 0] - cx) * z / focal_px, (p[:, 1] - cy) * z / focal_px])
                num, den = np.sum((metric[1:] - metric[0]) ** 2), np.sum((w[1:, :2] - w[0, :2]) ** 2)
                if den > 1e-9:
                    ratios.append(np.sqrt(num / den))
            if ratios:
                scale, source = float(np.median(ratios)), 'depth sensor'
        elif hand_length:
            scale, source = float(hand_length) / max(float(np.linalg.norm(rest[12, :2])), 1e-6), 'hand length'
        rest = rest * scale
        flat = [0, 5, 9, 13, 17] + [int(i) for chain in FINGER_IDX for i in chain]     # a stretched hand is flat
        rest[flat, 2] = 0.0
        rest = rest @ frames.hand_rotation(rest, side)                                # exactly the hand frame of this shape
        return cls(rest, side, source, scale)


class HandModel:
    """Forward kinematics of the shape for a batch of parameter vectors."""

    def __init__(self, shape):
        self.shape, rest = shape, shape.rest
        self.mcp = rest[FINGER_IDX[:, 0]]
        self.bones = np.stack([rest[FINGER_IDX[:, j + 1]] - rest[FINGER_IDX[:, j]] for j in range(3)], axis=1)   # finger, bone, xyz
        d = rest[FINGER_IDX[:, 3]] - rest[FINGER_IDX[:, 0]]
        d[:, 2] = 0.0
        self.axis = unit_rows(np.cross(d, Z_AXIS))              # bending about it moves the finger towards the palm normal
        self.thumb_bones = np.diff(rest[[1, 2, 3, 4]], axis=0)
        t = self.thumb_bones.copy()
        t[:, 2] = 0.0
        self.thumb_axis = unit_rows(np.cross(t, Z_AXIS))
        self.rest_params = np.zeros(N_PARAMS)

    def hand_points(self, x):
        """x (B,N_PARAMS) -> landmarks in the hand frame (B,21,3)."""
        b = x.shape[0]
        p = np.empty((b, 21, 3))
        p[:, PALM] = self.shape.rest[list(PALM)]
        c = rotation_about(Z_AXIS, x[:, I_ABD:I_ABD + 4])                       # (B,4,3,3)
        pos = np.broadcast_to(self.mcp, (b, 4, 3))
        for j, start in enumerate((I_F0, I_F1, I_F2)):
            c = c @ rotation_about(self.axis[None], x[:, start:start + 4])
            pos = pos + np.einsum('bkij,kj->bki', c, self.bones[:, j])
            p[:, FINGER_IDX[:, j + 1]] = pos
        c = rotation_from_vector(x[:, I_THUMB:I_THUMB + 3])                     # (B,3,3)
        pos = np.broadcast_to(self.shape.rest[1], (b, 3))
        for j in range(3):
            if j:
                c = c @ rotation_about(self.thumb_axis[j], x[:, I_THUMB + 2 + j])
            pos = pos + np.einsum('bij,j->bi', c, self.thumb_bones[j])
            p[:, 2 + j] = pos
        return p


def visibility(points, focal_px):
    """Which landmarks are not hidden by another part of the hand. `points`: 21x3 camera coordinates (m)."""
    z = points[:, 2]
    uv = focal_px * points[:, :2] / np.maximum(z[:, None], 0.05)
    a, b = SEGMENTS[:, 0], SEGMENTS[:, 1]
    d = uv[b] - uv[a]
    t = np.clip(np.sum((uv[:, None] - uv[a][None]) * d[None], axis=-1) / (np.sum(d * d, axis=1)[None] + 1e-9), 0.0, 1.0)
    dist = np.linalg.norm(uv[:, None] - (uv[a][None] + t[..., None] * d[None]), axis=-1)
    z_seg = z[a][None] + t * (z[b] - z[a])[None]
    hides = (dist < focal_px * BONE_RADIUS / np.maximum(z_seg, 0.05)) & (z_seg < z[:, None] - HIDING_DEPTH)
    own = (np.arange(21)[:, None] == a[None]) | (np.arange(21)[:, None] == b[None])
    covered = np.any(hides & ~own, axis=1)
    v0, v1, v2 = (uv[TRIANGLES[:, k]] for k in range(3))
    e0, e1 = v1 - v0, v2 - v0
    den = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    w = uv[:, None] - v0[None]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (w[..., 0] * e1[:, 1] - w[..., 1] * e1[:, 0]) / den
        r = (e0[:, 0] * w[..., 1] - e0[:, 1] * w[..., 0]) / den
    inside = (s >= 0) & (r >= 0) & (s + r <= 1)
    z_tri = z[TRIANGLES[:, 0]] + s * (z[TRIANGLES[:, 1]] - z[TRIANGLES[:, 0]]) + r * (z[TRIANGLES[:, 2]] - z[TRIANGLES[:, 0]])
    vertex = np.zeros((21, len(TRIANGLES)), bool)
    for k in range(3):
        vertex[TRIANGLES[:, k], np.arange(len(TRIANGLES))] = True
    covered |= np.any(inside & (z_tri < z[:, None] - HIDING_DEPTH) & ~vertex, axis=1)
    return ~covered


@dataclass
class FitResult:
    rotation: np.ndarray            # hand frame H -> camera axes (columns x fingers, y, z palm normal)
    wrist: np.ndarray               # wrist position, camera axes (m)
    landmarks: np.ndarray           # 21x3 camera axes (m), metric
    hand_landmarks: np.ndarray      # 21x3 in the hand frame, wrist at the origin
    visible: np.ndarray             # 21 bools: not hidden by another part of the hand
    error_px: float                 # median distance between the model and MediaPipe's pixels (640 px image scale)
    ok: bool
    reinitialised: bool = False
    z_error: float = 0.0            # m, median disagreement of the visible landmarks' depth with MediaPipe's (relative to each other)
    cov_rot: Optional[np.ndarray] = None   # 3x3 (rad^2), camera axes: how well THIS frame's image fixes the palm rotation (large = weakly observed)
    cov_pos: Optional[np.ndarray] = None   # 3x3 (m^2), camera axes: the same for the wrist position (the depth axis is the weak one)

    @property
    def rot_sigma_deg(self):
        """Uncertainty of the palm rotation in its least-observed direction (degrees); inf when unknown."""
        return float('inf') if self.cov_rot is None else float(np.degrees(np.sqrt(np.linalg.eigvalsh(self.cov_rot)[-1])))

    @property
    def tip_vectors(self):
        """Wrist -> fingertip (5x3, hand frame): what the finger retargeting wants."""
        return self.hand_landmarks[TIPS]


class _Frame:
    """The data of one frame the residuals compare the model with."""

    def __init__(self, pixels, z_mp, visible, depth, focal, cx, cy, sigma_px, q_prev, prior_scale, r_ref, t_ref, use_z=True, pose_prior=0.0):
        self.pixels, self.focal, self.cx, self.cy = pixels, focal, cx, cy
        self.w_px = np.where(visible, 1.0, 1.0 / HIDDEN_PX_FACTOR) / sigma_px
        self.visible = visible
        self.w_z = visible / SIGMA_Z * (1.0 if use_z else 0.0)
        self.mean_mask = visible / max(int(visible.sum()), 1)
        self.z_mp = z_mp - np.sum(self.mean_mask * z_mp)
        # the sensor measures the wrist only if it can see it: a hidden wrist reads a finger in front or the wall behind
        self.depth = depth if depth is not None and visible[0] else None
        self.q_prev, self.prior_scale, self.pose_prior = q_prev, prior_scale, pose_prior
        self.r_ref, self.t_ref = r_ref, t_ref


class HandFit:
    """Tracks the calibrated hand through the frames. `update` returns a FitResult."""

    CURL_STARTS = (0.0, 0.6, 1.1)                    # first frame: open, half closed, closed

    def __init__(self, shape):
        self.shape, self.model = shape, HandModel(shape)
        self.reset()

    def reset(self):
        self.q = np.zeros(N_PARAMS - 6)
        self.r = self.t = self.points = None
        self._bad = self._age = self._z_bad = 0

    # -- the least-squares problem ---------------------------------------------------------------------------
    def _camera_points(self, x, r_ref, t_ref):
        p = self.model.hand_points(x)
        r = r_ref @ rotation_from_vector(x[:, I_ROT:I_ROT + 3])
        return np.einsum('bij,bnj->bni', r, p) + (t_ref + x[:, I_POS:I_POS + 3])[:, None]

    def _residuals(self, x, f):
        pts = self._camera_points(x, f.r_ref, f.t_ref)
        z = np.maximum(pts[..., 2], 0.05)
        uv = np.stack([f.cx + f.focal * pts[..., 0] / z, f.cy + f.focal * pts[..., 1] / z], axis=-1)
        r_px = ((uv - f.pixels[None]) * f.w_px[None, :, None]).reshape(len(x), -1)
        z_rel = pts[..., 2] - np.sum(f.mean_mask[None] * pts[..., 2], axis=1, keepdims=True)
        parts = [r_px, (z_rel - f.z_mp[None]) * f.w_z[None]]
        if f.depth is not None:
            parts.append(((pts[:, 0, 2] - f.depth) / SIGMA_DEPTH)[:, None])
        parts.append(x[:, I_ROT:I_ROT + 3] * (f.pose_prior / SIGMA_ROT))
        parts.append(x[:, I_POS:I_POS + 3] * (f.pose_prior / SIGMA_MOVE))
        parts.append((x[:, I_ABD:] - f.q_prev[None]) * (f.prior_scale / SIGMA_PRIOR))
        parts.append((x[:, I_F2:I_F2 + 4] - 0.7 * x[:, I_F1:I_F1 + 4]) / SIGMA_COUPLING)
        return np.concatenate(parts, axis=1)

    def _jacobian(self, x, f, h=1e-6):
        r = self._residuals(np.vstack([x[None], x[None] + h * np.eye(len(x))]), f)
        return ((r[1:] - r[0]) / h).T

    def _solve(self, f, nfev):
        x0 = np.clip(np.concatenate([np.zeros(6), f.q_prev]), LOWER + 1e-6, UPPER - 1e-6)
        return least_squares(lambda x: self._residuals(x[None], f)[0], x0, jac=lambda x: self._jacobian(x, f),
                             bounds=(LOWER, UPPER), loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=nfev, ftol=1e-6)

    def _frame(self, world, pixels, depth, focal, cx, cy, visible, q_prev, prior_scale, r_ref, t_ref, use_z=True, pose_prior=0.0):
        sigma = SIGMA_PX * (2.0 * cx / 640.0)
        return _Frame(pixels, world[:, 2] * self.shape.mp_scale, visible, depth, focal, cx, cy, sigma, q_prev, prior_scale,
                      r_ref, t_ref, use_z, pose_prior)

    def _cost(self, x, f):
        r = self._residuals(x[None], f)[0] / 1.5
        return float(0.5 * 1.5 ** 2 * np.sum(2.0 * (np.sqrt(1.0 + r * r) - 1.0)))

    @staticmethod
    def _advance(r, t, x):
        return r @ rotation_from_vector(x[None, I_ROT:I_ROT + 3])[0], t + x[I_POS:I_POS + 3], x[6:].copy()

    def _error_px(self, pts, pixels, focal, cx, cy):
        uv = np.column_stack([cx + focal * pts[:, 0] / pts[:, 2], cy + focal * pts[:, 1] / pts[:, 2]])
        return float(np.median(np.linalg.norm(uv - pixels, axis=1))) * 640.0 / (2.0 * cx)

    def _distance_from_bones(self, pixels, focal):
        """Distance of the hand from the camera (m) from how long its bones look. A bone looks shorter than it is when it
        leans towards or away from the camera, never longer, so the true distance is at most f * length / pixels for
        every bone, and the bones across the view give it tightly. Needs no depth from MediaPipe at all."""
        rest = self.shape.rest
        pairs = np.array([(a, b) for a, b in zip(range(0, 20), range(1, 21)) if b % 4 != 1] + [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17), (0, 9)])
        length = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1)
        pix = np.linalg.norm(pixels[pairs[:, 0]] - pixels[pairs[:, 1]], axis=1)
        usable = pix > 10.0
        z = focal * length[usable] / pix[usable] if usable.any() else np.array([0.6])
        return float(np.clip(np.min(z), 0.15, 4.0))

    # -- first frame: several starting hypotheses ------------------------------------------------------------------
    def _first(self, world, pixels, depth, focal, cx, cy):
        """The pose of the first frame, found without trusting MediaPipe's depth. A flat palm, or fingers pointing at the
        camera, fit the 2D landmarks about equally well from several orientations, so:
          1. many starting orientations (MediaPipe's palm and its mirror image, and the 24 axis-aligned ones) and open /
             closed fingers, each fitted to the 2D landmarks alone;
          2. the best few are refined, and each is scored again with MediaPipe's z of the landmarks that are visible IN
             THAT POSE (the depth of landmarks it hides is ignored), which tells the mirror images apart;
          3. the winner is polished with the z tie-breaker."""
        w = (world - world[0]) * self.shape.mp_scale
        idx = list(PALM)
        rest = self.shape.rest[idx] - self.shape.rest[idx].mean(0)
        z = self._distance_from_bones(pixels, focal)                  # an upper bound on the distance
        if depth is not None and depth > 1.2 * z:                     # farther than the hand can be: the background behind the wrist
            depth = None
        t0 = frames.pixel_to_camera(pixels[0][0], pixels[0][1], depth or z, focal, cx, cy)
        observed = w[idx] - w[idx].mean(0)
        starts = [kabsch(rest, observed), kabsch(rest, observed * np.array([1.0, 1.0, -1.0]))] + list(CUBE_ROTATIONS)
        everything = np.ones(21, bool)
        candidates = []
        for r0 in starts:
            for curl in (0.0, 1.0):
                q0 = np.zeros(N_PARAMS - 6)
                q0[I_F0 - 6:I_F0 - 2], q0[I_F1 - 6:I_F1 - 2], q0[I_F2 - 6:I_F2 - 2] = curl, curl, 0.8 * curl
                f = self._frame(world, pixels, None, focal, cx, cy, everything, q0, 0.0, r0, t0, use_z=False)
                result = self._solve(f, 8)
                candidates.append((result.cost, r0, q0, result.x))
        candidates.sort(key=lambda c: c[0])
        refined = []
        for _, r0, q0, x in candidates[:8]:
            r, t, q = self._advance(r0, t0, x)
            f = self._frame(world, pixels, None, focal, cx, cy, everything, q, 0.0, r, t, use_z=False)
            r, t, q = self._advance(r, t, self._solve(f, 60).x)
            pts = self._camera_points(np.concatenate([np.zeros(6), q])[None], r, t)[0]
            refined.append((self._error_px(pts, pixels, focal, cx, cy), self._z_error(pts, world, visibility(pts, focal)), r, t, q))
        limit = 1.5 * min(c[0] for c in refined) + 1.0                 # about as good a fit to the image ...
        best = min((c for c in refined if c[0] <= limit), key=lambda c: c[1])      # ... and the visible landmarks' depth agrees best
        _, _, r, t, q = best
        visible = visibility(self._camera_points(np.concatenate([np.zeros(6), q])[None], r, t)[0], focal)
        f = self._frame(world, pixels, depth, focal, cx, cy, visible, q, 0.3, r, t)
        return self._advance(r, t, self._solve(f, 40).x)

    # -- one frame -----------------------------------------------------------------------------------------------
    def update(self, world, pixels, depth, focal, cx, cy) -> FitResult:
        world, pixels = np.asarray(world, dtype=float), np.asarray(pixels, dtype=float)
        fresh = self.r is None
        if depth is not None and not fresh and abs(depth - self.t[2]) > MAX_DEPTH_JUMP:
            depth = None                                        # not the wrist (the background, a finger): ignore this reading
        if fresh:
            r, t, q = self._first(world, pixels, depth, focal, cx, cy)
        else:
            f = self._frame(world, pixels, depth, focal, cx, cy, visibility(self.points, focal), self.q, 1.0, self.r, self.t,
                            pose_prior=1.0)
            x = self._solve(f, 8 if self._age >= 3 else 30).x
            r, t, q = self.r @ rotation_from_vector(x[None, I_ROT:I_ROT + 3])[0], self.t + x[I_POS:I_POS + 3], x[6:]
        x = np.concatenate([np.zeros(6), q])[None]
        pts = self._camera_points(x, r, t)[0]
        error = self._error_px(pts, pixels, focal, cx, cy)
        visible = visibility(pts, focal)
        z_error = self._z_error(pts, world, visible)
        ok = error <= BAD_PX and int(visible.sum()) >= MIN_VISIBLE and bool(np.all(np.isfinite(pts)))
        if not ok:
            self._bad += 1
            if fresh or self._bad >= 3:                    # start again from MediaPipe's pose on the next frame
                self.reset()
            return FitResult(r, t, pts, self.model.hand_points(x)[0], visible, error, False, fresh, z_error)
        self._bad = 0
        self._age += 1
        self.r, self.t, self.q, self.points = r, t, q, pts
        self._z_bad = self._z_bad + 1 if z_error > Z_BAD else 0
        if self._z_bad >= Z_BAD_FRAMES:                       # the image fits but the depth never agrees: probably the mirror image
            self.reset()
        cov_rot, cov_pos = self.covariance(r, t, q, world, pixels, depth, focal, cx, cy, visible)
        return FitResult(r, t, pts, self.model.hand_points(x)[0], visible, error, True, fresh, z_error, cov_rot, cov_pos)

    COV_SCALE = 1.0                  # MediaPipe's 2D errors are not independent from landmark to landmark: inflates the covariance (set from the NEES check)

    def covariance(self, r, t, q, world, pixels, depth, focal, cx, cy, visible):
        """(cov_rot, cov_pos) of the pose in camera axes from the image evidence alone (Gauss-Newton, the pose priors left out; the
        fingers' joints are marginalised out). A flat palm facing the camera has a huge variance about the axes in its plane; a palm
        seen at an angle, or a hand with fingers spread in depth, does not."""
        f = self._frame(world, pixels, depth, focal, cx, cy, visible, q, 0.3, r, t, pose_prior=0.0)
        x = np.concatenate([np.zeros(6), q])
        res = self._residuals(x[None], f)[0]
        weight = 1.0 / np.sqrt(1.0 + (res / 1.5) ** 2)                 # soft-L1's weight: outlying residuals count less
        J = self._jacobian(x, f)
        H = J.T @ (weight[:, None] * J) + 1e-9 * np.eye(len(x))
        cov = np.linalg.inv(H)[:6, :6] * self.COV_SCALE
        return r @ cov[:3, :3] @ r.T, cov[3:, 3:]

    def _z_error(self, pts, world, visible):
        if visible.sum() < 3:
            return 0.0
        z_model, z_mp = pts[visible, 2], world[visible, 2] * self.shape.mp_scale
        return float(np.median(np.abs((z_model - z_model.mean()) - (z_mp - z_mp.mean()))))
