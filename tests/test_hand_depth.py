"""Palm orientation from the depth image, and landmarks hidden behind other things."""
import numpy as np

from g1tele.hand import hand_rotation
from g1tele.perception.camera import Intrinsics
from g1tele.perception.hand_depth import (facing_camera, fuse_palm_rotation, occluded_landmarks,
                                          palm_normal_from_depth)
from g1tele.perception.lifting import project
from g1tele.perception.synthetic import synthetic_hand_landmarks

K = Intrinsics(fx=600.0, fy=600.0, cx=424.0, cy=240.0, width=848, height=480)


def rot_about(axis, deg):
    a = np.radians(deg)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(a), np.sin(a)
    x = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return c * np.eye(3) + s * x + (1 - c) * np.outer(ax, ax)


def hand_scene(R, side='right', wrist=(0.1, 0.0, 1.8), curls=None, noise=0.002, seed=0):
    """Landmarks (camera axes, absolute) with a rendered palm plate in a depth image."""
    rng = np.random.default_rng(seed)
    pts = synthetic_hand_landmarks(R, np.zeros(6) if curls is None else curls, side) + np.asarray(wrist)
    image = np.array([project(p, K) for p in pts])
    depth = np.full((K.height, K.width), 5.0, np.float32)
    n_face = R[:, 2] if R[2, 2] < 0 else -R[:, 2]
    discs = []
    for x in np.arange(-0.03, 0.21, 0.002):                # palm and flat fingers: a hand silhouette
        for y in np.arange(-0.05, 0.0501, 0.002):
            p = np.asarray(wrist) + R[:, 0] * x + R[:, 1] * y + 0.01 * n_face
            u, v = project(p, K)
            discs.append((p[2], int(round(u)), int(round(v))))
    for z, u, v in sorted(discs, reverse=True):
        depth[v, u] = z
    depth[depth < 5.0] += rng.normal(0, noise, (depth < 5.0).sum()).astype(np.float32)
    return pts, image, depth


# palm facing the camera, tilted 30 degrees, for both hands and both faces
FACE = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # fingers right, palm towards the camera


def angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def test_facing_camera_from_the_2d_landmarks():
    rng = np.random.default_rng(0)
    for side in ('right', 'left'):
        for _ in range(40):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            pts = synthetic_hand_landmarks(q, np.zeros(6), side) + [0, 0, 1.8]
            image = np.array([project(p, K) for p in pts])
            n = hand_rotation(pts, side)[:, 2]
            face = facing_camera(image, side)
            if abs(n[2]) > 0.6:                          # clearly face-on or back-on
                assert (face > 0) == (n[2] < 0), (side, n, face)


def test_depth_gives_the_palm_normal_and_the_sign_from_the_2d_winding():
    errs = []
    for side in ('right', 'left'):
        for tilt, flip in [(0, False), (25, False), (45, False), (25, True), (45, True)]:
            R = rot_about([0, 1, 0], tilt) @ rot_about([1, 0, 0], 20) @ FACE
            if flip:                                     # the back of the hand towards the camera
                R = R @ np.diag([1.0, -1.0, -1.0])
            if side == 'left':
                R = hand_rotation(synthetic_hand_landmarks(R, np.zeros(6), 'left'), 'left')
            pts, image, depth = hand_scene(R, side)
            truth_n = hand_rotation(pts, side)[:, 2]
            got = palm_normal_from_depth(depth, K, image, side)
            assert got is not None, (side, tilt, flip)
            errs.append(angle(got.normal, truth_n))
            assert got.confidence > 0.4 and got.can_override
    assert max(errs) < 8.0, errs


def test_edge_on_palm_has_no_confident_normal():
    R = rot_about([0, 1, 0], 88) @ FACE
    pts, image, depth = hand_scene(R, 'right')
    got = palm_normal_from_depth(depth, K, image, 'right', landmark_normal=hand_rotation(pts, 'right')[:, 2])
    assert got is None or got.confidence < 0.3 or not got.can_override


def test_depth_repairs_a_flipped_landmark_normal():
    R = rot_about([0, 1, 0], 30) @ FACE
    pts, image, depth = hand_scene(R, 'right')
    flipped = pts * np.array([1.0, 1.0, -1.0])               # what the hand model does when it is confused
    r_lm = hand_rotation(flipped - flipped[0], 'right')
    n_true = hand_rotation(pts, 'right')[:, 2]
    assert angle(r_lm[:, 2], n_true) > 40                    # the landmark frame is wrong
    cue = palm_normal_from_depth(depth, K, image, 'right')
    r, source = fuse_palm_rotation(r_lm, cue, 0.7, 50.0)
    assert source == 'depth' and angle(r[:, 2], n_true) < 12
    assert abs(np.linalg.det(r) - 1) < 1e-6 and np.allclose(r.T @ r, np.eye(3), atol=1e-6)


def test_agreeing_landmarks_are_only_nudged():
    R = rot_about([0, 1, 0], 30) @ FACE
    pts, image, depth = hand_scene(R, 'right')
    r_lm = hand_rotation(pts - pts[0] + np.random.default_rng(1).normal(0, 0.004, pts.shape), 'right')
    cue = palm_normal_from_depth(depth, K, image, 'right', landmark_normal=r_lm[:, 2])
    r, source = fuse_palm_rotation(r_lm, cue, 0.7, 50.0)
    assert source == 'fused'
    n_true = hand_rotation(pts, 'right')[:, 2]
    assert angle(r[:, 2], n_true) <= angle(r_lm[:, 2], n_true) + 1.0


def test_occluders_are_found_but_the_hands_own_back_is_not():
    R = rot_about([0, 1, 0], 20) @ FACE
    pts, image, depth = hand_scene(R, 'right')
    plane = palm_normal_from_depth(depth, K, image, 'right').plane
    clean = occluded_landmarks(depth, image, pts, plane, K, 0.045)
    assert not clean.any()
    # an object 15 cm in front of the hand covers the index fingertip (and not its neighbour)
    u, v = image[8]
    blocked = depth.copy()
    blocked[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] = 1.65
    out = occluded_landmarks(blocked, image, pts, plane, K, 0.045)
    assert out[8] and not out[12] and out.sum() == 1
    # background behind a fingertip is not judged: thin fingers read background half the time
    far = depth.copy()
    far[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] = 4.0
    assert not occluded_landmarks(far, image, pts, plane, K, 0.045).any()
    # the knuckles and the wrist are never judged
    u, v = image[9]
    blocked[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] = 1.2
    assert not occluded_landmarks(blocked, image, pts, plane, K, 0.045)[9]
