"""Aligning the hand model's palm to the depth image."""
import numpy as np

from g1tele.hand import hand_rotation
from g1tele.perception.camera import Intrinsics
from g1tele.perception.hand_align import align_hand, aligned_landmarks, blend, kabsch
from g1tele.perception.lifting import project
from g1tele.perception.synthetic import synthetic_hand_landmarks

K = Intrinsics(fx=600.0, fy=600.0, cx=424.0, cy=240.0, width=848, height=480)
FACE = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # fingers right, palm towards the camera


def rot(axis, deg):
    a = np.radians(deg)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * (kx @ kx)


def angle(r1, r2):
    return np.degrees(np.arccos(np.clip((np.trace(r1.T @ r2) - 1) / 2, -1, 1)))


def scene(R, side='right', wrist=(0.1, 0.0, 1.0), noise=0.002, seed=0):
    """True landmarks (absolute, camera axes), their image points, and a depth image of the palm plate."""
    rng = np.random.default_rng(seed)
    pts = synthetic_hand_landmarks(R, np.zeros(6), side) + np.asarray(wrist)
    image = np.array([project(p, K) for p in pts])
    depth = np.full((K.height, K.width), 5.0, np.float32)
    n_face = R[:, 2] if R[2, 2] < 0 else -R[:, 2]
    for x in np.arange(-0.03, 0.21, 0.002):
        for y in np.arange(-0.05, 0.0501, 0.002):
            p = np.asarray(wrist) + R[:, 0] * x + R[:, 1] * y + 0.01 * n_face
            u, v = project(p, K)
            depth[int(round(v)), int(round(u))] = p[2]
    m = depth < 5.0
    depth[m] += rng.normal(0, noise, m.sum()).astype(np.float32)
    return pts, image, depth


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    r_true = rot([1, 2, 3], 40)
    r, rms = kabsch(a - a.mean(0), (a @ r_true.T) - (a @ r_true.T).mean(0))
    assert angle(r, r_true) < 1e-3 and rms < 1e-9


def test_blend_goes_a_fraction_of_the_way():
    r = rot([0, 0, 1], 50)
    assert abs(angle(np.eye(3), blend(r, 0.5)) - 25) < 1e-6
    assert angle(np.eye(3), blend(r, 0.0)) < 1e-9 and angle(r, blend(r, 1.0)) < 1e-6


def test_a_mistilted_hand_model_is_rotated_back_onto_the_measured_palm():
    errs_before, errs_after = [], []
    for tilt in (0, 20, 40):
        for roll in (-30, 0, 30):
            R = rot([0, 1, 0], tilt) @ rot([1, 0, 0], roll) @ FACE
            pts, image, depth = scene(R)
            wrong = rot([1, 1, 0], 25) @ (pts - pts[0]) .T            # the model's guess is 25 degrees off
            wrong = wrong.T + pts[0]
            true_frame = hand_rotation(pts, 'right')
            al = align_hand(depth, K, image, wrong)
            assert al is not None and al.confidence > 0.3 and not al.mirrored
            fixed = aligned_landmarks(al, 1.0, 60.0)
            errs_before.append(angle(hand_rotation(wrong, 'right'), true_frame))
            errs_after.append(angle(hand_rotation(fixed, 'right'), true_frame))
    assert np.mean(errs_before) > 20
    assert np.max(errs_after) < 8.0, errs_after


def test_a_hand_that_came_back_mirrored_in_depth_gets_the_measured_palm_frame():
    R = rot([0, 1, 0], 35) @ rot([1, 0, 0], 20) @ FACE
    pts, image, depth = scene(R)
    mirrored = (pts - pts[0]) * np.array([1.0, 1.0, -1.0]) + pts[0]
    true_frame = hand_rotation(pts, 'right')
    assert angle(hand_rotation(mirrored, 'right'), true_frame) > 40
    al = align_hand(depth, K, image, mirrored)
    # a flat palm mirrored along the viewing axis can be rotated onto the same measured patch, so which candidate wins
    # does not matter: what counts is that the palm frame that comes out is the measured one
    assert al is not None and al.confidence > 0.5
    assert angle(hand_rotation(aligned_landmarks(al, 1.0, 60.0), 'right'), true_frame) < 10.0


def test_a_correct_hand_is_left_alone():
    R = rot([0, 1, 0], 30) @ FACE
    pts, image, depth = scene(R)
    al = align_hand(depth, K, image, pts)
    assert al is not None and not al.mirrored
    assert angle(hand_rotation(aligned_landmarks(al, 0.8, 60.0), 'right'), hand_rotation(pts, 'right')) < 6.0


def test_no_depth_means_no_change():
    R = rot([0, 1, 0], 30) @ FACE
    pts, image, depth = scene(R)
    assert align_hand(np.zeros_like(depth), K, image, pts) is None


def test_a_bad_fit_has_low_confidence_and_an_implausible_correction_is_ignored():
    R = rot([0, 1, 0], 30) @ FACE
    pts, image, depth = scene(R, noise=0.04)                       # a very noisy depth image
    al = align_hand(depth, K, image, pts)
    assert al is None or al.confidence < 0.4
    pts, image, depth = scene(R)
    al = align_hand(depth, K, image, pts)
    al.R, al.confidence = rot([0, 0, 1], 100), 0.5
    np.testing.assert_allclose(aligned_landmarks(al, 1.0, 60.0), al.world)    # 100 degrees > the 60 allowed, fit not excellent
    al.confidence = 0.9
    assert not np.allclose(aligned_landmarks(al, 1.0, 60.0), al.world)        # an excellent fit may undo a mirrored hand
