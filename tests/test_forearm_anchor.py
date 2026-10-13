import numpy as np

from g1tele.filters import OneEuro
from g1tele.perception.tracker import anchor_to_forearm


def rot(axis, deg):
    a = np.radians(deg)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * (kx @ kx)


def frame(f, n_hint=(0, 0, 1.0)):
    f = np.asarray(f, float) / np.linalg.norm(f)
    n = np.asarray(n_hint, float)
    n = n - (n @ f) * f
    n /= np.linalg.norm(n)
    return np.column_stack([f, np.cross(n, f), n])


def filters():
    return [OneEuro(1.0, 1.0) for _ in range(3)]


def angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def test_a_finger_axis_along_the_forearm_is_unchanged():
    a = np.array([1.0, 0.0, 0.0])
    r = frame(a)
    out = anchor_to_forearm(r, a, filters(), 0.0, 0.3, 70.0)
    np.testing.assert_allclose(out, r, atol=1e-9)


def test_the_result_is_a_rotation_and_the_palm_turns_with_the_finger_axis():
    a = np.array([1.0, 0.0, 0.0])
    f = rot([0, 0, 1], 30) @ a
    r = frame(f, (0, 0, 1))
    out = anchor_to_forearm(r, a, filters(), 0.0, 0.3, 70.0)
    assert abs(np.linalg.det(out) - 1) < 1e-9 and np.allclose(out.T @ out, np.eye(3), atol=1e-9)
    assert angle(out[:, 0], a) < angle(f, a)               # pulled toward the forearm
    # the palm normal is still perpendicular to the finger axis and kept its side
    assert abs(out[:, 0] @ out[:, 2]) < 1e-9 and out[2, 2] > 0.9


def test_the_pull_is_the_weight():
    a = np.array([1.0, 0.0, 0.0])
    f = rot([0, 0, 1], 40) @ a
    for w in (0.0, 0.3, 0.6):
        out = anchor_to_forearm(frame(f), a, filters(), 0.0, w, 80.0)
        # the sideways part shrinks by (1 - w)
        side = np.linalg.norm(out[:, 0] - (out[:, 0] @ a) * a)
        want = np.sin(np.radians(40)) * (1 - w)
        assert abs(side / np.linalg.norm(out[:, 0]) - want) < 0.02, (w, side, want)


def test_a_finger_axis_far_from_the_forearm_is_held_inside_the_cone():
    a = np.array([1.0, 0.0, 0.0])
    f = rot([0, 0, 1], 85) @ a
    out = anchor_to_forearm(frame(f), a, filters(), 0.0, 0.0, 60.0)
    assert angle(out[:, 0], a) <= 60.0 + 1e-6


def test_the_sideways_noise_is_smoothed_but_a_real_bend_gets_through():
    a = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    flt = filters()
    raw, anchored = [], []
    for i in range(300):                                    # a steady 15 degree bend with 3 degrees of noise
        f = rot([0, 0, 1], 15 + rng.normal(0, 3)) @ a
        out = anchor_to_forearm(frame(f), a, flt, i / 30.0, 0.0, 70.0)
        if i > 60:
            raw.append(angle(f, a))
            anchored.append(angle(out[:, 0], a))
    assert np.std(anchored) < 0.6 * np.std(raw)             # smoother
    assert abs(np.mean(anchored) - 15) < 2.0                # and unbiased with weight 0
