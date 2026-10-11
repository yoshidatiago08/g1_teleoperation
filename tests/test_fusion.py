import numpy as np

from g1tele.perception.fusion import FusionParams, fuse_arm

LU, LF = 0.30, 0.27


def arm(rng):
    s = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.1), 1.8])
    while True:
        u, f = rng.normal(size=3), rng.normal(size=3)
        u, f = u / np.linalg.norm(u), f / np.linalg.norm(f)
        e, w = s + LU * u, s + LU * u + LF * f
        if e[2] > 0.8 and w[2] > 0.8:
            return s, e, w


def test_exact_inputs_give_exact_output():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s, e, w = arm(rng)
        res = fuse_arm(s, e / np.linalg.norm(e), np.linalg.norm(e), w / np.linalg.norm(w),
                       np.linalg.norm(w), LU, LF)
        np.testing.assert_allclose(res.wrist, w, atol=1e-4)
        np.testing.assert_allclose(res.elbow, e, atol=1e-4)


def test_picks_the_right_root_when_depth_is_a_few_cm_off():
    """A reading 5 cm off (a hand covering the wrist) must still select the right one of the two
    intersections of the wrist ray with the bone sphere."""
    rng = np.random.default_rng(1)
    wrong = 0
    for _ in range(200):
        s, e, w = arm(rng)
        res = fuse_arm(s, e / np.linalg.norm(e), np.linalg.norm(e) - 0.04, w / np.linalg.norm(w),
                       np.linalg.norm(w) - 0.05, LU, LF)
        wrong += np.linalg.norm(res.wrist - w) > 0.15
    assert wrong <= 4


def test_missing_depth_falls_back_to_bones_and_previous_frame():
    rng = np.random.default_rng(2)
    s, e, w = arm(rng)
    prev = (np.linalg.norm(e), np.linalg.norm(w))
    res = fuse_arm(s, e / np.linalg.norm(e), None, w / np.linalg.norm(w), None, LU, LF, prev)
    assert np.linalg.norm(res.wrist - w) < 0.02 and np.linalg.norm(res.elbow - e) < 0.02


def test_no_elbow_uses_wrist_depth_and_clamps_to_reach():
    s = np.array([0.0, 0.0, 1.8])
    w = np.array([0.0, 0.0, 1.8 - 0.4])
    res = fuse_arm(s, None, None, w / np.linalg.norm(w), np.linalg.norm(w), LU, LF)
    assert res.elbow is None and np.linalg.norm(res.wrist - w) < 1e-6
    far = np.array([0.0, 0.0, 1.8 + 1.0])
    res = fuse_arm(s, None, None, far / np.linalg.norm(far), np.linalg.norm(far), LU, LF)
    assert np.linalg.norm(res.wrist - s) < 1.1 * (LU + LF)
