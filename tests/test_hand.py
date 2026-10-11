import numpy as np

from g1tele.hand import curls, hand_rotation, raw_features, robot_hand_in_link
from g1tele.perception.synthetic import synthetic_hand_landmarks


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_orientation_is_recovered_for_both_hands():
    rng = np.random.default_rng(0)
    for side in ('right', 'left'):
        base = robot_hand_in_link(side)       # the frame definition itself, per side
        for _ in range(30):
            r = random_rotation(rng)
            pts = synthetic_hand_landmarks(r, np.zeros(6), side)
            got = hand_rotation(pts, side)
            # same definition on both ends: the recovered frame is the true one up to the palm geometry
            assert np.degrees(np.arccos(np.clip((np.trace(got.T @ r) - 1) / 2, -1, 1))) < 8.0
        assert base is not None


def test_each_finger_is_measured_independently():
    rng = np.random.default_rng(1)
    r = random_rotation(rng)
    for finger in range(4):
        c = np.zeros(6)
        c[finger] = 1.0
        got = curls(raw_features(synthetic_hand_landmarks(r, c, 'right')))
        assert got[finger] > 0.85
        assert np.max(np.delete(got[:4], finger)) < 0.1      # the others stay open


def test_curl_roundtrip_error_is_small():
    rng = np.random.default_rng(2)
    err = []
    for _ in range(100):
        c = rng.uniform(0, 1, 6)
        pts = synthetic_hand_landmarks(random_rotation(rng), c, 'left')
        err.append(np.abs(curls(raw_features(pts)) - c))
    assert np.mean(err) < 0.03
