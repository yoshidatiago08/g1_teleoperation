import numpy as np

from g1tele.filters import OneEuro, OneEuroRotation
from g1tele.geometry import from_rotvec, rotation_angle


def test_one_euro_smooths_noise_when_still():
    rng = np.random.default_rng(0)
    f = OneEuro(1.5, 20.0)
    raw, out = [], []
    for i in range(300):
        x = np.array([0.5, 0.2, 1.0]) + rng.normal(0, 0.006, 3)
        raw.append(x), out.append(f(x, i / 30))
    raw, out = np.array(raw[60:]), np.array(out[60:])
    assert out.std(axis=0).mean() < 0.65 * raw.std(axis=0).mean()   # beta=20 trades some smoothing for low lag


def test_one_euro_follows_fast_motion_with_small_lag():
    f = OneEuro(1.5, 20.0)
    lag = []
    for i in range(120):
        t = i / 30
        x = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])   # 0.5 Hz, 0.3 m amplitude
        y = f(x, t)
        if i > 30:
            lag.append(abs(y[0] - x[0]))
    assert max(lag) < 0.03     # under 3 cm at ~0.9 m/s peak speed


def test_rotation_filter_converges_and_stays_a_rotation():
    f = OneEuroRotation(2.0, 2.0)
    target = from_rotvec(np.array([0.3, -0.4, 0.2]))
    rng = np.random.default_rng(1)
    for i in range(120):
        noisy = from_rotvec(rng.normal(0, 0.02, 3)) @ target
        out = f(noisy, i / 30)
    assert rotation_angle(out, target) < 0.03
    np.testing.assert_allclose(out @ out.T, np.eye(3), atol=1e-9)
