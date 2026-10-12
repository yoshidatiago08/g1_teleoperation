import numpy as np

from g1tele.perception.gestures import NodDetector


def run(signal, fps=30.0, width=40.0, **kw):
    """signal(t) -> (dx, dy) of the nose in head widths. Returns the times at which the detector fired."""
    det = NodDetector(**kw)
    fired = []
    for i in range(int(8 * fps)):
        t = i / fps
        dx, dy = signal(t)
        if det.update(t, np.array([300.0 + dx * width, 200.0 + dy * width]), width):
            fired.append(round(t, 2))
    return fired


def nods(starts, depth=0.25, dur=0.5):
    def f(t):
        for s in starts:
            if s <= t < s + dur:
                return 0.0, depth * np.sin(np.pi * (t - s) / dur)      # down and back up
        return 0.0, 0.0
    return f


def test_two_nods_fire_once_when_the_second_completes():
    fired = run(nods([2.0, 2.8]))
    assert len(fired) == 1 and 3.2 < fired[0] < 3.5


def test_one_nod_does_not_fire():
    assert run(nods([2.0])) == []


def test_nods_too_far_apart_do_not_fire():
    assert run(nods([1.5, 4.0])) == []


def test_small_movements_do_not_fire():
    assert run(nods([2.0, 2.8], depth=0.08)) == []


def test_shaking_the_head_does_not_fire():
    def shake(t):
        return (0.3 * np.sin(2 * np.pi * 1.5 * t), 0.02 * np.sin(2 * np.pi * 1.5 * t)) if 2 < t < 5 else (0.0, 0.0)
    assert run(shake) == []


def test_slow_leaning_and_looking_down_do_not_fire():
    def lean(t):
        return 0.0, 0.6 * min(1.0, max(0.0, (t - 2.0) / 3.0))          # looks down at the table over 3 s
    assert run(lean) == []


def test_cooldown_after_a_switch_ignores_the_next_nods():
    fired = run(nods([2.0, 2.8, 3.8, 4.6]))
    assert len(fired) == 1


def test_a_nod_with_noise_still_fires():
    rng = np.random.default_rng(0)
    base = nods([2.0, 2.8])
    fired = run(lambda t: (rng.normal(0, 0.02), base(t)[1] + rng.normal(0, 0.03)))
    assert len(fired) == 1
