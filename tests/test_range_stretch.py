import numpy as np

from g1tele.hand import RangeStretcher


def drive(fn, seconds=60.0, fps=30.0, **kw):
    st = RangeStretcher(**kw)
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        out.append(st(t, np.full(6, fn(t))))
    return np.array(out), fps


def test_fingers_that_only_reach_part_of_the_range_are_stretched_to_all_of_it():
    # opens to 0.15 and closes to 0.80 only, once a second: the robot should still open and close fully
    out, fps = drive(lambda t: 0.475 + 0.325 * np.sin(2 * np.pi * t))
    late = out[int(30 * fps):, 0]
    assert late.min() == 0.0 and late.max() == 1.0
    assert abs(np.mean(late) - 0.5) < 0.1                       # and the middle stays in the middle


def test_nothing_happens_before_there_is_enough_history():
    out, fps = drive(lambda t: 0.4 + 0.2 * np.sin(2 * np.pi * t), seconds=5.0)
    x = 0.4 + 0.2 * np.sin(2 * np.pi * np.arange(len(out)) / fps)
    np.testing.assert_allclose(out[:, 0], np.clip((x - 0.06) / 0.88, 0, 1), atol=1e-9)     # only the end snap, no stretch


def test_a_hand_held_half_closed_stays_half_closed():
    out, fps = drive(lambda t: 0.5)
    np.testing.assert_allclose(out[-1], 0.5, atol=0.02)


def test_the_range_is_never_compressed():
    # a full-range signal is left alone (apart from the snap at the ends)
    out, fps = drive(lambda t: 0.5 + 0.5 * np.sin(2 * np.pi * t))
    late = out[int(30 * fps):, 0]
    x = 0.5 + 0.5 * np.sin(2 * np.pi * (np.arange(len(late)) / fps + 30))
    assert np.max(np.abs(late - np.clip((x - 0.06) / 0.88, 0, 1))) < 0.05


def test_values_near_the_ends_snap_to_them():
    st = RangeStretcher(snap=0.06)
    assert st(0.0, np.full(6, 0.03))[0] == 0.0 and st(0.1, np.full(6, 0.97))[0] == 1.0
