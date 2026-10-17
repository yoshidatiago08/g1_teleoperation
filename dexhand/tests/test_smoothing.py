"""The uncertainty-aware pose smoother, the fit's per-frame covariance, and the jitter report."""
import numpy as np
import pytest

pytest.importorskip('scipy')

from dex_hand import analysis, frames, record
from dex_hand import hand_model as hm
from dex_hand.frames import rotation_angle, rotation_from_vec
from dex_hand.smoothing import LandmarkFilter, PoseSmoother
from dex_hand.sources import Observation
from dex_hand.synthetic import FACING_CAMERA, make_hand

from test_hand_model import F, CX, CY, HAND_LENGTH, WRIST, endon, observe

DT = 1 / 30
ISO_ROT, ISO_POS = np.eye(3) * np.radians(3) ** 2, np.eye(3) * 0.005 ** 2


def run(smoother, rotations, positions, cov_rot=ISO_ROT, cov_pos=ISO_POS):
    return [smoother.update(r, p, cov_rot, cov_pos, DT) for r, p in zip(rotations, positions)]


def test_noise_around_a_steady_hand_is_reduced():
    rng = np.random.default_rng(0)
    rots = [rotation_from_vec(rng.normal(0, np.radians(3), 3)) for _ in range(150)]
    pos = [rng.normal(0, 0.005, 3) + [0, 0, 0.7] for _ in range(150)]
    out = run(PoseSmoother(), rots, pos)
    rough = lambda rs: np.nanmedian(analysis.roughness(rs[30:]))
    assert rough([o[0] for o in out]) < 0.5 * rough(rots)
    assert np.nanmedian(analysis.position_roughness([o[1] for o in out][30:])) < 0.6 * np.nanmedian(analysis.position_roughness(pos[30:]))


def test_a_weakly_observed_axis_is_smoothed_more_than_a_well_observed_one():
    rng = np.random.default_rng(1)
    cov = np.diag(np.radians([2.0, 2.0, 30.0]) ** 2)                        # about z (the weak axis) the image says little
    rots = [rotation_from_vec(rng.normal(0, 1, 3) * np.sqrt(np.diag(cov))) for _ in range(300)]
    out = [o[0] for o in run(PoseSmoother(), rots, [np.zeros(3)] * 300, cov_rot=cov)][60:]
    spread = lambda rs: np.std([frames.rotvec(r) for r in rs], axis=0)
    gain = spread(out) / spread(rots[60:])
    assert gain[2] < 0.85 * gain[0] and gain[2] < 0.5                        # follows the motion model where the image says little


def test_a_moving_hand_is_followed_with_little_lag():
    t = np.arange(120) * DT
    angle = 1.2 * np.sin(2 * np.pi * t / 1.6)                               # turns +-70 degrees in 0.8 s
    rots = [rotation_from_vec([0, a, 0]) for a in angle]
    pos = [np.array([0.1 * np.sin(2 * np.pi * x / 1.6), 0, 0.7]) for x in t]
    out = run(PoseSmoother(), rots, pos, cov_rot=np.eye(3) * np.radians(1) ** 2, cov_pos=np.eye(3) * 0.002 ** 2)
    err = [np.degrees(rotation_angle(o[0], r)) for o, r in zip(out[20:], rots[20:])]
    assert np.median(err) < 3.0 and max(err) < 10.0
    assert np.max([np.linalg.norm(o[1] - p) for o, p in zip(out[20:], pos[20:])]) < 0.02


def test_a_single_flipped_frame_is_ignored_but_a_real_jump_is_followed():
    s = PoseSmoother()
    run(s, [np.eye(3)] * 40, [np.zeros(3)] * 40)
    flipped = rotation_from_vec([0, np.radians(120), 0])
    r, _ = s.update(flipped, np.zeros(3), ISO_ROT, ISO_POS, DT)
    assert np.degrees(rotation_angle(r, np.eye(3))) < 25                    # one bad frame hardly moves it
    for _ in range(12):
        r, _ = s.update(flipped, np.zeros(3), ISO_ROT, ISO_POS, DT)
    assert np.degrees(rotation_angle(r, flipped)) < 5                       # the hand really is there now


def test_without_a_covariance_the_measurement_passes_through():
    s = PoseSmoother()
    r = rotation_from_vec([0.2, 0.1, 0.0])
    out = s.update(r, np.ones(3), None, None, DT)
    assert np.allclose(out[0], r) and np.allclose(out[1], 1.0)


@pytest.fixture(scope='module')
def shape():
    rng = np.random.default_rng(1)
    flat = make_hand('right', [0.0] * 5)
    samples = [observe(rng, flat, np.array([0.0, 0.0, 0.7])) for _ in range(20)]
    return hm.HandShape.from_samples('right', [s[0] for s in samples], hand_length=HAND_LENGTH)


def sigma(shape, rotation, curl=0.0):
    rng, fit, true = np.random.default_rng(2), hm.HandFit(shape), None
    true = make_hand('right', [curl] * 5, rotation)
    out = []
    for _ in range(12):
        w, p = observe(rng, true, WRIST)
        res = fit.update(w, p, None, F, CX, CY)
    return res


def test_the_fit_knows_that_a_flat_palm_facing_the_camera_is_poorly_determined(shape):
    face_on = sigma(shape, FACING_CAMERA)
    tilted = sigma(shape, rotation_from_vec([1.05, 0, 0]) @ FACING_CAMERA)
    assert face_on.ok and tilted.ok
    assert face_on.rot_sigma_deg > 1.5 * tilted.rot_sigma_deg
    assert face_on.cov_rot.shape == (3, 3) and face_on.cov_pos.shape == (3, 3)
    assert np.sqrt(face_on.cov_pos[2, 2]) > np.sqrt(face_on.cov_pos[0, 0])    # the distance to the camera is the weak position axis


def test_the_report_has_the_jitter_numbers_by_viewing_angle(tmp_path):
    rng = np.random.default_rng(3)
    rec = record.Recorder(str(tmp_path / 'take.npz'))
    t = 1000.0
    flat = make_hand('right', [0.0] * 5)
    for _ in range(48):
        w, p = observe(rng, flat, np.array([0.0, 0.0, 0.7]))
        rec.add(Observation(w, p, 0.7, None), F, CX, CY, None, True, t)
        t += DT
    for rotation in (FACING_CAMERA, endon((0.2, 0.15)), rotation_from_vec([1.05, 0, 0]) @ FACING_CAMERA):
        hand = make_hand('right', [0.2] * 5, rotation)
        for _ in range(45):
            w, p = observe(rng, hand, WRIST)
            rec.add(Observation(w, p, None, None), F, CX, CY, None, False, t)
            t += DT
    rec.save()
    result = analysis.analyse(record.load(tmp_path / 'take.npz'), 'right', hand_length=HAND_LENGTH)
    assert np.nanmedian(result['rough_smooth_deg']) < 0.8 * np.nanmedian(result['rough_model_deg'])
    assert 0 <= result['lag_ms'] <= 150
    text = analysis.report(result)
    for needle in ('JITTER', 'smoother lag', 'by palm normal vs the camera axis', 'by finger axis vs the camera axis', 'uncertainty'):
        assert needle in text, needle
    analysis.plot(result, str(tmp_path / 'take.png'))


def landmark_series(rng, n, spikes=False, moving=False):
    """(truth pixels, noisy pixels, noisy world) frames of a hand, optionally drifting across the image and with single-frame landmark glitches."""
    truth, noisy, worlds = [], [], []
    base = make_hand('right', [0.2] * 5)
    for k in range(n):
        offset = np.array([0.1 * np.sin(2 * np.pi * k / 30), 0.0, 0.0]) if moving else np.zeros(3)
        pts = base + WRIST + offset
        px = np.column_stack([CX + F * pts[:, 0] / pts[:, 2], CY + F * pts[:, 1] / pts[:, 2]])
        noise = px + rng.normal(0, 1.5, px.shape)
        if spikes and k % 9 == 4:
            noise[rng.integers(1, 21)] += rng.normal(0, 30, 2)
        truth.append(px), noisy.append(noise), worlds.append(base + rng.normal(0, 0.005, base.shape))
    return truth, noisy, worlds


def landmark_error(truth, pixels):
    return np.median(np.linalg.norm(np.array(truth[20:]) - np.array(pixels[20:]), axis=2))


def test_the_landmark_filter_steadies_noisy_landmarks_and_ignores_a_single_frame_glitch():
    for spikes in (False, True):
        truth, noisy, worlds = landmark_series(np.random.default_rng(4), 150, spikes=spikes)
        f = LandmarkFilter()
        filtered = [f.update(w, p, DT)[1] for w, p in zip(worlds, noisy)]
        assert landmark_error(truth, filtered) < 0.75 * landmark_error(truth, noisy)
        if spikes:
            worst = max(np.linalg.norm(np.array(truth[k]) - filtered[k], axis=1).max() for k in range(20, 150) if k % 9 == 4)
            assert worst < 15.0                                              # a 30 px glitch does not get through


def test_the_landmark_filter_follows_a_moving_hand_closely():
    truth, noisy, worlds = landmark_series(np.random.default_rng(5), 150, moving=True)
    f = LandmarkFilter()
    filtered = [f.update(w, p, DT)[1] for w, p in zip(worlds, noisy)]
    assert landmark_error(truth, filtered) < 1.5 * landmark_error(truth, noisy) + 0.5


def test_a_landmark_that_really_moves_is_taken_one_frame_later():
    truth, noisy, worlds = landmark_series(np.random.default_rng(6), 40)
    f = LandmarkFilter()
    for k in range(30):
        f.update(worlds[k], noisy[k], DT)
    moved = np.array(noisy[30]).copy()
    moved[8] += [40.0, 0.0]                                                  # the index fingertip really goes 40 px away and stays
    first = f.update(worlds[30], moved, DT)[1][8]
    second = f.update(worlds[31], moved, DT)[1][8]
    assert abs(first[0] - noisy[30][8][0]) < 15                              # not believed on the first frame
    assert abs(second[0] - moved[8][0]) < 5                                  # confirmed on the second
