"""Offline study of a recording (record.py): the hand model against the raw MediaPipe landmarks, frame by frame."""
import time

import numpy as np

from dex_hand import frames
from dex_hand.hand_model import HandFit, HandShape
from dex_hand.record import load
from dex_hand.smoothing import LandmarkFilter, PoseSmoother

LANDMARK_NAMES = ['wrist', 'thumb cmc', 'thumb mcp', 'thumb ip', 'thumb tip', 'index mcp', 'index pip', 'index dip', 'index tip', 'middle mcp',
                  'middle pip', 'middle dip', 'middle tip', 'ring mcp', 'ring pip', 'ring dip', 'ring tip', 'pinky mcp', 'pinky pip', 'pinky dip', 'pinky tip']

ANGLE_BUCKETS = ((0, 30), (30, 60), (60, 91))      # degrees


def roughness(rotations):
    """Per frame, how far (degrees) a rotation is from the midpoint of its two neighbours: 0 for steady motion, large for jitter.
    NaN where a neighbour is missing. A single noisy frame shows up as a spike."""
    out = np.full(len(rotations), np.nan)
    for k in range(1, len(rotations) - 1):
        a, b, c = rotations[k - 1], rotations[k], rotations[k + 1]
        if a is None or b is None or c is None:
            continue
        mid = frames.rotation_from_vec(0.5 * frames.rotvec(c @ a.T)) @ a
        out[k] = np.degrees(frames.rotation_angle(b, mid))
    return out


def position_roughness(positions):
    """Per frame (mm): distance of a position from the midpoint of its neighbours."""
    p = np.asarray(positions, dtype=float)
    out = np.full(len(p), np.nan)
    out[1:-1] = 1000 * np.linalg.norm(p[1:-1] - 0.5 * (p[:-2] + p[2:]), axis=1)
    return out


def _landmark_roughness(pixels):
    p = np.asarray(pixels, dtype=float)
    out = np.full(p.shape[:2], np.nan)
    out[1:-1] = np.linalg.norm(p[1:-1] - 0.5 * (p[:-2] + p[2:]), axis=2)
    return out


def lag_ms(reference, delayed, t):
    """How many ms `delayed` (the smoothed rotations) trails `reference` (the unsmoothed ones): the shift in frames that matches best."""
    best, best_err = 0, np.inf
    for s in range(0, 8):
        pairs = [(reference[k - s], delayed[k]) for k in range(s, len(reference)) if reference[k - s] is not None and delayed[k] is not None]
        if len(pairs) < 10:
            continue
        err = np.mean([frames.rotation_angle(a, b) for a, b in pairs])
        if err < best_err:
            best, best_err = s, err
    return 1000.0 * best * float(np.median(np.diff(t))) if len(t) > 1 else 0.0


def analyse(data, side='right', hand_length=None, calibration=None):
    """`data`: a loaded recording. `calibration`: (first, last) frame indices of the stretched palm-to-camera hand
    (default: the frames flagged while the operator was calibrating). Returns a dict of per-frame arrays and a summary."""
    seen = np.where(data['seen'])[0]
    if calibration is None:
        cal = [i for i in seen if data['calibrating'][i]]
    else:
        cal = [i for i in seen if calibration[0] <= i <= calibration[1]]
    if len(cal) < 10:
        raise SystemExit('no calibration frames: press c in the recording, or pass --calibration FIRST:LAST')
    focal, cx, cy = data['intrinsics'][cal[0]]
    depths = [None if np.isnan(data['depth'][i]) else float(data['depth'][i]) for i in cal]
    shape = HandShape.from_samples(side, [data['world'][i] for i in cal], [data['pixels'][i] for i in cal],
                                   depths if any(d is not None for d in depths) else None, focal, cx, cy, hand_length)
    if shape is None:
        raise SystemExit('the calibration frames have no usable hand')
    rows = []
    fitter, last_t, smoother, landmarks = HandFit(shape), None, PoseSmoother(), LandmarkFilter()
    pix_raw, pix_filtered = [], []
    raw_ref, model_ref = [], []
    for i in seen:
        t = data['t'][i]
        if last_t is not None and t - last_t > 0.5:
            fitter.reset()
            smoother.reset()
            landmarks.reset()
        dt = None if last_t is None else t - last_t
        last_t = t
        focal, cx, cy = data['intrinsics'][i]
        raw_world, raw_pixels = data['world'][i], data['pixels'][i]
        world, pixels = landmarks.update(raw_world, raw_pixels, dt)                  # stage 1: MediaPipe's landmarks, filtered
        pix_raw.append(raw_pixels.copy())
        pix_filtered.append(pixels.copy())
        depth = None if np.isnan(data['depth'][i]) else float(data['depth'][i])
        start = time.perf_counter()
        fit = fitter.update(world, pixels, depth, focal, cx, cy)
        elapsed = time.perf_counter() - start
        if fit.ok:
            if fit.reinitialised:
                smoother.reset()
            smooth_r, smooth_p = smoother.update(fit.rotation, fit.wrist, fit.cov_rot, fit.cov_pos, dt)
        else:
            smoother.reset()
            smooth_r = smooth_p = None
        raw_r = frames.hand_rotation(raw_world, side)
        raw_depth = depth if depth is not None else frames.depth_from_scale(raw_world, raw_pixels, focal)
        rows.append({'i': int(i), 't': float(t), 'ok': fit.ok, 'error_px': fit.error_px, 'z_error_cm': 100 * fit.z_error,
                     'visible': int(fit.visible.sum()), 'ms': 1000 * elapsed, 'reinitialised': fit.reinitialised,
                     'model_z': float(fit.wrist[2]), 'raw_z': np.nan if raw_depth is None else float(raw_depth),
                     'model_r': fit.rotation if fit.ok else None, 'raw_r': raw_r, 'smooth_r': smooth_r, 'smooth_p': smooth_p,
                     'model_p': fit.wrist if fit.ok else None, 'sigma_deg': fit.rot_sigma_deg if fit.ok else np.nan,
                     'normal_deg': np.degrees(np.arccos(min(abs(fit.rotation[2, 2]), 1.0))) if fit.ok else np.nan,
                     'finger_deg': np.degrees(np.arccos(min(abs(fit.rotation[2, 0]), 1.0))) if fit.ok else np.nan, 'calibrating': bool(data['calibrating'][i]),
                     'tips': fit.tip_vectors, 'raw_tips': None if raw_r is None else frames.tip_vectors(raw_world, raw_r)})
    out = {k: np.array([r[k] for r in rows]) for k in ('i', 't', 'ok', 'error_px', 'z_error_cm', 'visible', 'ms', 'reinitialised', 'model_z', 'raw_z')}
    # compare the two rotations relative to their own value at the calibration (a constant frame offset does not matter)
    cal_rows = [r for r in rows if r['calibrating'] and r['ok'] and r['raw_r'] is not None and r['model_r'] is not None]
    if cal_rows:
        offset = frames.mean_rotation([r['model_r'].T @ r['raw_r'] for r in cal_rows])
        out['rotation_difference_deg'] = np.array([np.degrees(frames.rotation_angle(r['model_r'] @ offset, r['raw_r']))
                                                   if r['raw_r'] is not None and r['ok'] else np.nan for r in rows])
    out['tip_difference_mm'] = np.array([1000 * np.linalg.norm(r['tips'] - r['raw_tips'], axis=1).mean()
                                         if r['raw_tips'] is not None and r['ok'] else np.nan for r in rows])
    t_all = np.array([r['t'] for r in rows])
    out['sigma_deg'] = np.array([r['sigma_deg'] for r in rows])
    out['normal_deg'] = np.array([r['normal_deg'] for r in rows])
    out['finger_deg'] = np.array([r['finger_deg'] for r in rows])
    out['rough_raw_deg'] = roughness([r['raw_r'] for r in rows])
    out['rough_model_deg'] = roughness([r['model_r'] for r in rows])
    out['rough_smooth_deg'] = roughness([r['smooth_r'] for r in rows])
    out['pos_rough_model_mm'] = position_roughness([r['model_p'] if r['model_p'] is not None else (np.nan,) * 3 for r in rows])
    out['pos_rough_smooth_mm'] = position_roughness([r['smooth_p'] if r['smooth_p'] is not None else (np.nan,) * 3 for r in rows])
    out['lag_ms'] = lag_ms([r['model_r'] for r in rows], [r['smooth_r'] for r in rows], t_all)
    scale = 640.0 / (2.0 * data['intrinsics'][seen[0]][1])
    out['landmark_jitter_raw_px'] = _landmark_roughness(pix_raw) * scale          # frames x 21, pixels at 640 wide
    out['landmark_jitter_filtered_px'] = _landmark_roughness(pix_filtered) * scale
    out['shape'] = shape
    return out


def report(result):
    s, n = result['shape'], len(result['i'])
    ok = result['ok']
    pct = lambda a, q: float(np.nanpercentile(a, q)) if np.isfinite(a).any() else float('nan')
    lines = [f'hand: {100 * s.hand_length:.1f} cm wrist -> middle fingertip ({s.scale_source}), bend at the calibration {np.degrees(s.bend()):.0f} deg',
             f'frames with a hand: {n}; the model followed {100 * ok.mean():.0f}% (unreliable: {int((~ok).sum())}, restarts: {int(result["reinitialised"].sum())})',
             f'fit error (px, 640 wide): median {pct(result["error_px"], 50):.1f}, 95th {pct(result["error_px"], 95):.1f}',
             f'depth disagreement with MediaPipe on visible landmarks (cm): median {pct(result["z_error_cm"], 50):.1f}, 95th {pct(result["z_error_cm"], 95):.1f}',
             f'visible landmarks per frame: median {pct(result["visible"], 50):.0f} of 21, 5th percentile {pct(result["visible"], 5):.0f}',
             f'distance to the camera, model - raw (cm): median {pct(100 * (result["model_z"] - result["raw_z"]), 50):.1f}, '
             f'5th/95th {pct(100 * (result["model_z"] - result["raw_z"]), 5):.1f} / {pct(100 * (result["model_z"] - result["raw_z"]), 95):.1f}',
             f'fingertip vectors, model vs raw: median {pct(result["tip_difference_mm"], 50):.0f} mm, 95th {pct(result["tip_difference_mm"], 95):.0f} mm',
             f'time per frame: median {pct(result["ms"], 50):.0f} ms, 95th {pct(result["ms"], 95):.0f} ms']
    lines += ['', 'JITTER (how far each frame is from the midpoint of its neighbours; smaller = steadier)']
    jr, jf = result['landmark_jitter_raw_px'], result['landmark_jitter_filtered_px']
    lines.append(f'MediaPipe landmarks in the image, px at 640 wide (median over landmarks / 95th): raw {pct(jr, 50):.1f} / {pct(jr, 95):.1f}, '
                 f'filtered {pct(jf, 50):.1f} / {pct(jf, 95):.1f}')
    worst = np.argsort([-pct(jr[:, k], 50) for k in range(21)])[:5]
    lines.append('the landmarks that shake most (median px, raw -> filtered): ' +
                 ', '.join(f'{LANDMARK_NAMES[k]} {pct(jr[:, k], 50):.1f} -> {pct(jf[:, k], 50):.1f}' for k in worst))
    med = lambda a: pct(a, 50)
    lines.append(f'palm rotation, deg (median / 95th): raw MediaPipe {med(result["rough_raw_deg"]):.1f} / {pct(result["rough_raw_deg"], 95):.1f}, '
                 f'model on filtered landmarks {med(result["rough_model_deg"]):.1f} / {pct(result["rough_model_deg"], 95):.1f}, '
                 f'model + pose smoother {med(result["rough_smooth_deg"]):.1f} / {pct(result["rough_smooth_deg"], 95):.1f}   (smoother lag {result["lag_ms"]:.0f} ms)')
    lines.append(f'wrist position, mm (median / 95th): model {med(result["pos_rough_model_mm"]):.1f} / {pct(result["pos_rough_model_mm"], 95):.1f}, '
                 f'model + pose smoother {med(result["pos_rough_smooth_mm"]):.1f} / {pct(result["pos_rough_smooth_mm"], 95):.1f}')
    sig = result['sigma_deg']
    lines.append(f'palm rotation uncertainty from the image (deg, least-observed axis): median {med(sig):.0f}, 95th {pct(sig, 95):.0f}; '
                 f'frames above 10 deg: {100 * np.nanmean(sig > 10):.0f}%, above 20 deg: {100 * np.nanmean(sig > 20):.0f}%')
    for title, key in (('palm normal vs the camera axis  (0 = palm facing the camera, 90 = edge-on)', 'normal_deg'),
                       ('finger axis vs the camera axis   (0 = fingers pointing at the camera, 90 = across the view)', 'finger_deg')):
        lines.append(f'by {title}:')
        lines.append('    angle        frames   uncertainty   jitter raw   jitter model   jitter + pose smoother   (deg, median)')
        for lo, hi in ANGLE_BUCKETS:
            m = (result[key] >= lo) & (result[key] < hi)
            if m.sum() < 5:
                continue
            lines.append(f'    {lo:2d}-{min(hi, 90):2d} deg   {int(m.sum()):6d}   {pct(sig[m], 50):9.0f}   {pct(result["rough_raw_deg"][m], 50):10.1f}   '
                         f'{pct(result["rough_model_deg"][m], 50):12.1f}   {pct(result["rough_smooth_deg"][m], 50):20.1f}')
    lines.append('')
    if 'rotation_difference_deg' in result:
        d = result['rotation_difference_deg']
        lines.insert(6, f'palm rotation, model vs raw (relative to the calibration): median {pct(d, 50):.0f} deg, 95th {pct(d, 95):.0f} deg')
    return '\n'.join(lines)


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = result['t'] - result['t'][0]
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(t, result['error_px']); ax[0].set_ylabel('fit error (px)')
    ax[1].plot(t, result['z_error_cm']); ax[1].set_ylabel('depth disagreement (cm)')
    ax[2].plot(t, 100 * result['model_z'], label='model'); ax[2].plot(t, 100 * result['raw_z'], label='raw / sensor', alpha=0.6)
    ax[2].set_ylabel('distance (cm)'); ax[2].legend()
    if 'rotation_difference_deg' in result:
        ax[3].plot(t, result['rotation_difference_deg'])
    ax[3].set_ylabel('model vs raw rotation (deg)')
    ax[4].plot(t, result['rough_raw_deg'], label='raw MediaPipe', alpha=0.5); ax[4].plot(t, result['rough_model_deg'], label='model', alpha=0.7)
    ax[4].plot(t, result['rough_smooth_deg'], label='model + smoother'); ax[4].plot(t, result['sigma_deg'] / 5, '--', label='uncertainty / 5', alpha=0.5)
    ax[4].set_ylabel('palm jitter (deg)'); ax[4].set_xlabel('s'); ax[4].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
