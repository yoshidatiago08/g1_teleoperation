"""Offline study of a recording (record.py): the hand model against the raw MediaPipe landmarks, frame by frame."""
import time

import numpy as np

from dex_hand import frames
from dex_hand.hand_model import HandFit, HandShape
from dex_hand.record import load


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
    fitter, last_t = HandFit(shape), None
    raw_ref, model_ref = [], []
    for i in seen:
        t = data['t'][i]
        if last_t is not None and t - last_t > 0.5:
            fitter.reset()
        last_t = t
        focal, cx, cy = data['intrinsics'][i]
        world, pixels = data['world'][i], data['pixels'][i]
        depth = None if np.isnan(data['depth'][i]) else float(data['depth'][i])
        start = time.perf_counter()
        fit = fitter.update(world, pixels, depth, focal, cx, cy)
        elapsed = time.perf_counter() - start
        raw_r = frames.hand_rotation(world, side)
        raw_depth = depth if depth is not None else frames.depth_from_scale(world, pixels, focal)
        rows.append({'i': int(i), 't': float(t), 'ok': fit.ok, 'error_px': fit.error_px, 'z_error_cm': 100 * fit.z_error,
                     'visible': int(fit.visible.sum()), 'ms': 1000 * elapsed, 'reinitialised': fit.reinitialised,
                     'model_z': float(fit.wrist[2]), 'raw_z': np.nan if raw_depth is None else float(raw_depth),
                     'model_r': fit.rotation, 'raw_r': raw_r, 'calibrating': bool(data['calibrating'][i]),
                     'tips': fit.tip_vectors, 'raw_tips': None if raw_r is None else frames.tip_vectors(world, raw_r)})
    out = {k: np.array([r[k] for r in rows]) for k in ('i', 't', 'ok', 'error_px', 'z_error_cm', 'visible', 'ms', 'reinitialised', 'model_z', 'raw_z')}
    # compare the two rotations relative to their own value at the calibration (a constant frame offset does not matter)
    cal_rows = [r for r in rows if r['calibrating'] and r['ok'] and r['raw_r'] is not None]
    if cal_rows:
        offset = frames.mean_rotation([r['model_r'].T @ r['raw_r'] for r in cal_rows])
        out['rotation_difference_deg'] = np.array([np.degrees(frames.rotation_angle(r['model_r'] @ offset, r['raw_r']))
                                                   if r['raw_r'] is not None and r['ok'] else np.nan for r in rows])
    out['tip_difference_mm'] = np.array([1000 * np.linalg.norm(r['tips'] - r['raw_tips'], axis=1).mean()
                                         if r['raw_tips'] is not None and r['ok'] else np.nan for r in rows])
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
    if 'rotation_difference_deg' in result:
        d = result['rotation_difference_deg']
        lines.insert(6, f'palm rotation, model vs raw (relative to the calibration): median {pct(d, 50):.0f} deg, 95th {pct(d, 95):.0f} deg')
    return '\n'.join(lines)


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = result['t'] - result['t'][0]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, result['error_px']); ax[0].set_ylabel('fit error (px)')
    ax[1].plot(t, result['z_error_cm']); ax[1].set_ylabel('depth disagreement (cm)')
    ax[2].plot(t, 100 * result['model_z'], label='model'); ax[2].plot(t, 100 * result['raw_z'], label='raw / sensor', alpha=0.6)
    ax[2].set_ylabel('distance (cm)'); ax[2].legend()
    if 'rotation_difference_deg' in result:
        ax[3].plot(t, result['rotation_difference_deg'])
    ax[3].set_ylabel('model vs raw rotation (deg)'); ax[3].set_xlabel('s')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
