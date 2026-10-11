"""Debug overlay for the operator's window: landmarks, tracking status, calibration prompt, timings."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from g1tele import protocol
from g1tele.protocol import OperatorFrame

COLORS = {protocol.TRACKING: (60, 220, 60), protocol.HELD: (0, 190, 255), protocol.LOST: (60, 60, 240),
          protocol.CALIBRATING: (0, 220, 255), protocol.IDLE: (160, 160, 160)}


def draw(image: np.ndarray, frame: OperatorFrame, debug: dict, feedback: Optional[dict] = None) -> np.ndarray:
    img = image.copy()
    meas = debug.get('measures') or {}
    for side, parts in meas.items():
        status = debug.get('status', {}).get(side, protocol.CALIBRATING if frame.mode == protocol.CALIBRATING
                                            else protocol.IDLE)
        color = COLORS.get(status, (200, 200, 200))
        pts = {}
        for key in 'SEW':
            m = parts.get(key)
            if m is None:
                continue
            u, v = int(m.uv[0]), int(m.uv[1])
            pts[key] = (u, v)
            ok = m.vis >= 0.5
            cv2.circle(img, (u, v), 7 if key == 'W' else 5, color if ok else (120, 120, 120), -1 if ok else 2)
            if m.P is None and ok:
                cv2.circle(img, (u, v), 11, (0, 0, 255), 1)      # visible but no depth
        for a, b in (('S', 'E'), ('E', 'W')):
            if a in pts and b in pts:
                cv2.line(img, pts[a], pts[b], color, 2)
        if 'W' in pts:
            cv2.putText(img, side[0].upper(), (pts['W'][0] + 8, pts['W'][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # What the tracker concluded (fused 3D arm, re-projected onto the image, yellow) against the raw 2D
    # landmarks (the coloured lines above): they should lie on top of each other. And the hand frame
    # at each wrist: red = fingers, blue = palm normal (out of the palm), green = the third axis.
    k = debug.get('intrinsics')
    if k is not None:
        def px(p3):
            return None if p3 is None or p3[2] <= 0.05 else (int(k.fx * p3[0] / p3[2] + k.cx), int(k.fy * p3[1] / p3[2] + k.cy))
        for side, pts3 in (debug.get('fused') or {}).items():
            if pts3 is None:
                continue
            s3, e3, w3 = pts3
            a, b, c = px(s3), px(e3), px(w3)
            for p0, p1 in ((a, b), (b, c)):
                if p0 and p1:
                    cv2.line(img, p0, p1, (0, 230, 255), 1, cv2.LINE_AA)
            r_cam = (debug.get('hand_axes') or {}).get(side)
            if r_cam is not None and c is not None and w3 is not None:
                for col, bgr in ((0, (0, 0, 255)), (1, (0, 200, 0)), (2, (255, 80, 0))):
                    tip = px(w3 + 0.09 * r_cam[:, col])
                    if tip:
                        cv2.arrowedLine(img, c, tip, bgr, 2, tipLength=0.25)
    for side, points in (debug.get('hands') or {}).items():
        if points is not None:
            for u, v in points:
                cv2.circle(img, (int(u), int(v)), 2, (255, 0, 255), -1)
    y = 24
    mode_color = COLORS.get(frame.mode, (255, 255, 255))
    cv2.putText(img, frame.mode, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    if frame.prompt:
        y += 28
        warn = frame.prompt.startswith('WARNING')
        cv2.putText(img, frame.prompt[:110], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 60, 255) if warn else (0, 255, 255), 2)
    for side, arm in frame.arms.items():
        y += 24
        cv2.putText(img, f'{side[0].upper()}: {arm.status} q={arm.quality:.2f}'
                    + ('  hand' if arm.hand_R is not None else ''), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(arm.status, (255, 255, 255)), 1)
    t = frame.timing_ms
    h = img.shape[0]
    line = 'perception ' + ' '.join(f'{k} {t[k]:.1f}' for k in ('pose', 'lift', 'track', 'hands', 'total') if k in t) + ' ms'
    cv2.putText(img, line, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if feedback:
        st = feedback['status']
        fb = (f"robot: in {st.get('fps_in', 0)} fps  apply {st.get('apply_ms', 0)} ms  "
              + ('REC ' if st.get('recording') else '') + ' '.join(
                  f"{s[0].upper()} ik {e} mm" for s, e in st.get('ik_error_mm', {}).items()))
        cv2.putText(img, fb, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        sc = st.get('scene') or {}
        if sc:
            text = f"cylinder -> target {sc['to_target_cm']:.1f} cm   lifted {sc['lifted_cm']:.1f} cm"
            if sc.get('on_target'):
                text += '   ON TARGET'
            cv2.putText(img, text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
    return img
